from optuna_automl.pipeline import AutomlPipeline
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from registry import Registry
import optuna
import copy
import time
import pandas as pd
import dill

class AutoML():

    def __init__(self, cv=5, metric="accuracy", scoring_functions=None):
        self.cv = cv
        self.metric = metric
        self.scoring_functions = scoring_functions

        self.best_params = None
        self._ppl = AutomlPipeline(steps=[])
        self.ml_task = None
        self.ml_params = {}
        self.run_stats = {}

        self._metric = self.metric
        self._scoring_functions = []
        self._scoring_functions.append(self._metric)

    def train(self, data, target, time_budget=10000):
        self._fit(data, target, time_budget)

    def _fit(self, data, target, time_budget=10800):

        ## Cleanup before fitting
        self.best_params = None
        self.ml_params = {}
        self.time_taken = None

        ## Check if pipeline is empty
        if not self.get_pipeline_steps():
            Exception("No components added to pipeline. Add components to pipeline using add_pipe or use available pipelines")

        ## Prepare data
        X, y = self._prepare_data(data, target)

        ## Create ml params
        self.ml_params['ml_task'] = self.ml_task or self.get_ml_task(X, y)

        ## Create optuna study
        study = optuna.create_study(direction="minimize")
        s = time.perf_counter()
        study.optimize(lambda trial: self._objective_func(trial, X, y), timeout=time_budget)
        e = time.perf_counter()
        self.time_taken = e - s

        self.best_params = study.best_params
        self.best_score = self.run_stats[study.best_trial.number]['scores']
        self._ppl.set_params(self.best_params, "main")
        self._ppl.fit(X, y)

    def _objective_func(self, trial, X, y):
        ppl = copy.deepcopy(self._ppl)
        ppl.set_trial_params(trial, "main", self.ml_params)
        scores = cross_validate(estimator=ppl, X=X, y=y, n_jobs=-1, cv=self.cv, scoring=self._scoring_functions, error_score="raise")
        self.run_stats[trial.number] = {"scores": {k:scores[k].mean() for k in scores}, "params": trial.params}
        eval_score = scores[f'test_{self._metric}'].mean()
        return -eval_score

    def predict(self, X):
        return self._label_encoder.inverse_transform(self._ppl.predict(X))

    def get_ml_task(self, X, y):
        return "classification"

    def get_pipeline_steps(self):
        return self._ppl.steps

    def add_pipe(self, name, task):
        self._ppl.add_pipe(name, task)
        return self

    def _prepare_data(self, data, target):
        if not isinstance(data, pd.DataFrame) and not isinstance(data, str):
            raise Exception(f"Dataset of type {type(X)} is not supported. Please pass pandas dataframe or path to csv/excel file.")

        if isinstance(data, str):
            try:
                data = pd.read_excel(data)
            except ValueError:
                data = pd.read_csv(data)
        
        if not isinstance(target, str):
            raise Exception(f"Target should be a string value, not {type(target)}")

        if target not in data.columns:
            raise Exception(f"Target column {target} does not exist in data.")

        X = data.drop(target, axis=1)
        y = data[target].values

        self.ml_task = self.ml_task or self.get_ml_task(X, y)

        if self.ml_task == "classification":
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)

        return X, y

    def get_available_components(task):
        try:
            return Registry.registry[task].keys() 
        except KeyError:
            raise Exception(f"No task with name {task} found")
        
    @staticmethod    
    def load_model(path):
        dill.load(open(path, "rb"))

    def save_model(self, path):
        dill.dump(self, open(path, "wb"))