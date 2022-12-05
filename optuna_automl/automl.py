from optuna_automl.pipeline import AutomlPipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import optuna
import copy
import time
import pandas as pd
import dill

class AutoML():

    def __init__(self, cv=5):
        self.cv = cv

        self.best_params = None
        self._ppl = AutomlPipeline(steps=[])
        self.ml_task = None
        self.ml_params = {}

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
        self._ppl.set_params(self.best_params, "main")
        self._ppl.fit(X, y)

    def _objective_func(self, trial, X, y):
        ppl = copy.deepcopy(self._ppl)
        ppl.set_trial_params(trial, "main", self.ml_params)
        score = cross_val_score(ppl, X, y, n_jobs=-1, cv=self.cv, scoring='accuracy', error_score="raise")
        accuracy = score.mean()
        return 1 - accuracy    

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
        
    @staticmethod    
    def load_model(path):
        dill.load(open(path, "rb"))

    def save_model(self, path):
        dill.dump(self, open(path, "wb"))