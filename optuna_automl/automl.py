from optuna_automl.pipeline import AutomlPipeline
from sklearn.model_selection import cross_val_score
import optuna
import copy
import time

class AutoML():

    def __init__(self, cv=5):
        self.cv = cv

        self.best_params = None
        self._ppl = AutomlPipeline(steps=[])
        self.ml_task = None
        self.ml_params = {}
    
    def train(self, data, target, time_budget=10800):
        if not self.get_pipeline_steps():
            Exception("No components added to pipeline. Add components to pipeline using add_pipe or use available pipelines")

        ## Create ml params
        self.ml_task = self.ml_task or self.get_ml_task(data, target)
        self.ml_params['ml_task'] = self.ml_task

        ## Create optuna study
        study = optuna.create_study(direction="minimize")
        s = time.perf_counter()
        study.optimize(lambda trial: self._objective_func(trial, data, target), timeout=time_budget)
        e = time.perf_counter()
        self.time_taken = e - s

        self.best_params = study.best_params
        self._ppl.set_params(self.best_params, "main")
        self._ppl.fit(data, target)

    def _objective_func(self, trial, X, y):
        ppl = copy.deepcopy(self._ppl)
        ppl.set_trial_params(trial, "main", self.ml_params)
        score = cross_val_score(ppl, X, y, n_jobs=-1, cv=self.cv, scoring='accuracy', error_score="raise")
        accuracy = score.mean()
        return 1 - accuracy    

    def predict(self, X):
        pass

    def get_ml_task(self, X, y):
        return "classification"

    def get_pipeline_steps(self):
        return self._ppl.steps

    def add_pipe(self, name, task):
        self._ppl.add_pipe(name, task)
        