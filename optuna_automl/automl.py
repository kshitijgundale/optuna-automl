import pandas as pd
import optuna
from sklearn.model_selection import cross_val_score
from optuna_automl.pipelines.basic_pipeline import basic_pipeline

class AutoML():
    
    def __init__(
        self,
        ml_task='auto',
        time_budget=3600,
        cv=5,
        feat_types=None
    ): 
        self.ml_task = self._ml_task()
        self.time_budget = time_budget
        self.cv = cv
        self.feat_types = feat_types

    def _ml_task(self):
        return self.ml_task 

    def objective(trial, X, y, feat_types, cv):
        ppl = AutomlPipeline(steps=basic_pipeline(X, y, feat_types=feat_types))
        ppl.set_trial_params(trial, "main")
        
        score = cross_val_score(ppl, X, y, n_jobs=-1, cv=cv, scoring='accuracy', error_score="raise")
        accuracy = score.mean()

        return 1 - accuracy

    def train(self, data, target):
        X = data.drop([target], axis=1)
        y = data[target]
        feat_types = self.feat_types
        cv = self.cv

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial : AutoML.objective(trial, X, y, feat_types, cv), timeout=self.time_budget)

        study.best_params

        return self

    def predict(data):
        pass

        