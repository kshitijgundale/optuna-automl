from optuna_automl.automl_choice import AutomlEstimatorChoice, AutomlPreprocessingChoice
from optuna_automl.registry import CLASSIFICATION, FEATURE_PREPROCESSING
import pandas as pd
from optuna_automl.data_preprocessing import DataPreprocessing
from optuna_automl.automl import AutoML
from optuna_automl.pipeline import AutomlPipeline

class BasicPipeline(AutoML):

    def __init__(self, cv=5, ml_task="auto", feat_types=None):

        self._ml_task = ml_task
        self.feat_types = feat_types

        super().__init__(cv=cv)

    def train(self, data, target, time_budget=10000):
        X, y = self._prepare_data(data, target)

        if self._ml_task == "auto":
            self.ml_task = self.get_ml_task(X, y)
        else:
            if self._ml_task not in ['classification', 'regression']:
                raise Exception(f"ml_task {ml_task} not recognized.")
            else:
                self.ml_task = self._ml_task

        if not self.get_pipeline_steps():
            self._ppl.add_pipe('data_preprocessing', DataPreprocessing(self.feat_types).pipeline(X, y))
            self._ppl.add_pipe('feature_preprocessing', FEATURE_PREPROCESSING)
            self._ppl.add_pipe('estimator', self.ml_task)   

        self._fit(data, target, time_budget)     
