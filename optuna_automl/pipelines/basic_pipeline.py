from optuna_automl.automl_choice import AutomlEstimatorChoice
from optuna_automl.registry import CLASSIFICATION
import pandas as pd
from optuna_automl.data_preprocessing import DataPreprocessing

def basic_pipeline(X, y, feat_types=None):

    return [
        ('data_preprocessing', DataPreprocessing(feat_types).pipeline(X, y)),
        ('estimator', AutomlEstimatorChoice(task_name=CLASSIFICATION))
    ]