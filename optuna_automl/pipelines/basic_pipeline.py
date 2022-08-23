from optuna_automl.automl_choice import AutomlEstimatorChoice, AutomlPreprocessingChoice
from optuna_automl.registry import CLASSIFICATION, FEATURE_PREPROCESSING
import pandas as pd
from optuna_automl.data_preprocessing import DataPreprocessing

def basic_pipeline(X, y, feat_types=None):

    return [
        ('data_preprocessing', DataPreprocessing(feat_types).pipeline(X, y)),
        ('feature_preprocessing', AutomlPreprocessingChoice(task_name=FEATURE_PREPROCESSING)),
        ('estimator', AutomlEstimatorChoice(task_name=CLASSIFICATION))
    ]