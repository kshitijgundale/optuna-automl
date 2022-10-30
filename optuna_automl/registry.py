CLASSIFICATION = 'classification'
REGRESSION = 'regression'
NUMERICAL_IMPUTATION = 'numerical_imputation'
CATEGORICAL_IMPUTATION = 'categorical_imputation'
TEXT_IMPUTATION = 'text_imputation'
FEATURE_PREPROCESSING = 'feature_preprocessing'
CATEGORICAL_ENCODING = 'categorical_encoding'
TEXT_ENCODING = 'text_encoding'

class Registry():

    registry = {
        CLASSIFICATION: {},
        REGRESSION: {},
        NUMERICAL_IMPUTATION: {},
        CATEGORICAL_IMPUTATION: {},
        TEXT_IMPUTATION: {},
        TEXT_ENCODING: {},
        CATEGORICAL_ENCODING: {},
        FEATURE_PREPROCESSING: {}
    }

    @staticmethod
    def add_component_to_registry(task, params, component_class):
        Registry.registry[task][component_class.name] = {'class': component_class, 'params': params}

    @staticmethod
    def get_all_available_components(task):
        return Registry.registry[task].keys()

    @staticmethod
    def get_component_class(task, name):
        return Registry.registry[task][name]['class']

    @staticmethod
    def get_component_params(task, name):
        return Registry.registry[task][name]['params']

from optuna_automl.components.classification import random_forest
from optuna_automl.components.classification import extra_trees

from optuna_automl.components.data_preprocessing.imputation import simple_imputer
from optuna_automl.components.data_preprocessing.encoding import count_encoder
from optuna_automl.components.data_preprocessing.encoding import summary_encoder
from optuna_automl.components.data_preprocessing.encoding import one_hot_encoder
from optuna_automl.components.data_preprocessing.encoding import tfidf_vectorizer

from optuna_automl.components.feature_preprocessing import select_percentile
from optuna_automl.components.feature_preprocessing import rfe