from sklearn.impute import SimpleImputer as SI
from sklearn.base import TransformerMixin
import pandas as pd
from optuna_automl.hyperparameters import CategoricalHyperparameter
from optuna_automl.registry import Registry, NUMERICAL_IMPUTATION, CATEGORICAL_IMPUTATION, TEXT_IMPUTATION
from optuna_automl.automl_component import AutomlComponent

class SimpleImputer(AutomlComponent, TransformerMixin):

    name = "simple_imputer"

    def __init__(self, strategy):
        self.preprocessor = None
        self.strategy = strategy

    def fit(self, X, y=None):
        self.preprocessor = SI(strategy=self.strategy)
        self.preprocessor.fit(X, y)
        return self
    
    def transform(self, X):
        return pd.DataFrame(self.preprocessor.transform(X), columns=X.columns)
    
# Categorical
strategy = CategoricalHyperparameter('strategy', ["most_frequent", "constant"])
Registry.add_component_to_registry(CATEGORICAL_IMPUTATION, [strategy], SimpleImputer)

# Numerical
strategy = CategoricalHyperparameter('strategy', ["most_frequent", "constant", "mean", "median"])
Registry.add_component_to_registry(NUMERICAL_IMPUTATION, [strategy], SimpleImputer)

# Text
strategy = CategoricalHyperparameter('strategy', ["most_frequent", "constant"])
Registry.add_component_to_registry(TEXT_IMPUTATION, [strategy], SimpleImputer)

