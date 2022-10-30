from sklearn.feature_selection import chi2, f_classif, SelectPercentile as SP
import pandas as pd
from optuna_automl.hyperparameters import CategoricalHyperparameter, FloatHyperparameter
from optuna_automl.registry import Registry, FEATURE_PREPROCESSING
from optuna_automl.automl_component import AutomlComponent

class SelectPercentile(AutomlComponent):

    name = "select_percentile"

    def __init__(self, percentile, score_func):
        self.preprocessor = None
        self.features_selected = None

        self.percentile = percentile
        if score_func == "chi2":
            self.score_func = chi2
        elif score_func == "f_classif":
            self.score_func = f_classif

    def fit(self, X, y):
        self.preprocessor = SP(
            score_func = self.score_func,
            percentile = self.percentile
        )
        self.preprocessor.fit(X, y)
        self.features_selected = X.columns[self.preprocessor.get_support()]
        return self

    def transform(self, X):
        return pd.DataFrame(data=self.preprocessor.transform(X), columns=self.features_selected)

percentile = FloatHyperparameter("percentile", 0, 100)
score_func = CategoricalHyperparameter("score_func", ["chi2", "f_classif"])

params = [percentile, score_func]

Registry.add_component_to_registry(FEATURE_PREPROCESSING, params, SelectPercentile)

    