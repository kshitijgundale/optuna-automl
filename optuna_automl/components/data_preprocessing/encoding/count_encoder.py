from category_encoders import CountEncoder as CE
from optuna_automl.registry import Registry, CATEGORICAL_ENCODING
import json
import pandas as pd

class CountEncoder():

    name = "count_encoder"

    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y=None):
        self.preprocessor = CE(handle_unknown=0, cols=X.columns)
        self.preprocessor.fit(X)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

Registry.add_component_to_registry(CATEGORICAL_ENCODING, [], CountEncoder)