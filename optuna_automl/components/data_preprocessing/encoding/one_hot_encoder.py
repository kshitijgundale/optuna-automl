from category_encoders.one_hot import OneHotEncoder as OHE
from optuna_automl.registry import Registry, CATEGORICAL_ENCODING
from optuna_automl.automl_component import AutomlComponent

class OneHotEncoder(AutomlComponent):

    name = "one_hot_encoder"

    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y=None):
        self.preprocessor = OHE(handle_unknown="value", cols=X.columns, use_cat_names=True)
        self.preprocessor.fit(X)
        return self
    
    def transform(self, X):
        return self.preprocessor.transform(X)

Registry.add_component_to_registry(CATEGORICAL_ENCODING, [], OneHotEncoder)