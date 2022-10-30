from category_encoders import SummaryEncoder as SE
from optuna_automl.registry import Registry, CATEGORICAL_ENCODING
from optuna_automl.automl_component import AutomlComponent

class SummaryEncoder(AutomlComponent):

    name = "summary_encoder"

    def __init__(self):
        self.preprocessor = None

    def fit(self, X, y):
        self.preprocessor = SE(handle_unknown='value', cols=X.columns)
        self.preprocessor.fit(X, y)
        return self

    def transform(self, X):
        return self.preprocessor.transform(X)

Registry.add_component_to_registry(CATEGORICAL_ENCODING, [], SummaryEncoder)