import pandas as pd
from sklearn.base import TransformerMixin
import json

class ColumnTransformer(TransformerMixin):

    def __init__(self, transformers):
        self.transformers = transformers
        
    def fit(self, X, y):
        for name, transformer, cols in self.transformers:
            transformer.fit(X[cols], y)
        return self

    def transform(self, X):
        transformed_data = []
        
        for name, transformer, cols in self.transformers:
            transformed_data.append(transformer.transform(X[cols]).reset_index(drop=True))

        X_transformed = pd.concat(transformed_data, axis=1)

        print(X_transformed)

        return X_transformed

    def set_trial_params(self, trial, name):
        for step_name, transformer, _ in self.transformers:
            transformer.set_trial_params(trial, step_name)
        return self

    def set_params(self, params, name):
        for step_name, transformer, _ in self.transformers:
            transformer.set_params(params, step_name)
        return self