import pandas as pd
from sklearn.base import TransformerMixin
from optuna_automl.pipeline import AutomlPipeline

class ColumnTransformer(TransformerMixin):

    def __init__(self, transformers=[]):
        self.transformers = transformers

        self.branches = {}
        
    def fit(self, X, y):
        for name, transformer, cols in self.transformers:
            transformer.fit(X[cols], y)
        return self

    def transform(self, X):
        transformed_data = []
        
        for name, transformer, cols in self.transformers:
            transformed_data.append(transformer.transform(X[cols]).reset_index(drop=True))

        X_transformed = pd.concat(transformed_data, axis=1)

        return X_transformed

    def set_trial_params(self, trial, name, ml_params):
        for step_name, transformer, _ in self.transformers:
            transformer.set_trial_params(trial, step_name, ml_params)
        return self

    def set_params(self, params, name):
        for step_name, transformer, _ in self.transformers:
            transformer.set_params(params, step_name)
        return self

    def add_branch(self, name, cols):
        self.transformers.append([
            name,
            AutomlPipeline(steps=[]),
            cols
        ])
        self.branches[name] = len(self.transformers) - 1 
        return self.branches.keys()

    def add_branch_pipe(self, branch, name, task):
        if branch not in self.branches:
            raise Exception("Branch not added in ColumnTransformer. Use add_branch method to add a new branch.")

        self.transformers[self.branches[branch]][1].add_pipe(f"{branch}/{name}", task)

    def add_parent_prefix(self, prefix):
        for i in range(len(self.transformers)):
            self.transformers[i][0] = f"{prefix}/{self.transformers[i][0]}"

            self.transformers[i][1].add_parent_prefix(prefix)