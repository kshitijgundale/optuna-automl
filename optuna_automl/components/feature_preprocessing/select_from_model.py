from optuna_automl.automl_component import AutomlComponent
from optuna_automl.hyperparameters import CategoricalHyperparameter, FloatHyperparameter
from optuna_automl.registry import Registry, FEATURE_PREPROCESSING

import pandas as pd
from sklearn.feature_selection import SelectFromModel as SFM
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LiR

class SelectFromModel(AutomlComponent):

    name = "select_from_model"

    def __init__(self, max_features, estimator):
        self.max_features = max_features
        self.estimator = estimator

    def fit(self, X, y):

        if self.estimator == 'knc':
            estimator = KNC()
        elif self.estimator == 'rfc':
            estimator = RFC()
        elif self.estimator == 'lr':
            estimator = LR()
        elif self.estimator == 'rfr':
            estimator = RFR()
        elif self.estimator == 'lir':
            estimator = LiR()

        self.preprocessor = SFM(
            max_features=lambda X: int(len(X.columns) * self.max_features),
            estimator = estimator
        )
        self.preprocessor.fit(X, y)
        self.features_selected = X.columns[self.preprocessor.get_support()]
        return self

    def transform(self, X):
        return pd.DataFrame(data=self.preprocessor.transform(X), columns=self.features_selected)

    @classmethod
    def set_trial_params(cls, trial, task_name, step_name, ml_params):
        search_space = Registry.get_component_params(task_name, cls.name)
        params = {}
        for param in search_space:
            if param.name == "estimator":
                if ml_params['ml_task'] == 'classification':
                    value = param.set_trial(trial, f'{step_name}__{param.name}', include=['rfc', 'lr'])
                elif ml_params['ml_task'] == 'regression':
                    value = param.set_trial(trial, f'{step_name}__{param.name}', include=['lir', 'rfr'])
            else:
                value = param.set_trial(trial, f'{step_name}__{param.name}')
            params[param.name] = value
        return params

estimator = CategoricalHyperparameter('estimator', ['rfc', 'lr', 'rfr', 'lir'])
max_features = FloatHyperparameter('max_features', 0, 1)

params = [estimator, max_features]

Registry.add_component_to_registry(FEATURE_PREPROCESSING, params, SelectFromModel)