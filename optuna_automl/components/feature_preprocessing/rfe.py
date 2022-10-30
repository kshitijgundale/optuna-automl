from optuna_automl.automl_component import AutomlComponent
from optuna_automl.hyperparameters import CategoricalHyperparameter, FloatHyperparameter
from optuna_automl.registry import Registry, FEATURE_PREPROCESSING

import pandas as pd
from sklearn.feature_selection import RFE as R
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LiR

class RFE(AutomlComponent):

    name = 'rfe'

    def __init__(self, estimator, n_features_to_select, step):
        self.preprocessor = None
        self.step = step
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select

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

        self.preprocessor = R(
            step=self.step,
            n_features_to_select=self.n_features_to_select,
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
step = FloatHyperparameter('step', 0, 1)
n_features_to_select = FloatHyperparameter('n_features_to_select', 0, 1)

params = [estimator, step, n_features_to_select]

Registry.add_component_to_registry(FEATURE_PREPROCESSING, params, RFE)






