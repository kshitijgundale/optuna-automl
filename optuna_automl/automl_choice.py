from sklearn.base import TransformerMixin
from optuna_automl.registry import Registry

class AutomlChoice():

    def __init__(self, task_name):
        self.task_name = task_name
        self.component = None

    def get_component_name(self, trial):
        return self.component.name

    def set_component(self, trial, step_name):
        avaliable_components = Registry.get_all_available_components(self.task_name)
        self.component = trial.suggest_categorical(f'{step_name}', avaliable_components)

    def set_component_params(self, trial, step_name, ml_params):
        self.component = Registry.get_component_class(self.task_name, self.component)
        params = self.component.set_trial_params(trial, self.task_name, step_name, ml_params)
        self.component = self.component(**params)
            
    def set_trial_params(self, trial, step_name, ml_params):
        self.set_component(trial, step_name)
        self.set_component_params(trial, step_name, ml_params)
        return self

    def set_params(self, params, step_name):
        component_params = Registry.get_component_params(self.task_name, params[f'{step_name}'])
        init_params = {}
        for param in component_params:
            init_params[param.name] = params[f'{step_name}__{param.name}']

        self.component = Registry.get_component_class(self.task_name, params[f'{step_name}'])(**init_params)

    def fit(self, X, y):
        self.component.fit(X, y)
        return self

class AutomlPreprocessingChoice(AutomlChoice, TransformerMixin):

    def transform(self, X):
        return self.component.transform(X)

class AutomlEstimatorChoice(AutomlChoice):

    def predict(self, X):
        return self.component.predict(X)

    def predict_proba(self, X):
        return self.component.predict_proba(X)

    def fit(self, X, y):
        self.component.fit(X,y)
        self.classes_ = self.component.estimator.classes_
        return self
