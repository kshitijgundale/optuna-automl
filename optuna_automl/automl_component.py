from optuna_automl.registry import Registry

class AutomlComponent():

    @classmethod
    def set_trial_params(cls, trial, task_name, step_name, ml_params):
        search_space = Registry.get_component_params(task_name, cls.name)
        params = {}
        for param in search_space:
            value = param.set_trial(trial, f'{step_name}__{param.name}')
            params[param.name] = value
        return params
