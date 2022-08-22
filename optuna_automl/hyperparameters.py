class IntegerHyperparameter():

    def __init__(self, name, low, high) -> None:
        self.name = name
        self.low = low
        self.high = high

    def set_trial(self, trial, name):
        v = trial.suggest_int(name, self.low, self.high)
        return v

class FloatHyperparameter():

    def __init__(self, name, low, high) -> None:
        self.name = name
        self.low = low
        self.high = high

    def set_trial(self, trial, name):
        v = trial.suggest_float(name, self.low, self.high)
        return v

class CategoricalHyperparameter():

    def __init__(self, name, choices) -> None:
        self.name = name
        self.choices = choices

    def set_trial(self, trial, name):
        v = trial.suggest_categorical(name, self.choices)
        return v

class DiscreteHyperparameter():

    def __init__(self, name, low, high, q) -> None:
        self.name = name
        self.high = high
        self.low = low
        self.q = q

    def set_trial(self, trial, name):
        v = trial.suggest_discrete_uniform(name, self.low, self.high, self.q)
        return v

class LogUniformHyperparameter():

    def __init__(self, name, low, high) -> None:
        self.name = name
        self.low = low
        self.high = high

    def set_trial(self, trial, name):
        v = trial.suggest_loguniform(name, self.low, self.high)
        return v

class UniformHyperparameter():

    def __init__(self, name, low, high) -> None:
        self.name = name
        self.low = low
        self.high = high

    def set_trial(self, trial, name):
        v = trial.suggest_uniform(name, self.low, self.high)
        return v

class UnParametrizedHyperparameter():

    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value

    def set_trial(self, trial, name):
        v = trial.suggest_categorical(name, [self.value])
        return v