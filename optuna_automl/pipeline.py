from sklearn.pipeline import Pipeline

class AutomlPipeline(Pipeline):

    def set_trial_params(self, trial, name):
        for name, transformer in self.steps:
            transformer.set_trial_params(trial, name)
        return self

    def set_params(self, params, name):
        for name, transformer in self.steps:
            transformer.set_params(params, name)
        return self
    