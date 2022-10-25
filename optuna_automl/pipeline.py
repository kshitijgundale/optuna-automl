from sklearn.pipeline import Pipeline
from optuna_automl.registry import Registry, CLASSIFICATION, REGRESSION
from optuna_automl.automl_choice import AutomlEstimatorChoice, AutomlPreprocessingChoice
import optuna_automl
import copy

class AutomlPipeline(Pipeline):

    def set_trial_params(self, trial, name):
        for name, transformer in self.steps:
            transformer.set_trial_params(trial, name)
        return self

    def set_params(self, params, name):
        for name, transformer in self.steps:
            transformer.set_params(params, name)
        return self

    def add_pipe(self, name, task):

        if isinstance(task, str):
            if task in Registry.registry.keys():
                if task in [CLASSIFICATION, REGRESSION]:
                    self.steps.append(
                        (name, AutomlEstimatorChoice(task_name=task))
                    )
                else:
                    self.steps.append(
                        (name, AutomlPreprocessingChoice(task_name=task))
                    )
            else:
                raise Exception("Unkown task provided.")

        elif isinstance(task, optuna_automl.automl.AutoML):
            if len(task._ppl.steps) != 0:
                component = copy.deepcopy(task._ppl)
                component.add_parent_prefix(name)
                self.steps.append((name, component))
            else:
                raise Exception("Empty pipeline cannot be added.")

        elif isinstance(task, optuna_automl.column_transformer.ColumnTransformer):
            component = copy.deepcopy(task)
            component.add_parent_prefix(name)
            self.steps.append((name, component))

        else:
            raise Exception(f"Cannot use {type(task)} as pipeline component.")

    def add_parent_prefix(self, prefix):
        for i in range(len(self.steps)):
            self.steps[i] = (f"{prefix}/{self.steps[i][0]}", self.steps[i][1])
            if isinstance(self.steps[i][1], optuna_automl.pipeline.AutomlPipeline) or isinstance(self.steps[i][1], optuna_automl.column_transformer.ColumnTransformer):
                self.steps[i][1].add_parent_prefix(prefix)

    