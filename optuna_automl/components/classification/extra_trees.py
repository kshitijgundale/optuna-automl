from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier as ETC
from optuna_automl.hyperparameters import IntegerHyperparameter, FloatHyperparameter, CategoricalHyperparameter, UnParametrizedHyperparameter
from optuna_automl.registry import Registry, CLASSIFICATION
from optuna_automl.automl_component import AutomlComponent

class ExtraTreesClassifier(AutomlComponent, BaseEstimator, TransformerMixin):

    name = "extra_trees_classifier"

    def __init__(self, n_estimators, criterion, min_samples_leaf,
                 min_samples_split,  max_features, bootstrap, max_leaf_nodes,
                 max_depth, min_weight_fraction_leaf, min_impurity_decrease,
                 oob_score=False, n_jobs=1, random_state=None, verbose=0,
                 class_weight=None) -> None:
        
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.estimator = None

    def fit(self, X, y):
        self.estimator = ETC(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_features=self.max_features,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            bootstrap=self.bootstrap,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight=self.class_weight,
        )

        self.estimator.fit(X,y)
        return self
    
    def predict(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        return self.estimator.predict(X)

    def predict_proba(self, X):
        if self.estimator is None:
            raise NotImplementedError()

        return self.estimator.predict_proba(X)

n_estimators = IntegerHyperparameter('n_estimators', 10, 512)
criterion = CategoricalHyperparameter('criterion', ["gini", "entropy"])
max_features = FloatHyperparameter('max_features', 0.0, 1.0)
max_depth = UnParametrizedHyperparameter('max_depth', None)
min_samples_split = IntegerHyperparameter("min_samples_split", 2, 20)
min_samples_leaf = IntegerHyperparameter("min_samples_leaf", 1, 20)
min_weight_fraction_leaf = UnParametrizedHyperparameter("min_weight_fraction_leaf", 0.)
max_leaf_nodes = UnParametrizedHyperparameter("max_leaf_nodes", None)
min_impurity_decrease = UnParametrizedHyperparameter('min_impurity_decrease', 0.0)
bootstrap = CategoricalHyperparameter("bootstrap", ["True", "False"])

params = [
    n_estimators, criterion, max_features, 
    max_depth, min_samples_split, min_samples_leaf, 
    min_weight_fraction_leaf, max_leaf_nodes, min_impurity_decrease,
    bootstrap
]

Registry.add_component_to_registry(CLASSIFICATION, params, ExtraTreesClassifier)