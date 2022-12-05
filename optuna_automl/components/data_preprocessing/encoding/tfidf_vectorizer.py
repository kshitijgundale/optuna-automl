from sklearn.feature_extraction.text import TfidfVectorizer as TV
from optuna_automl.registry import Registry, TEXT_ENCODING
from sklearn.base import TransformerMixin
from optuna_automl.hyperparameters import CategoricalHyperparameter, IntegerHyperparameter
import pandas as pd
from optuna_automl.automl_component import AutomlComponent

class TfidfVectorizer(AutomlComponent, TransformerMixin):

    name = "tfidf_encoder"

    def __init__(self, ngram_range, max_features):
        self.preprocessor = None
        self.ngram_range = ngram_range
        self.max_features = max_features

    def fit(self, X, y=None):
        self.preprocessor = TV(ngram_range=self.ngram_range, max_features=self.max_features)
        self.preprocessor.fit(X.apply(lambda x : " ".join(x), axis=1))
        return self
    
    def transform(self, X):
        data = self.preprocessor.transform(X.apply(lambda x : " ".join(x), axis=1)).toarray()
        return pd.DataFrame(data=data, columns=[f"text_{i}" for i in range(len(data[0]))])

ngram_range = CategoricalHyperparameter("ngram_range", [(i,j) for i in range(1,2) for j in range(1,2) if i<=j])
max_features = IntegerHyperparameter("max_features", 1, 100)
Registry.add_component_to_registry(TEXT_ENCODING, [ngram_range, max_features], TfidfVectorizer)    


