import pandas as pd
from optuna_automl.automl_choice import AutomlPreprocessingChoice
from optuna_automl.column_transformer import ColumnTransformer
from optuna_automl.pipeline import AutomlPipeline   
from optuna_automl.registry import CATEGORICAL_ENCODING, CATEGORICAL_IMPUTATION, NUMERICAL_IMPUTATION, TEXT_ENCODING, TEXT_IMPUTATION 


class DataPreprocessing():

    CATEGORICAL = "categorical"
    NUMERICAL = "numerical"
    TEXT = "text"

    def __init__(self, feat_types):
        self.feat_types = feat_types
        self.ppl = None

    def pipeline(self, X, y):
        """
        Returns data preprocessing pipeline
        """
        col_types = {
            DataPreprocessing.CATEGORICAL: [],
            DataPreprocessing.NUMERICAL: [],
            DataPreprocessing.TEXT: []
        }

        if self.feat_types is None:
            self.feat_types = {}

        for col in X.columns:
            if col not in self.feat_types:
                col_type = DataPreprocessing.infer_field_type(X[col])
            else:
                col_type = self.feat_types[col]
                if col_type not in col_types.keys():
                    raise Exception(f"Unknown feat type supplied. Provide one of following - {','.join(col_types.keys())}")

            if col_type == DataPreprocessing.NUMERICAL:
                X[col] = pd.to_numeric(X[col])

            col_types[col_type].append(col)

        preprocessing_steps = []

        if col_types[DataPreprocessing.NUMERICAL]:
            preprocessing_steps.append(
                ('num_imputation', AutomlPreprocessingChoice(task_name=NUMERICAL_IMPUTATION), col_types[DataPreprocessing.NUMERICAL])
            )

        if col_types[DataPreprocessing.CATEGORICAL]:
            ppl = AutomlPipeline([
                ('cat_imputation', AutomlPreprocessingChoice(task_name=CATEGORICAL_IMPUTATION)),
                ('cat_encoding', AutomlPreprocessingChoice(task_name=CATEGORICAL_ENCODING))
            ])
            preprocessing_steps.append(('cat_preprocessing', ppl, col_types[DataPreprocessing.CATEGORICAL]))

        if col_types[DataPreprocessing.TEXT]:
            ppl = AutomlPipeline([
                ('text_imputation', AutomlPreprocessingChoice(task_name=TEXT_IMPUTATION)),
                ('text_encoding', AutomlPreprocessingChoice(task_name=TEXT_ENCODING))
            ])
            preprocessing_steps.append(('text_preprocessing', ppl, col_types[DataPreprocessing.TEXT]))

        return ColumnTransformer(preprocessing_steps)

    @staticmethod
    def infer_field_type(col):
        """
        Returns field type of given column

        Parameters
        ----------
        col : pandas Series

        Returns
        -------
        str
        """
        if not isinstance(col, pd.Series):
            raise TypeError(f"must be pandas Series, not {type(col)}")
        if len(col) == 0:
            raise Exception(f"empty input provided")
        return DataPreprocessing.NUMERICAL