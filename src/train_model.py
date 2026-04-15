from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import PROCESSED_DATA_PATH, read_csv, select_feature_columns, select_target_column

@dataclass(frozen=True)
class TrainingArtifacts:
    model: Pipeline
    x_test: pd.DataFrame
    y_test: pd.Series

def build_model(feature_columns: list[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_transformer, feature_columns)]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

def split_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    feature_columns = select_feature_columns()
    target_column = select_target_column()
    return train_test_split(
        df[feature_columns],
        df[target_column],
        test_size=0.25,
        random_state=42,
    )

def train_model() -> TrainingArtifacts:
    df = read_csv(PROCESSED_DATA_PATH)
    feature_columns = select_feature_columns()
    x_train, x_test, y_train, y_test = split_dataset(df)
    model = build_model(feature_columns)
    model.fit(x_train, y_train)
    return TrainingArtifacts(model=model, x_test=x_test, y_test=y_test)

def main() -> None:
    train_model()

if __name__ == "__main__":
    main()
