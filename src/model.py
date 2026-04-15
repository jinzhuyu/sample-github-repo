# This file contains the main modeling code for training and evaluating a Ridge regression model on stormwater event data.
from __future__ import annotations # imported to enable future annotations for better type hinting and code clarity.
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def save_result(data, path):
    try:
        data.to_csv(path, index=False)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def get_features():
    return [
        "rainfall_depth_mm",
        "duration_min",
        "impervious_frac",
        "infiltration_index",
        "runoff_coefficient_proxy",
    ]


def get_target():
    return "peak_flow_cms"


def build_model(features):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric, features)]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def split_data(df):
    X = df[get_features()]
    y = df[get_target()]
    return train_test_split(X, y, test_size=0.2, random_state=123)


def train(df):
    X_train, X_test, y_train, y_test = split_data(df)
    model = build_model(get_features())
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate(y_true, y_pred):
    return pd.DataFrame([{
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "n_test": len(y_true),
    }])


def make_predictions(X, y, y_pred):
    df = X.copy()
    df["observed_peak_flow_cms"] = y.values
    df["predicted_peak_flow_cms"] = y_pred
    return df


def main():
    data_path = "../data/processed_data/stormwater_events_features.csv"
    metrics_path = "../output/metrics.csv"
    predictions_path = "../output/predictions.csv"

    df = load_data(data_path)
    if df is None:
        return

    model, X_test, y_test = train(df)
    y_pred = model.predict(X_test)

    metrics = evaluate(y_test, y_pred)
    preds = make_predictions(X_test, y_test, y_pred)

    save_result(metrics, metrics_path)
    save_result(preds, predictions_path)


if __name__ == "__main__":
    main()