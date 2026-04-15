from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is required for this pipeline. Install it with: pip install xgboost"
    ) from e


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
        "rainfall_mm",
        "storm_duration_hr",
        "impervious_frac",
        "infiltration_index",
        "runoff_coefficient_proxy",
    ]


def get_target():
    return "peak_flow_cms"


def split_data(df):
    features = get_features()
    target = get_target()

    required = features + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    X = df[features]
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_preprocessor(features):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric, features)]
    )


def build_searches(features):
    preprocessor = build_preprocessor(features)

    searches = {
        "ridge": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", Ridge()),
                ]
            ),
            param_grid={
                "model__alpha": [0.1, 1.0, 10.0],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        ),
        "svr": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", SVR()),
                ]
            ),
            param_grid={
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.1, 0.5, 1.0],
                "model__kernel": ["rbf"],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        ),
        "random_forest": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestRegressor(random_state=42)),
                ]
            ),
            param_grid={
                "model__n_estimators": [100, 200, 300],
                "model__max_depth": [3, 5, 10],
                "model__min_samples_split": [2, 5],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        ),
        "xgboost": GridSearchCV(
            estimator=Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=42,
                            n_estimators=200,
                        ),
                    ),
                ]
            ),
            param_grid={
                "model__max_depth": [3, 5, 10],
                "model__learning_rate": [0.05, 0.1],
                "model__subsample": [0.8, 1.0],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        ),
    }

    return searches


def evaluate_predictions(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
        "n_test": len(y_true),
    }


def train_and_compare(df):
    X_train, X_test, y_train, y_test = split_data(df)
    searches = build_searches(get_features())

    metrics_rows = []
    prediction_tables = {}
    fitted_models = {}

    for name, search in searches.items():
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)

        metrics = evaluate_predictions(y_test, y_pred)
        metrics["model"] = name
        metrics["best_params"] = str(search.best_params_)
        metrics_rows.append(metrics)

        pred_df = X_test.copy()
        pred_df["observed_peak_flow_cms"] = y_test.values
        pred_df["predicted_peak_flow_cms"] = y_pred
        pred_df["model"] = name

        prediction_tables[name] = pred_df
        fitted_models[name] = best_model

    metrics_df = pd.DataFrame(metrics_rows).sort_values("rmse").reset_index(drop=True)
    predictions_df = pd.concat(prediction_tables.values(), ignore_index=True)

    best_model_name = metrics_df.loc[0, "model"]
    best_model = fitted_models[best_model_name]
    best_predictions_df = prediction_tables[best_model_name].reset_index(drop=True)

    return (metrics_df, predictions_df, best_model_name, best_model, best_predictions_df, fitted_models,)