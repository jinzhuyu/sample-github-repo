from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError(
        "xgboost is required for this pipeline. Install it with: pip install xgboost"
    ) from e


RANDOM_STATE = 123


def load_data(path):
    """Load CSV data."""
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def save_result(data, path):
    """Save a DataFrame to CSV."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        print(f"Saved to {path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def get_features():
    """Return feature column names."""
    return [
        "rainfall_mm",
        "storm_duration_hr",
        "impervious_frac",
        "infiltration_index",
        "runoff_coefficient_proxy",
    ]


def get_target():
    """Return target column name."""
    return "peak_flow_cms"


def split_data(df):
    """Split processed data into training and test sets."""
    features = get_features()
    target = get_target()

    required = features + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    X = df[features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )


def preprocess_features(X_train, X_test):
    """
    Impute missing values and scale features.

    We do this explicitly instead of using a sklearn Pipeline
    so students can see the preprocessing steps more clearly.
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    X_train_processed = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_processed = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
        index=X_test.index,
    )

    return X_train_processed, X_test_processed


def evaluate_predictions(y_true, y_pred):
    """Compute prediction metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def train_model_search(model_name, X_train, y_train):
    """Fit one model with GridSearchCV."""
    if model_name == "ridge":
        search = GridSearchCV(
            estimator=Ridge(),
            param_grid={"alpha": [0.1, 1.0, 10.0]},
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )

    elif model_name == "svr":
        search = GridSearchCV(
            estimator=SVR(),
            param_grid={
                "C": [0.1, 1.0, 10.0],
                "epsilon": [0.1, 0.5, 1.0],
                "kernel": ["rbf"],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )

    elif model_name == "random_forest":
        search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=RANDOM_STATE),
            param_grid={
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 10],
                "min_samples_split": [2, 5],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )

    elif model_name == "xgboost":
        search = GridSearchCV(
            estimator=XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_estimators=200,
            ),
            param_grid={
                "max_depth": [3, 5, 10],
                "learning_rate": [0.05, 0.1],
                "subsample": [0.8, 1.0],
            },
            scoring="neg_root_mean_squared_error",
            cv=3,
            n_jobs=-1,
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    search.fit(X_train, y_train)
    return search


def train_and_compare(df):
    """
    Train all candidate models, compare test performance,
    and return results tables for saving.
    """
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_processed, X_test_processed = preprocess_features(X_train, X_test)

    model_names = ["ridge", "svr", "random_forest", "xgboost"]

    metrics_rows = []
    prediction_tables = {}
    fitted_models = {}

    for name in model_names:
        search = train_model_search(name, X_train_processed, y_train)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test_processed)

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

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    predictions_df = pd.concat(
        prediction_tables.values(),
        ignore_index=True,
    )

    best_model_name = metrics_df.loc[0, "model"]
    best_model = fitted_models[best_model_name]
    best_predictions_df = prediction_tables[best_model_name].reset_index(drop=True)

    return (
        metrics_df,
        predictions_df,
        best_model_name,
        best_model,
        best_predictions_df,
        fitted_models,
    )


def build_feature_importance_df(model, feature_names):
    """
    Build a feature-importance table for tree-based models.

    Only Random Forest and XGBoost support feature_importances_.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    return pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )


def main():
    """
    Load processed data, train models, and save all accuracy outputs.

    Students can run this script after process_data.py.
    """
    processed_path = "../data/processed_data/stormwater_events_features.csv"

    metrics_path = "../output/accuracy/model_comparison_metrics.csv"
    predictions_path = "../output/accuracy/all_model_predictions.csv"
    best_predictions_path = "../output/accuracy/best_model_predictions.csv"
    rf_importance_path = "../output/accuracy/feature_importance_random_forest.csv"
    xgb_importance_path = "../output/accuracy/feature_importance_xgboost.csv"

    df = load_data(processed_path)
    if df is None:
        return

    (
        metrics_df,
        predictions_df,
        best_model_name,
        best_model,
        best_predictions_df,
        fitted_models,
    ) = train_and_compare(df)

    save_result(metrics_df, metrics_path)
    save_result(predictions_df, predictions_path)
    save_result(best_predictions_df, best_predictions_path)

    feature_names = get_features()

    rf_importance_df = build_feature_importance_df(
        fitted_models["random_forest"],
        feature_names,
    )
    if rf_importance_df is not None:
        save_result(rf_importance_df, rf_importance_path)

    xgb_importance_df = build_feature_importance_df(
        fitted_models["xgboost"],
        feature_names,
    )
    if xgb_importance_df is not None:
        save_result(xgb_importance_df, xgb_importance_path)

    print("Model training finished.")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()