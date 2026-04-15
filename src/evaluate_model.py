from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train_model import train_model
from utils import METRICS_PATH, PREDICTIONS_PATH, write_csv

def calculate_metrics(observed: pd.Series, predicted: pd.Series) -> pd.DataFrame:
    metrics = {
        "mae": mean_absolute_error(observed, predicted),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": r2_score(observed, predicted),
        "n_test": len(observed),
    }
    return pd.DataFrame([metrics])

def build_prediction_table(
    x_test: pd.DataFrame,
    observed: pd.Series,
    predicted: pd.Series,
) -> pd.DataFrame:
    result = x_test.copy()
    result["observed_peak_flow_cms"] = observed.values
    result["predicted_peak_flow_cms"] = predicted
    return result

def evaluate_and_save_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    artifacts = train_model()
    predicted = artifacts.model.predict(artifacts.x_test)
    metrics_df = calculate_metrics(artifacts.y_test, predicted)
    predictions_df = build_prediction_table(
        artifacts.x_test,
        artifacts.y_test,
        predicted,
    )
    write_csv(metrics_df, METRICS_PATH)
    write_csv(predictions_df, PREDICTIONS_PATH)
    return metrics_df, predictions_df

def main() -> None:
    evaluate_and_save_results()

if __name__ == "__main__":
    main()
