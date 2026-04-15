from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from evaluate_model import evaluate_and_save_results
from make_dataset import process_raw_dataset
from utils import FIGURES_DIR

def create_parity_plot(predictions_df: pd.DataFrame, output_path: Path) -> None:
    observed = predictions_df["observed_peak_flow_cms"]
    predicted = predictions_df["predicted_peak_flow_cms"]

    plt.figure(figsize=(6, 4.5))
    plt.scatter(observed, predicted, alpha=0.75)
    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Observed peak flow (cms)")
    plt.ylabel("Predicted peak flow (cms)")
    plt.title("Parity plot: observed vs predicted peak flow")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def create_residual_plot(predictions_df: pd.DataFrame, output_path: Path) -> None:
    predicted = predictions_df["predicted_peak_flow_cms"]
    residual = predictions_df["observed_peak_flow_cms"] - predictions_df["predicted_peak_flow_cms"]

    plt.figure(figsize=(6, 4.5))
    plt.scatter(predicted, residual, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted peak flow (cms)")
    plt.ylabel("Residual (observed - predicted)")
    plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()

def run_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    process_raw_dataset()
    metrics_df, predictions_df = evaluate_and_save_results()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    create_parity_plot(predictions_df, FIGURES_DIR / "parity_plot.png")
    create_residual_plot(predictions_df, FIGURES_DIR / "residual_plot.png")
    return metrics_df, predictions_df

def main() -> None:
    metrics_df, _ = run_pipeline()
    print("Pipeline finished successfully.")
    print(metrics_df.to_string(index=False))

if __name__ == "__main__":
    main()
