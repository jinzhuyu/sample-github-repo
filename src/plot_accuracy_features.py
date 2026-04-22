from pathlib import Path

from model import load_data
from make_plots import (
    create_feature_importance_plot,
    create_model_comparison_plots,
    create_parity_plot,
    create_residual_plot,
)


def main():
    """
    Load saved model outputs and create plots.

    Students can run this script after model.py.
    """
    metrics_path = "../output/accuracy/model_comparison_metrics.csv"
    best_predictions_path = "../output/accuracy/best_model_predictions.csv"
    rf_importance_path = "../output/accuracy/feature_importance_random_forest.csv"
    xgb_importance_path = "../output/accuracy/feature_importance_xgboost.csv"

    figures_dir = Path("../output/accuracy/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_data(metrics_path)
    best_predictions_df = load_data(best_predictions_path)

    if metrics_df is None or best_predictions_df is None:
        print("Required model output files were not found.")
        return

    best_model_name = metrics_df.sort_values("rmse", ascending=True).iloc[0]["model"]

    create_model_comparison_plots(metrics_df, figures_dir)

    create_parity_plot(
        best_predictions_df,
        figures_dir / f"parity_plot_{best_model_name}.png",
    )

    create_residual_plot(
        best_predictions_df,
        figures_dir / f"residual_plot_{best_model_name}.png",
    )

    rf_importance_df = load_data(rf_importance_path)
    if rf_importance_df is not None:
        create_feature_importance_plot(
            rf_importance_df,
            "random_forest",
            figures_dir / "feature_importance_random_forest.png",
        )

    xgb_importance_df = load_data(xgb_importance_path)
    if xgb_importance_df is not None:
        create_feature_importance_plot(
            xgb_importance_df,
            "xgboost",
            figures_dir / "feature_importance_xgboost.png",
        )

    print("Plotting finished.")
    print(f"Figures saved under: {figures_dir.resolve()}")


if __name__ == "__main__":
    main()