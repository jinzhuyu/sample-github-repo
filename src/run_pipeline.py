from pathlib import Path

from process_data import load_data as load_raw_data, process_data, save_data
from make_plots import create_parity_plot, create_residual_plot
from model import save_result, train_and_compare


def run_pipeline():
    raw_path = "../data/raw_data/stormwater_events_sample.csv"
    processed_path = "../data/processed_data/stormwater_events_features.csv"
    metrics_path = "../output/model_comparison_metrics.csv"
    predictions_path = "../output/all_model_predictions.csv"
    best_predictions_path = "../output/best_model_predictions.csv"
    figures_dir = Path("../output/figures")

    raw_df = load_raw_data(raw_path)
    if raw_df is None:
        return None, None, None

    processed_df = process_data(raw_df)
    save_data(processed_df, processed_path)

    metrics_df, predictions_df, best_model_name, _, best_predictions_df = train_and_compare(processed_df)

    save_result(metrics_df, metrics_path)
    save_result(predictions_df, predictions_path)
    save_result(best_predictions_df, best_predictions_path)

    figures_dir.mkdir(parents=True, exist_ok=True)
    create_parity_plot(
        best_predictions_df,
        figures_dir / f"parity_plot_{best_model_name}.png",
    )
    create_residual_plot(
        best_predictions_df,
        figures_dir / f"residual_plot_{best_model_name}.png",
    )

    return metrics_df, predictions_df, best_model_name


def main():
    metrics_df, _, best_model_name = run_pipeline()
    if metrics_df is None:
        return

    print("Pipeline finished successfully.")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()