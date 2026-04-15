from pathlib import Path

from make_dataset import load_data as load_raw_data, process_data, save_data
from make_plots import create_parity_plot, create_residual_plot
from model import evaluate, make_predictions, save_result, train


def run_pipeline():
    raw_path = "../data/raw_data/stormwater_events_sample.csv"
    processed_path = "../data/processed_data/stormwater_events_features.csv"
    metrics_path = "../output/metrics.csv"
    predictions_path = "../output/predictions.csv"
    figures_dir = Path("../output/figures")

    raw_df = load_raw_data(raw_path)
    if raw_df is None:
        return None, None

    processed_df = process_data(raw_df)
    save_data(processed_df, processed_path)

    model, x_test, y_test = train(processed_df)
    y_pred = model.predict(x_test)

    metrics = evaluate(y_test, y_pred)
    predictions = make_predictions(x_test, y_test, y_pred)

    save_result(metrics, metrics_path)
    save_result(predictions, predictions_path)

    figures_dir.mkdir(parents=True, exist_ok=True)
    create_parity_plot(predictions, figures_dir / "parity_plot.png")
    create_residual_plot(predictions, figures_dir / "residual_plot.png")

    return metrics, predictions


def main():
    metrics, _ = run_pipeline()
    if metrics is None:
        return

    print("Pipeline finished successfully.")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()