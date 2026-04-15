from pathlib import Path

from process_data import load_data as load_raw_data, process_data, save_data
from make_plots import (
    create_feature_importance_plot,
    create_model_comparison_plots,
    create_parity_plot,
    create_residual_plot,
)
from model import get_features, save_result, train_and_compare


def build_feature_importance_df(model, feature_names):
    regressor = model.named_steps["model"]
    return __import__("pandas").DataFrame(
        {
            "feature": feature_names,
            "importance": regressor.feature_importances_,
        }
    )


def load_or_build_feature_importance(
    model_name,
    csv_path,
    feature_names,
    fitted_models=None,
):
    csv_file = Path(csv_path)

    if fitted_models is not None:
        importance_df = build_feature_importance_df(
            fitted_models[model_name],
            feature_names,
        )
        save_result(importance_df, csv_path)
        return importance_df

    if not csv_file.exists():
        print(f"{model_name} importance file not found at {csv_path}.")
        return None

    importance_df = load_raw_data(csv_path)
    if importance_df is None:
        print(f"Failed to load {model_name} importance data from {csv_path}.")
        return None

    return importance_df


def run_pipeline(force_retrain=False):
    raw_path = "../data/raw_data/stormwater_events_sample.csv"
    processed_path = "../data/processed_data/stormwater_events_features.csv"
    metrics_path = "../output/model_comparison_metrics.csv"
    predictions_path = "../output/all_model_predictions.csv"
    best_predictions_path = "../output/best_model_predictions.csv"

    rf_importance_path = "../output/feature_importance_random_forest.csv"
    xgb_importance_path = "../output/feature_importance_xgboost.csv"

    figures_dir = Path("../output/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    processed_file = Path(processed_path)
    metrics_file = Path(metrics_path)
    predictions_file = Path(predictions_path)
    best_predictions_file = Path(best_predictions_path)
    rf_importance_file = Path(rf_importance_path)
    xgb_importance_file = Path(xgb_importance_path)

    if not processed_file.exists():
        raw_df = load_raw_data(raw_path)
        if raw_df is None:
            return None, None, None

        processed_df = process_data(raw_df)
        save_data(processed_df, processed_path)
    else:
        print(
            f"Processed data already exists at {processed_path}, "
            "skipping processing step."
        )
        processed_df = load_raw_data(processed_path)
        if processed_df is None:
            return None, None, None

    need_modeling = force_retrain or not (
        metrics_file.exists()
        and predictions_file.exists()
        and best_predictions_file.exists()
        and rf_importance_file.exists()
        and xgb_importance_file.exists()
    )

    fitted_models = None

    if need_modeling:
        print("Running model training and evaluation.")
        (
            metrics_df,
            predictions_df,
            best_model_name,
            _,
            best_predictions_df,
            fitted_models,
        ) = train_and_compare(processed_df)

        save_result(metrics_df, metrics_path)
        save_result(predictions_df, predictions_path)
        save_result(best_predictions_df, best_predictions_path)
    else:
        print("Model outputs already exist, skipping modeling step.")
        metrics_df = load_raw_data(metrics_path)
        predictions_df = load_raw_data(predictions_path)
        best_predictions_df = load_raw_data(best_predictions_path)

        if metrics_df is None or predictions_df is None or best_predictions_df is None:
            return None, None, None

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

    # Always regenerate feature-importance plots on every run.
    # If models were retrained, refresh the CSVs first.
    # If modeling was skipped, load the saved CSVs and replot using the latest plotting code.
    feature_names = get_features()

    for model_name, csv_path in [
        ("random_forest", rf_importance_path),
        ("xgboost", xgb_importance_path),
    ]:
        importance_df = load_or_build_feature_importance(
            model_name=model_name,
            csv_path=csv_path,
            feature_names=feature_names,
            fitted_models=fitted_models,
        )

        if importance_df is None:
            continue

        create_feature_importance_plot(
            importance_df,
            model_name,
            figures_dir / f"feature_importance_{model_name}.png",
        )

    return metrics_df, predictions_df, best_model_name


def main():
    metrics_df, _, best_model_name = run_pipeline(force_retrain=False)
    if metrics_df is None:
        return

    print("Pipeline finished successfully.")
    print(f"Best model: {best_model_name}")
    print(metrics_df.to_string(index=False))
    

if __name__ == "__main__":
    main()