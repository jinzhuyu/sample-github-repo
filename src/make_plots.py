import matplotlib.pyplot as plt
import pandas as pd
plt.rcParams["axes.axisbelow"] = True # ensure grid lines are behind points/bars

def format_model_name(name):
    '''Convert model identifier to a more readable format for plotting.'''
    label_map = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "svr": "SVR",
        "ridge": "Ridge",
    }
    return label_map.get(name, name)

def create_parity_plot(data, path):
    observed = data["observed_peak_flow_cms"]
    predicted = data["predicted_peak_flow_cms"]

    plt.figure(figsize=(5, 4))
    plt.scatter(observed, predicted, alpha=0.75)
    lower = min(observed.min(), predicted.min())
    upper = max(observed.max(), predicted.max())
    plt.plot([lower, upper], [lower, upper], linestyle="--")
    plt.xlabel("Observed peak flow (cms)")
    plt.ylabel("Predicted peak flow (cms)")
    
    # add y-axis grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # plt.title("Parity plot: observed vs predicted peak flow")
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    # save pdf as well with dpi = 250
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_residual_plot(data, path):
    predicted = data["predicted_peak_flow_cms"]
    residual = data["observed_peak_flow_cms"] - data["predicted_peak_flow_cms"]

    plt.figure(figsize=(5, 4))
    plt.scatter(predicted, residual, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted peak flow (cms)")
    plt.ylabel("Residual (observed - predicted)")

    # add y-axis grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # plt.title("Residual plot")
    plt.tight_layout()
    plt.savefig(path, dpi=400)
    # save pdf as well with dpi = 250
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_metric_bar_plot(metrics_df, metric, path):

    data = metrics_df.copy()

    if metric == "r2":
        data = data.sort_values(metric, ascending=False)
    else:
        data = data.sort_values(metric, ascending=True)

    labels = [format_model_name(m) for m in data["model"]]
    values = data[metric].values

    plt.figure(figsize=(4, 3.25))
    bars = plt.bar(labels, values)
    plt.ylim(0, max(values) * 1.25)
    plt.xticks(rotation=45)
    # plt.xlabel("Model")
    plt.ylabel(metric.upper())
    # plt.title(f"{metric.upper()} by model")
    
    # add y-axis grid lines
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # add values on top of bars
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height()*1.01,
            f"{val:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    # save pdf as well with dpi = 250
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_model_comparison_plots(metrics_df, figures_dir):
    create_metric_bar_plot(metrics_df, "rmse", figures_dir / "model_comparison_rmse.png")
    create_metric_bar_plot(metrics_df, "mae", figures_dir / "model_comparison_mae.png")
    create_metric_bar_plot(metrics_df, "r2", figures_dir / "model_comparison_r2.png")
    
def create_feature_importance_plot(data, model_name, path):
    feature_name_map = {
        "rainfall_mm": "Rainfall",
        "storm_duration_hr": "Storm duration",
        "impervious_frac": "Impervious fraction",
        "infiltration_index": "Infiltration index",
        "runoff_coefficient_proxy": "Runoff coefficient",
    }

    data = data.copy()
    data["feature"] = data["feature"].replace(feature_name_map)
    data = data.sort_values("importance", ascending=True)

    plt.figure(figsize=(6, 3.5))
    bars = plt.barh(data["feature"], data["importance"])

    plt.xlabel("Importance")
    plt.xlim(0, data["importance"].max() * 1.25)

    for bar, val in zip(bars, data["importance"].values):
        plt.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
        )

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()
    
def summarize_for_plot(data, x_col):
    """Compute mean and std of RMSE for a plotting variable."""
    return (
        data.groupby(x_col, as_index=False)
        .agg(
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
        )
        .sort_values(x_col)
    )


def create_data_size_robustness_plot(data, path):
    """
    Expected columns:
        - train_fraction
        - rmse
    """
    plot_data = summarize_for_plot(data, "train_fraction")

    plt.figure(figsize=(5, 4))
    plt.errorbar(
        plot_data["train_fraction"],
        plot_data["mean_rmse"],
        yerr=plot_data["std_rmse"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Training data fraction")
    plt.ylabel("RMSE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_missing_data_robustness_plot(data, path):
    """
    Expected columns:
        - missing_rate
        - rmse
    """
    plot_data = summarize_for_plot(data, "missing_rate")

    plt.figure(figsize=(5, 4))
    plt.errorbar(
        plot_data["missing_rate"],
        plot_data["mean_rmse"],
        yerr=plot_data["std_rmse"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Missing data fraction")
    plt.ylabel("RMSE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_perturbation_robustness_plot(data, path):
    """
    Expected columns:
        - noise_scale_sd_units
        - rmse
    """
    plot_data = summarize_for_plot(data, "noise_scale_sd_units")

    plt.figure(figsize=(5, 4))
    plt.errorbar(
        plot_data["noise_scale_sd_units"],
        plot_data["mean_rmse"],
        yerr=plot_data["std_rmse"],
        marker="o",
        capsize=4,
    )

    plt.xlabel("Noise scale (SD units)")
    plt.ylabel("RMSE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_bootstrap_robustness_boxplot(data, path):
    """
    Expected columns:
        - rmse
    """
    plt.figure(figsize=(5, 4))
    plt.boxplot(
        data["rmse"].dropna().values,
        tick_labels=["Random Forest"],
    )

    plt.ylabel("RMSE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_bootstrap_robustness_histogram(data, path):
    """
    Expected columns:
        - rmse
    """
    plt.figure(figsize=(5, 4))
    plt.hist(
        data["rmse"].dropna(),
        bins=20,
        alpha=0.7,
    )

    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.savefig(path.with_suffix(".pdf"), dpi=250)
    plt.close()


def create_all_robustness_plots(
    data_size_df,
    missing_data_df,
    perturbation_df,
    bootstrap_df,
    figures_dir,
):
    create_data_size_robustness_plot(
        data_size_df,
        figures_dir / "robustness_data_size_rmse.png",
    )
    create_missing_data_robustness_plot(
        missing_data_df,
        figures_dir / "robustness_missing_data_rmse.png",
    )
    create_perturbation_robustness_plot(
        perturbation_df,
        figures_dir / "robustness_perturbation_rmse.png",
    )
    create_bootstrap_robustness_boxplot(
        bootstrap_df,
        figures_dir / "robustness_bootstrap_boxplot.png",
    )
    create_bootstrap_robustness_histogram(
        bootstrap_df,
        figures_dir / "robustness_bootstrap_histogram.png",
    )