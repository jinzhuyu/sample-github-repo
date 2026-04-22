from __future__ import annotations

"""
Robustness checks implemented for Random Forest only:
1) Data size robustness
2) Missing-data robustness
3) Noise / perturbation robustness
4) Bootstrap robustness
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from make_plots import create_all_robustness_plots


RANDOM_STATE = 123


def get_features():
    """Return input feature names."""
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


def evaluate_predictions(y_true, y_pred):
    """Compute prediction metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
    }


def load_processed_data(path):
    """Load processed CSV and check required columns."""
    df = pd.read_csv(path)

    required = get_features() + [get_target()]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    return df


def build_preprocessor(features):
    """Build preprocessing pipeline for numeric features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, features)]
    )


def build_random_forest_search(features):
    """Build Random Forest hyperparameter search."""
    preprocessor = build_preprocessor(features)

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 4],
    }

    return GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1,
    )


def tune_random_forest_once(df):
    """Tune Random Forest once on the original dataset."""
    X = df[get_features()]
    y = df[get_target()]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    search = build_random_forest_search(get_features())
    search.fit(X_train, y_train)

    best_params = search.best_params_
    return best_params


def build_fixed_random_forest(features, best_params):
    """Build a fixed Random Forest pipeline using pre-tuned parameters."""
    preprocessor = build_preprocessor(features)

    model = RandomForestRegressor(
        n_estimators=best_params["model__n_estimators"],
        max_depth=best_params["model__max_depth"],
        min_samples_split=best_params["model__min_samples_split"],
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_random_forest(df, best_params, random_state=RANDOM_STATE):
    """Train fixed Random Forest and return evaluation metrics."""
    X = df[get_features()]
    y = df[get_target()]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    model = build_fixed_random_forest(get_features(), best_params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = evaluate_predictions(y_test, y_pred)
    metrics["best_params"] = str(best_params)

    return metrics, model


# --------------------------------------------------
# 1) Data size robustness
# --------------------------------------------------
def data_size_robustness(
    df,
    best_params,
    fractions=(0.4, 0.6, 0.8, 1.0),
    repeats=3,
):
    """Test robustness to reduced sample size."""
    rows = []
    n = len(df)

    for frac in fractions:
        sample_n = max(30, int(round(frac * n)))

        for rep in range(repeats):
            seed = RANDOM_STATE + rep

            sampled = df.sample(
                n=sample_n,
                replace=False,
                random_state=seed,
            )

            metrics, _ = train_random_forest(
                sampled,
                best_params=best_params,
                random_state=seed,
            )

            metrics["check"] = "data_size"
            metrics["train_fraction"] = frac
            metrics["sample_n"] = sample_n
            metrics["repeat"] = rep

            rows.append(metrics)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 2) Missing-data robustness
# --------------------------------------------------
def inject_missingness(df, rate, features, random_state):
    """Randomly inject missing values into feature columns."""
    rng = np.random.default_rng(random_state)
    out = df.copy()

    for col in features:
        mask = rng.random(len(out)) < rate
        out.loc[mask, col] = np.nan

    return out


def missing_data_robustness(
    df,
    best_params,
    missing_rates=(0.10, 0.20, 0.30),
    repeats=3,
):
    """Test robustness under missing feature values."""
    rows = []
    features = get_features()

    for rate in missing_rates:
        for rep in range(repeats):
            seed = RANDOM_STATE + rep

            df_miss = inject_missingness(
                df,
                rate,
                features,
                seed,
            )

            metrics, _ = train_random_forest(
                df_miss,
                best_params=best_params,
                random_state=seed,
            )

            metrics["check"] = "missing_data"
            metrics["missing_rate"] = rate
            metrics["repeat"] = rep

            rows.append(metrics)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 3) Noise / perturbation robustness
# --------------------------------------------------
def perturb_numeric_features(
    df,
    noise_scale,
    features,
    random_state,
):
    """Add Gaussian noise to numeric features."""
    rng = np.random.default_rng(random_state)
    out = df.copy()

    for col in features:
        std = out[col].std(ddof=0)

        if std == 0 or pd.isna(std):
            continue

        noise = rng.normal(0.0, noise_scale * std, len(out))
        out[col] = out[col] + noise

    return out


def perturbation_robustness(
    df,
    best_params,
    noise_scales=(0.025, 0.05, 0.10),
    repeats=3,
):
    """Test robustness to noisy feature inputs."""
    rows = []
    features = get_features()

    for scale in noise_scales:
        for rep in range(repeats):
            seed = RANDOM_STATE + rep

            df_perturbed = perturb_numeric_features(
                df,
                scale,
                features,
                seed,
            )

            metrics, _ = train_random_forest(
                df_perturbed,
                best_params=best_params,
                random_state=seed,
            )

            metrics["check"] = "perturbation"
            metrics["noise_scale_sd_units"] = scale
            metrics["repeat"] = rep

            rows.append(metrics)

    return pd.DataFrame(rows)


# --------------------------------------------------
# 4) Bootstrap robustness
# --------------------------------------------------
def bootstrap_robustness(df, best_params, n_boot=5):
    """
    Test robustness using bootstrap resampling.

    Use a small number of bootstraps by default to limit runtime,
    but this can be increased as needed.
    """
    rows = []

    for b in range(n_boot):
        seed = RANDOM_STATE + b

        boot = df.sample(
            frac=1.0,
            replace=True,
            random_state=seed,
        )

        metrics, _ = train_random_forest(
            boot,
            best_params=best_params,
            random_state=seed,
        )

        metrics["check"] = "bootstrap"
        metrics["bootstrap_id"] = b

        rows.append(metrics)

    return pd.DataFrame(rows)


# --------------------------------------------------
# Summary helpers
# --------------------------------------------------
def summarize_robustness(results):
    """Summarize robustness results across runs."""
    group_cols = ["check"]

    for col in ["train_fraction", "missing_rate", "noise_scale_sd_units"]:
        if col in results.columns:
            group_cols.append(col)

    summary = (
        results.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("rmse", "size"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
        )
        .reset_index()
    )

    return summary


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    """Run all robustness checks and save outputs."""

    # --------------------------------------------------
    # 1. Parse command-line arguments
    # --------------------------------------------------
    # This allows the script to be flexible:
    # users can change input data, output folder, and number of bootstrap runs
    parser = argparse.ArgumentParser(description="Run robustness checks.")

    parser.add_argument(
        "--data",
        type=str,
        default="../data/processed_data/stormwater_events_features.csv",
        help="Path to processed CSV file",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="../output/robustness/figures",
        help="Output directory for robustness CSVs",
    )
    parser.add_argument(
        "--bootstrap-runs",
        type=int,
        default=5,
        help="Number of bootstrap replications",
    )

    args = parser.parse_args()


    # --------------------------------------------------
    # 2. Load data
    # --------------------------------------------------
    # Load dataset and ensure required columns exist
    df = load_processed_data(args.data)


    # --------------------------------------------------
    # 3. Prepare output directory
    # --------------------------------------------------
    # Create output folder if it does not exist
    # (parents=True allows nested folders)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)


    # --------------------------------------------------
    # 4. Tune model ONCE (critical for efficiency)
    # --------------------------------------------------
    # Instead of tuning inside every robustness experiment,
    # we tune the Random Forest once on the full dataset
    # and reuse the best hyperparameters.
    best_params = tune_random_forest_once(df)

    print("Best parameters from one-time tuning:")
    print(best_params)


    # --------------------------------------------------
    # 5. Run robustness checks
    # --------------------------------------------------
    # Each function perturbs the data in a different way:
    # - data_size: reduce training data size
    # - missing_data: introduce missing values
    # - perturbation: add noise to features
    # - bootstrap: resample dataset with replacement
    #
    # IMPORTANT:
    # All experiments use the SAME tuned model parameters,
    # so we isolate the effect of data changes only.

    data_size_df = data_size_robustness(df, best_params=best_params)
    missing_df = missing_data_robustness(df, best_params=best_params)
    perturb_df = perturbation_robustness(df, best_params=best_params)
    bootstrap_df = bootstrap_robustness(
        df,
        best_params=best_params,
        n_boot=args.bootstrap_runs,
    )


    # --------------------------------------------------
    # 6. Save individual results
    # --------------------------------------------------
    # Each CSV corresponds to one type of robustness test
    data_size_df.to_csv(outdir / "robustness_data_size.csv", index=False)
    missing_df.to_csv(outdir / "robustness_missing_data.csv", index=False)
    perturb_df.to_csv(outdir / "robustness_perturbation.csv", index=False)
    bootstrap_df.to_csv(outdir / "robustness_bootstrap.csv", index=False)


    # --------------------------------------------------
    # 7. Combine all runs
    # --------------------------------------------------
    # This creates a single dataset containing ALL experiments,
    # useful for further analysis or plotting
    combined = pd.concat(
        [data_size_df, missing_df, perturb_df, bootstrap_df],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(outdir / "robustness_all_runs.csv", index=False)


    # --------------------------------------------------
    # 8. Summarize results
    # --------------------------------------------------
    # Aggregate results across repeated runs:
    # compute mean and variability (std) of metrics
    summary_df = summarize_robustness(combined)
    summary_df.to_csv(outdir / "robustness_summary.csv", index=False)


    # --------------------------------------------------
    # 9. Print key outputs
    # --------------------------------------------------
    # Helps quickly inspect results without opening CSV files
    print("Robustness checks finished.")
    print(f"Saved outputs under: {outdir.resolve()}")
    print(summary_df.to_string(index=False))


    # --------------------------------------------------
    # 10. Generate plots
    # --------------------------------------------------
    # Create visualization of robustness:
    # - error bars (mean ± std)
    # - bootstrap distributions
    #
    # These plots help interpret:
    #   sensitivity of model performance to data issues
    create_all_robustness_plots(
        data_size_df,
        missing_df,
        perturb_df,
        bootstrap_df,
        outdir,
    )


if __name__ == "__main__":
    main()