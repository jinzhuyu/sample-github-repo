# Stormwater ML Workflow Repository

This is an instructional GitHub repository for a sample applied machine learning project in stormwater analytics. The project generates a synthetic storm-event dataset, performs light feature engineering, trains and compares multiple regression models to predict peak flow, and saves evaluation outputs, robustness-check results, and plots.

The repository is designed for teaching and demonstration rather than production deployment. Its main value is to show a clear, reproducible end-to-end workflow that starts from data generation and ends with model evaluation, visualization, and robustness analysis.

## Project purpose

This repository demonstrates a simple end-to-end machine learning workflow for stormwater event analysis:

1. generate synthetic stormwater event data,
2. preprocess the dataset and add an engineered feature,
3. train and compare several regression models for peak-flow prediction,
4. create diagnostic and feature-importance plots for the best-performing models, and
5. evaluate model robustness under reduced data size, missing data, perturbed inputs, and bootstrap resampling.

## Workflow diagram

```mermaid
flowchart TD
    A[Generate synthetic raw data\n`src/generate_raw_data.py`] --> B[Process raw data and add\nengineered feature\n`src/process_data.py`]
    B --> C[Train and compare models\n`src/model.py`]
    C --> D[Save metrics, predictions,\nand feature importance tables]
    C --> E[Generate diagnostic figures\n`src/plot_accuracy_features.py`]
    C --> F[Run robustness checks\n`src/robustness_checks.py`]
    E --> G[Accuracy figures\n`output/accuracy/figures/`]
    F --> H[Robustness summaries and figures\n`output/robustness/figures/`]
```

## Repository structure

```text
sample-github-repo/
├── data/
│   ├── raw_data/
│   ├── processed_data/
│   └── data_README.md
├── doc/
│   ├── github_repo_guideline.pdf
│   ├── report_template_not_required.md
│   ├── repository_checklist.md
│   └── suggested_workflow.md
├── output/
│   ├── accuracy/
│   │   ├── figures/
│   │   ├── all_model_predictions.csv
│   │   ├── best_model_predictions.csv
│   │   ├── feature_importance_random_forest.csv
│   │   ├── feature_importance_xgboost.csv
│   │   └── model_comparison_metrics.csv
│   └── robustness/
│       └── figures/
├── src/
│   ├── generate_raw_data.py
│   ├── make_plots.py
│   ├── model.py
│   ├── plot_accuracy_features.py
│   ├── process_data.py
│   └── robustness_checks.py
├── LICENSE
├── README.md
└── requirements.txt
```

## What each script does

### `src/generate_raw_data.py`

Generates a synthetic event-level dataset and saves it to:

```text
../data/raw_data/stormwater_events_sample.csv
```

The generated variables are:

- `event_id`
- `storm_duration_hr`
- `rainfall_mm`
- `impervious_frac`
- `catchment_area_ha`
- `infiltration_index`
- `slope_pct`
- `peak_flow_cms`
- `time_to_peak_min`

This script is useful for demonstrating the workflow when real field observations are unavailable or when a lightweight instructional example is preferred.

### `src/process_data.py`

Loads the raw dataset, adds one engineered feature called `runoff_coefficient_proxy`, and saves the processed dataset to:

```text
../data/processed_data/stormwater_events_features.csv
```

This step separates raw data generation from feature preparation, which makes the workflow easier to understand, debug, and reproduce.

### `src/model.py`

Loads the processed dataset, splits data into training and test sets, preprocesses features with median imputation and standardization, and compares the following models:

- Ridge
- SVR
- Random Forest
- XGBoost

It evaluates each model using:

- MAE
- RMSE
- R²

It saves outputs to:

```text
../output/accuracy/model_comparison_metrics.csv
../output/accuracy/all_model_predictions.csv
../output/accuracy/best_model_predictions.csv
../output/accuracy/feature_importance_random_forest.csv
../output/accuracy/feature_importance_xgboost.csv
```

This script is the core modeling component of the repository. It supports side-by-side comparison of multiple regression approaches rather than focusing on a single model only.

### `src/make_plots.py`

Provides plotting utilities used by the plotting and robustness scripts. It creates:

- model-comparison bar plots,
- parity plots,
- residual plots,
- feature-importance plots,
- robustness error-bar plots, and
- bootstrap boxplots and histograms.

Plots are saved in both PNG and PDF formats.

### `src/plot_accuracy_features.py`

Loads saved outputs from `model.py` and creates:

- model comparison plots,
- a parity plot for the best model,
- a residual plot for the best model, and
- feature-importance plots for Random Forest and XGBoost if those files are available.

Figures are saved under:

```text
../output/accuracy/figures/
```

### `src/robustness_checks.py`

Runs robustness checks for the Random Forest model only. The implemented checks are:

1. data-size robustness,
2. missing-data robustness,
3. noise / perturbation robustness, and
4. bootstrap robustness.

The script tunes the Random Forest once, reuses the tuned hyperparameters across robustness experiments, saves CSV outputs, summarizes results, and generates robustness plots.

By default, outputs are saved under:

```text
../output/robustness/figures/
```

## Data overview

The raw synthetic dataset contains storm-event variables such as rainfall, storm duration, imperviousness, infiltration index, slope, peak flow, and time to peak. The processed dataset adds one engineered predictor, `runoff_coefficient_proxy`, for downstream modeling.

Because the dataset is synthetic, the repository is well suited for instruction, code testing, and workflow demonstration. However, results should not be interpreted as field-validated findings.

## Installation

Create and activate a virtual environment, then install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This project requires at least the following Python packages:

- numpy
- pandas
- matplotlib
- scikit-learn
- xgboost

## Recommended usage

Run the workflow in stages from the repository root.

### 1. Generate synthetic raw data

```bash
python src/generate_raw_data.py
```

### 2. Process the raw data

```bash
python src/process_data.py
```

### 3. Train and compare models

```bash
python src/model.py
```

### 4. Generate accuracy and feature plots

```bash
python src/plot_accuracy_features.py
```

### 5. Run robustness checks

```bash
python src/robustness_checks.py
```

You can also change the default robustness settings from the command line. For example:

```bash
python src/robustness_checks.py --bootstrap-runs 20
```

or

```bash
python src/robustness_checks.py \
  --data ../data/processed_data/stormwater_events_features.csv \
  --outdir ../output/robustness/figures \
  --bootstrap-runs 10
```

## Expected outputs

After running the full workflow, the repository should contain outputs such as:

### Accuracy outputs

- `output/accuracy/model_comparison_metrics.csv`
- `output/accuracy/all_model_predictions.csv`
- `output/accuracy/best_model_predictions.csv`
- `output/accuracy/feature_importance_random_forest.csv`
- `output/accuracy/feature_importance_xgboost.csv`
- `output/accuracy/figures/*.png`
- `output/accuracy/figures/*.pdf`

### Robustness outputs

- `output/robustness/figures/robustness_data_size.csv`
- `output/robustness/figures/robustness_missing_data.csv`
- `output/robustness/figures/robustness_perturbation.csv`
- `output/robustness/figures/robustness_bootstrap.csv`
- `output/robustness/figures/robustness_all_runs.csv`
- `output/robustness/figures/robustness_summary.csv`
- `output/robustness/figures/*.png`
- `output/robustness/figures/*.pdf`

## Notes

- The scripts use relative paths like `../data/...` and `../output/...`, so the repository structure should remain unchanged.
- For the smoothest execution, run commands from the repository root unless the scripts are explicitly written for another working directory.
- If some optional outputs are missing, check whether the corresponding upstream script completed successfully.

## Supporting documentation

The `doc/` folder can be used for workflow guidance, checklists, and optional reporting materials to support reproducibility and project organization.
