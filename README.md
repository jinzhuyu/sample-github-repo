This is an instructional GitHub repository for a sample applied macheine learning project. The project generates a synthetic storm-event dataset, performs light feature engineering, trains and compares multiple regression models to predict peak flow, and saves evaluation outputs, robustness-check results, and plots.

## Project purpose

This repository demonstrates a simple end-to-end machine learning workflow for stormwater event analysis:

1. generate synthetic stormwater event data,
2. preprocess the dataset and add an engineered feature,
3. train and compare several regression models for peak-flow prediction,
4. create diagnostic and feature-importance plots for the best-performing models, and
5. evaluate model robustness under reduced data size, missing data, perturbed inputs, and bootstrap resampling.

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

<<<<<<< HEAD
Generates a synthetic event-level dataset and saves it to:
=======
Generates a synthetic event-level dataset using NumPy random draws and saves it to `../data/raw_data/stormwater_events_sample.csv`. Variables include rainfall amount, storm duration, impervious fraction, catchment area, infiltration index, slope, peak flow, and time to peak. 
>>>>>>> d5c7f19d01fdcc2afab7fa3b563468d5e3c76329

```text
../data/raw_data/stormwater_events_sample.csv
```

The generated variables are:
Loads a CSV file, adds a single engineered feature called `runoff_coefficient_proxy`, and writes the processed dataset to `../data/processed_data/stormwater_events_features.csv`. 

### `src/model.py`

Builds a scikit-learn pipeline with median imputation, standardization, and Ridge regression. It trains on the processed dataset, evaluates predictions using MAE, RMSE, and R², and saves metrics and predictions to the `output/` folder. 

### `src/make_plots.py`

Creates parity plots, residual plots, model-comparison bar charts, and feature-importance charts using Matplotlib. The plotting utilities save both PNG and PDF versions of figures. 

### `src/run_pipeline.py`

Intended to orchestrate preprocessing, training, evaluation, and plotting in one place. It reads raw data, processes features, trains the model, saves metrics and predictions, and generates diagnostic plots. 

## Data overview

According to `data/data_README.md`, the repository separates raw and processed data to support transparency and reproducibility. That document describes the project data as synthetic stormwater-event observations and identifies a raw dataset and a processed, feature-engineered dataset. 

The code in `generate_raw_data.py` shows the raw synthetic data fields currently produced:
>>>>>>> d5c7f19d01fdcc2afab7fa3b563468d5e3c76329

- `event_id`
- `storm_duration_hr`
- `rainfall_mm`
- `impervious_frac`
- `catchment_area_ha`
- `infiltration_index`
- `slope_pct`
- `peak_flow_cms`
- `time_to_peak_min`

### `src/process_data.py`

Loads the raw dataset, adds one engineered feature called `runoff_coefficient_proxy`, and saves the processed dataset to:

```text
../data/processed_data/stormwater_events_features.csv
```

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

- The scripts use relative paths like `../data/...` and `../output/...`, so they are intended to be run from the `src/` folder structure shown above while keeping the repository layout unchanged.

## Supporting documentation

The `doc/` folder can be used for workflow guidance, checklists, and optional reporting materials to support reproducibility and project organization.
