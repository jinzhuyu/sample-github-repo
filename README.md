# sample-github-repo

A small, instructional Githubrepository for a stormwater analytics workflow. The project generates a synthetic storm-event dataset, performs light feature engineering, trains a regression model to predict peak flow, and saves evaluation outputs and plots.  

## Project purpose

This repository demonstrates a simple end-to-end machine learning pipeline for stormwater event analysis:

1. generate synthetic stormwater event data,
2. preprocess the dataset and add engineered features,
3. train and evaluate a Ridge regression model for peak-flow prediction, and
4. create diagnostic figures from predictions. 

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
│   ├── figures/
│   ├── metrics.csv
│   ├── predictions.csv
│   └── other saved outputs
├── src/
│   ├── generate_raw_data.py
│   ├── make_plots.py
│   ├── model.py
│   ├── process_data.py
│   └── run_pipeline.py
├── LICENSE
├── README.md
└── requirements.txt
```

The top-level folders and several supporting documentation files are visible in the repository tree, including `data`, `doc`, `output`, and `src`. The output folder already contains saved prediction and metrics files as well as figure assets. 

## What each script does

### `src/generate_raw_data.py`

Generates a synthetic event-level dataset using NumPy random draws and saves it to `../data/raw_data/stormwater_events_sample.csv`. Variables include rainfall amount, storm duration, impervious fraction, catchment area, infiltration index, slope, peak flow, and time to peak. citeturn689219view0

### `src/process_data.py`

Loads a CSV file, adds a single engineered feature called `runoff_coefficient_proxy`, and writes the processed dataset to `../data/processed_data/stormwater_events_features.csv`. 

### `src/model.py`

Builds a scikit-learn pipeline with median imputation, standardization, and Ridge regression. It trains on the processed dataset, evaluates predictions using MAE, RMSE, and R², and saves metrics and predictions to the `output/` folder. 

### `src/make_plots.py`

Creates parity plots, residual plots, model-comparison bar charts, and feature-importance charts using Matplotlib. The plotting utilities save both PNG and PDF versions of figures. 

### `src/run_pipeline.py`

Intended to orchestrate preprocessing, training, evaluation, and plotting in one place. It reads raw data, processes features, trains the model, saves metrics and predictions, and generates diagnostic plots. 

## Data overview

According to `data/data_README.md`, the repository separates raw and processed data to support transparency and reproducibility. That document describes the project data as synthetic stormwater-event observations and identifies a raw dataset and a processed, feature-engineered dataset. citeturn346430view3

The code in `generate_raw_data.py` shows the raw synthetic data fields currently produced:

- `event_id`
- `storm_duration_hr`
- `rainfall_mm`
- `impervious_frac`
- `catchment_area_ha`
- `infiltration_index`
- `slope_pct`
- `peak_flow_cms`
- `time_to_peak_min` 

The processed data currently add:

- `runoff_coefficient_proxy`

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependency list in the repository contains NumPy, pandas, Matplotlib, and scikit-learn. 

## Recommended usage

Run the workflow in stages from the repository root.

### 1. Generate synthetic raw data

```bash
python src/generate_raw_data.py
```

### 2. Process the data

```bash
python src/process_data.py
```

### 3. Train and evaluate the model and then generate plots about model accuracy and feature importance

```bash
python src/run_pipeline.py
```

## Expected outputs

The repository already shows typical output artifacts, including metrics files, prediction files, and figures. Examples visible in `output/` include `metrics.csv`, `predictions.csv`, `all_model_predictions.csv`, `best_model_predictions.csv`, feature-importance CSV files, and a `figures/` subfolder. 

## Supporting documentation

The `doc/` folder includes a repository checklist, a suggested workflow, and a report template. Those files emphasize modular scripts, data documentation, reproducibility, and storing final deliverables in the documentation/report area. 
