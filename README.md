# Stormwater Peak-Flow Prediction: Sample Course Repository

## Overview
This sample repository is a fully reproducible template that students can use to organize a data-driven project professionally. It demonstrates a clean folder structure, clear documentation, functional-style Python code, sample data, and generated outputs.

The example project predicts **peak stormwater flow** from simple catchment and rainfall descriptors using a baseline regression model. The scientific problem is intentionally simple so students can focus on **good repository practice**, reproducibility, and code organization.

## Objectives
The repository shows how to:
1. organize a project into logical folders;
2. keep raw and processed data separate;
3. write modular Python scripts in a functional style;
4. generate reproducible outputs such as tables and figures; and
5. document the workflow so another person can run the project end to end.

## Repository Structure
```text
sample_student_repo/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── stormwater_events_sample.csv
│   └── processed/
│       └── stormwater_events_features.csv
├── src/
│   ├── utils.py
│   ├── make_dataset.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── run_pipeline.py
├── outputs/
│   ├── figures/
│   │   ├── parity_plot.png
│   │   └── residual_plot.png
│   └── tables/
│       ├── metrics.csv
│       └── test_predictions.csv
└── docs/
    ├── repository_checklist.md
    ├── student_workflow.md
    ├── literature/
    └── report/
        └── report_template.md
```

## Data
The dataset in this example is **synthetic** and included directly in the repository so students can run the workflow immediately.

The raw file `data/raw/stormwater_events_sample.csv` contains event-level predictors:
- storm duration,
- rainfall depth,
- impervious fraction,
- catchment area,
- infiltration index,
- slope,
- observed peak flow, and
- time to peak.

The processed file is generated from the raw file and adds a derived feature for demonstration purposes.

## Methods
The workflow consists of four steps:
1. read and validate the raw data;
2. create a processed feature table;
3. split the data into train and test sets;
4. fit a baseline ridge regression model and evaluate it using MAE, RMSE, and R².

All scripts are written in a **functional programming style**:
- each script exposes small reusable functions;
- data are passed explicitly between functions;
- there is minimal hidden state; and
- execution happens through a `main()` entry point.

## Requirements
Install Python 3.10+ and then install the packages listed in `requirements.txt`.

## How to Run the Project
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_pipeline.py
```

After running the pipeline, check:
- `outputs/tables/metrics.csv`
- `outputs/tables/test_predictions.csv`
- `outputs/figures/parity_plot.png`
- `outputs/figures/residual_plot.png`

## Expected Results
With the included synthetic dataset and default random seed, the baseline model should produce a reasonably strong fit. The exact values may vary slightly across environments, but the example outputs included in the repository were generated successfully.

## Suggestions for Students
Students can adapt this template by replacing:
- the synthetic dataset with their own dataset,
- the baseline model with their own method,
- the example report template with their project report, and
- the simple figures with their own final results.

## Authors
Sample instructional repository prepared for teaching repository organization and reproducible workflows.
