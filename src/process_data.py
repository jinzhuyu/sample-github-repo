import pandas as pd


def load_data(file_path):
    """
    Load CSV data from disk.

    This function is reused across scripts to keep data loading consistent.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def add_engineered_features(df):
    """
    Add derived (engineered) features.

    Feature engineering transforms raw variables into more informative inputs
    for machine learning models.

    Here, we construct a proxy for the runoff coefficient:
    - increases with impervious surface (more runoff)
    - decreases with infiltration (less runoff)
    """
    result = df.copy()

    result["runoff_coefficient_proxy"] = (
        0.25
        + 0.65 * result["impervious_frac"]     # urbanization effect
        - 0.003 * result["infiltration_index"] # infiltration reduces runoff
    ).round(3)

    return result


def process_data(df):
    """
    Apply all preprocessing steps.

    This wrapper makes it easy to extend the pipeline later
    (e.g., add normalization, filtering, or additional features).
    """
    return add_engineered_features(df)


def save_data(df, file_path):
    """
    Save processed DataFrame to CSV.

    This creates a clean dataset that will be used by model.py.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def main():
    """
    Process raw data into model-ready data.

    Workflow step:
    raw data → feature engineering → processed data
    """
    raw_path = "../data/raw_data/stormwater_events_sample.csv"
    out_path = "../data/processed_data/stormwater_events_features.csv"

    # Load raw dataset
    df = load_data(raw_path)
    if df is None:
        return

    # Apply feature engineering
    df = process_data(df)

    # Save processed dataset for modeling
    save_data(df, out_path)


if __name__ == "__main__":
    main()