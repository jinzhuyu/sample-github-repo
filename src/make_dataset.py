import pandas as pd


def load_data(file_path):
    """Load CSV data."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features."""
    result = df.copy()
    result["runoff_coefficient_proxy"] = (
        0.25
        + 0.65 * result["impervious_frac"]
        - 0.003 * result["infiltration_index"]
    ).round(3)
    return result


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline wrapper."""
    return add_engineered_features(df)


def save_data(df: pd.DataFrame, file_path):
    """Save DataFrame to CSV."""
    try:
        df.to_csv(file_path, index=False)
        print(f"Saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")


def main():
    raw_path = "../data/raw_data/stormwater_events_sample.csv"
    out_path = "../data/processed_data/stormwater_events_features.csv"

    df = load_data(raw_path)
    if df is None:
        return

    df = process_data(df)
    save_data(df, out_path)


if __name__ == "__main__":
    main()