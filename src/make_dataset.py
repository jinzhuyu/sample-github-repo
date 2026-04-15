from __future__ import annotations

import pandas as pd

from utils import PROCESSED_DATA_PATH, RAW_DATA_PATH, read_csv, write_csv

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["runoff_coefficient_proxy"] = (
        0.25 + 0.65 * result["impervious_frac"] - 0.003 * result["infiltration_index"]
    ).round(3)
    return result

def process_raw_dataset() -> pd.DataFrame:
    raw_df = read_csv(RAW_DATA_PATH)
    processed_df = add_engineered_features(raw_df)
    write_csv(processed_df, PROCESSED_DATA_PATH)
    return processed_df

def main() -> None:
    process_raw_dataset()

if __name__ == "__main__":
    main()
