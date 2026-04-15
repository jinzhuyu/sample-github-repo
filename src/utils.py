from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "stormwater_events_sample.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "stormwater_events_features.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "tables" / "metrics.csv"
PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "tables" / "test_predictions.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

def ensure_directories(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def select_feature_columns() -> list[str]:
    return [
        "storm_duration_hr",
        "rainfall_mm",
        "impervious_frac",
        "catchment_area_ha",
        "infiltration_index",
        "slope_pct",
    ]

def select_target_column() -> str:
    return "peak_flow_cms"

def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

def split_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    feature_columns = select_feature_columns()
    target_column = select_target_column()
    validate_required_columns(df, [*feature_columns, target_column])
    return df[feature_columns].copy(), df[target_column].copy()
