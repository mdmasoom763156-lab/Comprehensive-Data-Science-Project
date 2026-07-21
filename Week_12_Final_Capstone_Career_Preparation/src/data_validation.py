"""Dataset loading, validation, and data-quality summaries."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from .config import ID_COLUMN, RAW_FEATURES, TARGET


EXPECTED_COLUMNS = [ID_COLUMN, *RAW_FEATURES, TARGET]


def load_and_validate(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(frame.columns)
    unexpected = set(frame.columns) - set(EXPECTED_COLUMNS)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"Unexpected columns: {sorted(unexpected)}")
    frame = frame[EXPECTED_COLUMNS].copy()
    numeric = ["Tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen", TARGET]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty:
        raise ValueError("Dataset is empty")
    if frame[ID_COLUMN].duplicated().any():
        raise ValueError("CustomerID must be unique")
    if frame.isna().any().any():
        raise ValueError("Dataset contains missing values")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Churn must contain only 0 and 1")
    if (frame[["Tenure", "MonthlyCharges", "TotalCharges"]] < 0).any().any():
        raise ValueError("Numeric charge and tenure fields cannot be negative")
    return frame


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def quality_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Column": frame.columns,
            "Data_Type": [str(frame[c].dtype) for c in frame.columns],
            "Missing": [int(frame[c].isna().sum()) for c in frame.columns],
            "Unique": [int(frame[c].nunique(dropna=False)) for c in frame.columns],
        }
    )
