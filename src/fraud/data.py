"""Loading and splitting the Kaggle ``creditcard.csv`` file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]
REQUIRED = ["Time", *PCA_COLUMNS, "Amount", "Class"]


@dataclass
class Split:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame
    y_valid: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    def describe(self) -> dict[str, dict[str, float]]:
        out = {}
        for name in ("train", "valid", "test"):
            y = getattr(self, f"y_{name}")
            out[name] = {"rows": int(len(y)), "frauds": int(y.sum()), "fraud_rate": float(y.mean())}
        return out


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    # The raw file has ~1,000 exact duplicate rows; they would leak between splits.
    return df.drop_duplicates().reset_index(drop=True)


def time_split(df: pd.DataFrame, valid_frac: float = 0.15, test_frac: float = 0.15) -> Split:
    """Split by transaction time, not at random.

    Fraud patterns drift, so a model must be judged on transactions that happen
    *after* the ones it learned from. A random split would overstate performance.
    """
    if not 0 < valid_frac < 1 or not 0 < test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("fractions must be in (0, 1) and sum to less than 1")
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(df)
    train_end = int(n * (1 - valid_frac - test_frac))
    valid_end = int(n * (1 - test_frac))
    X = df.drop(columns="Class")
    y = df["Class"].astype(int)
    return Split(
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[train_end:valid_end], y.iloc[train_end:valid_end],
        X.iloc[valid_end:], y.iloc[valid_end:],
    )
