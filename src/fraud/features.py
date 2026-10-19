"""Feature engineering as a scikit-learn transformer, so it travels with the model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SECONDS_PER_DAY = 24 * 60 * 60


class TransactionFeatures(BaseEstimator, TransformerMixin):
    """Adds features derived from ``Time`` and ``Amount``.

    * ``log_amount``  - amounts are heavily right-skewed
    * ``hour_sin`` / ``hour_cos`` - time of day on a circle (23:59 is close to 00:00)
    * ``amount_z`` - how unusual the amount is versus the training data
    * ``is_zero_amount`` - $0 authorisations are a known card-testing signal

    The raw ``Time`` column (seconds since the first transaction) is dropped
    because it only encodes position in the file.
    """

    def fit(self, X: pd.DataFrame, y=None):
        log_amount = np.log1p(X["Amount"].to_numpy(dtype=float))
        self.amount_mean_ = float(log_amount.mean())
        self.amount_std_ = float(log_amount.std()) or 1.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.drop(columns=["Time"]).copy()
        log_amount = np.log1p(X["Amount"].to_numpy(dtype=float))
        hour = (X["Time"].to_numpy(dtype=float) % SECONDS_PER_DAY) / 3600.0
        out["log_amount"] = log_amount
        out["amount_z"] = (log_amount - self.amount_mean_) / self.amount_std_
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_zero_amount"] = (X["Amount"].to_numpy() == 0).astype(int)
        return out.drop(columns=["Amount"])

    def get_feature_names_out(self, input_features=None):
        base = [c for c in (input_features or []) if c not in ("Time", "Amount")]
        return np.array(base + ["log_amount", "amount_z", "hour_sin", "hour_cos", "is_zero_amount"])
