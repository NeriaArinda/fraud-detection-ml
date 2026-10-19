"""Candidate models. Every one is a full Pipeline: features -> scaler -> classifier."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import TransactionFeatures


def candidates(random_state: int = 42) -> dict[str, Pipeline]:
    """Models compared during training.

    Class imbalance (~0.17% fraud) is handled with class weights rather than
    oversampling, so the probabilities stay meaningful for threshold tuning.
    """
    return {
        "logistic_regression": Pipeline([
            ("features", TransactionFeatures()),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)),
        ]),
        "random_forest": Pipeline([
            ("features", TransactionFeatures()),
            ("model", RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                class_weight="balanced_subsample", n_jobs=-1, random_state=random_state,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("features", TransactionFeatures()),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
                class_weight="balanced", early_stopping=True, validation_fraction=0.1,
                random_state=random_state,
            )),
        ]),
    }
