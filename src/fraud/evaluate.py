"""Metrics that matter for fraud: precision/recall trade-offs and money."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass
class CostModel:
    """Business costs used to pick the alert threshold.

    * a missed fraud costs the transaction amount (chargeback)
    * every alert costs ``review_cost`` (an analyst checks it)
    """

    review_cost: float = 5.0


@dataclass
class Report:
    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    alerts: int
    frauds_caught: int
    frauds_missed: int
    fraud_amount_caught: float
    fraud_amount_missed: float
    total_cost: float
    cost_if_no_model: float

    @property
    def savings(self) -> float:
        return self.cost_if_no_model - self.total_cost

    def to_dict(self) -> dict:
        d = asdict(self)
        d["savings"] = self.savings
        return d


def expected_cost(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                  threshold: float, costs: CostModel) -> float:
    flagged = scores >= threshold
    missed = (y == 1) & ~flagged
    return float(amounts[missed].sum() + flagged.sum() * costs.review_cost)


def best_threshold(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
                   costs: CostModel, grid: int = 400) -> float:
    """Threshold that minimises total cost. Pick it on validation data only."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.90, 1.0, grid)))
    candidates = np.concatenate([candidates, [0.5]])
    cost = [expected_cost(y, scores, amounts, t, costs) for t in candidates]
    return float(candidates[int(np.argmin(cost))])


def report(y: np.ndarray, scores: np.ndarray, amounts: np.ndarray,
           threshold: float, costs: CostModel) -> Report:
    y = np.asarray(y)
    scores = np.asarray(scores)
    amounts = np.asarray(amounts, dtype=float)
    flagged = scores >= threshold
    tp = int(((y == 1) & flagged).sum())
    fn = int(((y == 1) & ~flagged).sum())
    alerts = int(flagged.sum())
    return Report(
        pr_auc=float(average_precision_score(y, scores)),
        roc_auc=float(roc_auc_score(y, scores)),
        threshold=float(threshold),
        precision=tp / alerts if alerts else 0.0,
        recall=tp / (tp + fn) if (tp + fn) else 0.0,
        alerts=alerts,
        frauds_caught=tp,
        frauds_missed=fn,
        fraud_amount_caught=float(amounts[(y == 1) & flagged].sum()),
        fraud_amount_missed=float(amounts[(y == 1) & ~flagged].sum()),
        total_cost=expected_cost(y, scores, amounts, threshold, costs),
        cost_if_no_model=float(amounts[y == 1].sum()),
    )


def recall_at_precision(y: np.ndarray, scores: np.ndarray, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y, scores)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0
