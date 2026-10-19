"""Train, compare and save the fraud models.

    fraud-train data/creditcard.csv --out artifacts
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from .data import load, time_split  # noqa: E402
from .evaluate import CostModel, best_threshold, recall_at_precision, report  # noqa: E402
from .models import candidates  # noqa: E402


def run(csv: Path, out: Path, review_cost: float = 5.0, only: list[str] | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    costs = CostModel(review_cost=review_cost)
    split = time_split(load(csv))
    print("split:", json.dumps(split.describe()))

    results: dict[str, dict] = {}
    fitted = {}
    curves = {}
    for name, pipe in candidates().items():
        if only and name not in only:
            continue
        start = time.perf_counter()
        pipe.fit(split.X_train, split.y_train)
        seconds = time.perf_counter() - start

        valid_scores = pipe.predict_proba(split.X_valid)[:, 1]
        threshold = best_threshold(split.y_valid.to_numpy(), valid_scores,
                                   split.X_valid["Amount"].to_numpy(), costs)
        valid = report(split.y_valid, valid_scores, split.X_valid["Amount"], threshold, costs)
        results[name] = {
            "fit_seconds": round(seconds, 1),
            "valid": valid.to_dict(),
            "valid_recall_at_90_precision": recall_at_precision(split.y_valid, valid_scores, 0.9),
        }
        fitted[name] = (pipe, threshold)
        curves[name] = precision_recall_curve(split.y_valid, valid_scores)
        print(f"{name:22s} PR-AUC {valid.pr_auc:.3f}  recall {valid.recall:.2f}  "
              f"precision {valid.precision:.2f}  savings ${valid.savings:,.0f}  ({seconds:.0f}s)")

    # Choose on validation savings, then report the untouched test set once.
    best = max(results, key=lambda n: results[n]["valid"]["savings"])
    model, threshold = fitted[best]
    test_scores = model.predict_proba(split.X_test)[:, 1]
    test = report(split.y_test, test_scores, split.X_test["Amount"], threshold, costs)

    summary = {
        "best_model": best,
        "threshold": threshold,
        "review_cost": review_cost,
        "split": split.describe(),
        "models": results,
        "test": test.to_dict(),
        "test_recall_at_90_precision": recall_at_precision(split.y_test, test_scores, 0.9),
    }
    joblib.dump({"model": model, "threshold": threshold, "name": best}, out / "model.joblib")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _plot_pr(curves, out / "pr_curves.png")
    _plot_importance(model, split, out / "top_features.png")
    print(f"\nbest: {best}  test PR-AUC {test.pr_auc:.3f}  recall {test.recall:.2f}  "
          f"precision {test.precision:.2f}  savings ${test.savings:,.0f}")
    return summary


def _plot_pr(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (precision, recall, _) in curves.items():
        ax.plot(recall, precision, lw=2, label=name.replace("_", " "))
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Precision-recall on validation data", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(model, split, path: Path, n: int = 15) -> None:
    """Permutation importance on a fraud-enriched validation sample."""
    from sklearn.inspection import permutation_importance

    X, y = split.X_valid, split.y_valid
    rng = np.random.default_rng(0)
    idx = np.concatenate([np.flatnonzero(y.to_numpy() == 1),
                          rng.choice(np.flatnonzero(y.to_numpy() == 0), size=min(5000, int((y == 0).sum())), replace=False)])
    result = permutation_importance(model, X.iloc[idx], y.iloc[idx], scoring="average_precision",
                                    n_repeats=5, random_state=0, n_jobs=-1)
    order = np.argsort(result.importances_mean)[::-1][:n][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(X.columns[order], result.importances_mean[order], color="#3B6EA8")
    ax.set_title("Most important inputs (drop in PR-AUC when shuffled)", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", type=Path, help="path to creditcard.csv")
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--review-cost", type=float, default=5.0, help="cost of one analyst review ($)")
    parser.add_argument("--only", nargs="*", help="train only these models")
    args = parser.parse_args(argv)
    run(args.csv, args.out, args.review_cost, args.only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
