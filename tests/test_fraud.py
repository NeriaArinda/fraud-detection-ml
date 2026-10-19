import json

import numpy as np
import pandas as pd
import pytest

from fraud.data import PCA_COLUMNS, load, time_split
from fraud.evaluate import CostModel, best_threshold, expected_cost, report
from fraud.features import TransactionFeatures
from fraud.predict import score
from fraud.train import run


def synthetic(n: int = 4000, fraud_rate: float = 0.03, seed: int = 0) -> pd.DataFrame:
    """Data shaped like creditcard.csv where V1 and V2 carry the fraud signal."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < fraud_rate).astype(int)
    df = pd.DataFrame(rng.normal(size=(n, 28)), columns=PCA_COLUMNS)
    df["V1"] += y * 3
    df["V2"] -= y * 2
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)))
    df["Amount"] = np.round(rng.exponential(80, n) + y * 150, 2)
    df["Class"] = y
    return df


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "creditcard.csv"
    df = synthetic()
    pd.concat([df, df.head(5)]).to_csv(path, index=False)  # include duplicates
    return path


def test_load_drops_duplicates_and_validates(csv, tmp_path):
    assert len(load(csv)) == 4000
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load(bad)


def test_time_split_is_chronological():
    split = time_split(synthetic(1000))
    assert split.X_train["Time"].max() <= split.X_valid["Time"].min()
    assert split.X_valid["Time"].max() <= split.X_test["Time"].min()
    assert len(split.y_train) + len(split.y_valid) + len(split.y_test) == 1000
    with pytest.raises(ValueError):
        time_split(synthetic(100), 0.6, 0.5)


def test_features():
    df = synthetic(200).drop(columns="Class")
    df.loc[0, "Amount"] = 0
    out = TransactionFeatures().fit(df).transform(df)
    assert "Time" not in out and "Amount" not in out
    assert out.loc[0, "is_zero_amount"] == 1
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1)
    assert abs(out["amount_z"].mean()) < 1e-9


def test_cost_and_threshold():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    amounts = np.array([10, 10, 100, 200])
    costs = CostModel(review_cost=5)
    assert expected_cost(y, scores, amounts, 0.5, costs) == 10  # two reviews, nothing missed
    assert expected_cost(y, scores, amounts, 0.95, costs) == 300  # everything missed
    t = best_threshold(y, scores, amounts, costs)
    assert 0.2 < t <= 0.8
    r = report(y, scores, amounts, t, costs)
    assert r.recall == 1 and r.precision == 1 and r.savings == 290


def test_end_to_end_training(csv, tmp_path):
    out = tmp_path / "artifacts"
    summary = run(csv, out, only=["logistic_regression"])
    assert summary["best_model"] == "logistic_regression"
    assert summary["test"]["pr_auc"] > 0.8  # the synthetic signal is strong
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["threshold"] == summary["threshold"]
    assert (out / "pr_curves.png").exists()

    import joblib

    scored = score(joblib.load(out / "model.joblib"), synthetic(300, seed=1))
    assert {"fraud_probability", "alert"} <= set(scored.columns)
    assert scored["fraud_probability"].between(0, 1).all()
