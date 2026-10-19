"""Score new transactions with a saved model.

    fraud-predict artifacts/model.joblib transactions.csv --out scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def score(bundle: dict, df: pd.DataFrame) -> pd.DataFrame:
    features = df.drop(columns=["Class"], errors="ignore")
    out = df.copy()
    out["fraud_probability"] = bundle["model"].predict_proba(features)[:, 1]
    out["alert"] = out["fraud_probability"] >= bundle["threshold"]
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("scored.csv"))
    args = parser.parse_args(argv)

    bundle = joblib.load(args.model)
    scored = score(bundle, pd.read_csv(args.csv))
    scored.to_csv(args.out, index=False)
    print(f"{len(scored):,} transactions scored with {bundle['name']}; "
          f"{int(scored['alert'].sum()):,} alerts -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
