"""
train_lgbm.py - Step 3: first real model for "defect again in this system within 90 days?"

Trains LightGBM on the train split, uses the validation split (a LATER year) to stop
training at the right time, and reports the final score ONCE on the untouched test split.
Everything is compared against the simple rule baseline.

Usage:
    pip install lightgbm scikit-learn
    python train_lgbm.py --in-dir processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

CATEGORICAL = ["ata", "AircraftMake", "AircraftModel"]
NUMERIC = ["hours", "cycles", "n_all_last30", "n_all_last90", "n_all_last365",
           "n_ata_last30", "n_ata_last90", "n_ata_last365", "n_all_before",
           "days_since_prev_same_ata", "month"]
FEATURES = CATEGORICAL + NUMERIC   # NOTE: aircraft_id and date are deliberately NOT features.


def report(name: str, y_true, prob, threshold: float = 0.5) -> None:
    pred = (prob >= threshold).astype(int)
    print(f"{name:<22} acc={accuracy_score(y_true, pred):.3f}  "
          f"precision={precision_score(y_true, pred):.3f}  recall={recall_score(y_true, pred):.3f}  "
          f"AUC={roc_auc_score(y_true, prob):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    ap.add_argument("--model-out", type=Path, default=Path("models/risk_lgbm.joblib"))
    args = ap.parse_args()

    df = pd.read_parquet(args.in_dir / "risk_dataset.parquet")
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")   # LightGBM handles categories natively

    train, val, test = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    model = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(
        train[FEATURES], train["label"],
        eval_set=[(val[FEATURES], val["label"])],
        callbacks=[lgb.early_stopping(50, verbose=False)],   # stop when validation stops improving
    )
    print(f"best number of trees: {model.best_iteration_}\n")

    y = test["label"]
    prob = model.predict_proba(test[FEATURES])[:, 1]
    rule = (test["n_ata_last90"] >= 2).astype(float)

    print("TEST SET (2025 -> June 2026):")
    print(f"{'always say no':<22} acc={(y == 0).mean():.3f}")
    report("rule: n_ata_last90>=2", y, rule.to_numpy())
    report("LightGBM", y, prob)

    imp = pd.Series(model.booster_.feature_importance("gain"), index=FEATURES)
    print("\nTop features (by gain):")
    print((imp / imp.sum()).sort_values(ascending=False).head(8).round(3).to_string())

    # Save everything the API needs: the model, the exact feature order, and the category lists
    # (LightGBM maps categories to numbers, so the API must use the same lists).
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "features": FEATURES,
        "categories": {c: [str(x) for x in df[c].cat.categories] for c in CATEGORICAL},
        "horizon_days": 90,
    }
    joblib.dump(bundle, args.model_out)
    print(f"\nsaved risk model to {args.model_out}")


if __name__ == "__main__":
    main()