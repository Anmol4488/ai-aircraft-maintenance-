"""
build_text_dataset.py - Step 4: dataset for the SmolLM2 text task.

Task 1: read a defect report's text -> predict the system (ATA chapter, e.g. "53" = fuselage).

Same date-based split as the risk model. Extra rule to avoid fake-good scores:
validation/test texts that also appear EXACTLY in the training set are removed
(many reports are near-copy-paste, and the model would just memorise them).

Usage:
    python build_text_dataset.py --in-dir processed --min-class-count 1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    ap.add_argument("--min-class-count", type=int, default=1000,
                    help="keep only systems with at least this many reports")
    ap.add_argument("--train-size", type=int, default=60_000, help="subsample for a fast first run")
    ap.add_argument("--eval-size", type=int, default=6_000)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_dir / "reports_clean.parquet",
                         columns=["date", "ata", "AircraftMake", "AircraftModel", "Discrepancy"])
    df = df.rename(columns={"Discrepancy": "text", "ata": "label"})
    df = df.dropna(subset=["label", "text"])

    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_class_count].index
    print(f"systems kept: {len(keep)} of {len(counts)} (cover {counts[keep].sum() / counts.sum():.1%} of reports)")
    df = df[df["label"].isin(keep)]

    df["split"] = "train"
    df.loc[df["date"] > TRAIN_END, "split"] = "val"
    df.loc[df["date"] > VAL_END, "split"] = "test"

    train_texts = set(df.loc[df["split"] == "train", "text"])
    leak = (df["split"] != "train") & df["text"].isin(train_texts)
    print(f"removed val/test rows whose exact text is in train: {int(leak.sum())}")
    df = df[~leak]

    parts = []
    for name, n in (("train", args.train_size), ("val", args.eval_size), ("test", args.eval_size)):
        part = df[df["split"] == name]
        parts.append(part.sample(min(n, len(part)), random_state=42))
    out = pd.concat(parts).reset_index(drop=True)

    out.to_parquet(args.in_dir / "text_dataset.parquet", index=False)
    print(out.groupby("split").size().to_string())
    print("\nmost common systems in train:")
    print(out[out.split == "train"]["label"].value_counts(normalize=True).head(6).round(3).to_string())


if __name__ == "__main__":
    main()