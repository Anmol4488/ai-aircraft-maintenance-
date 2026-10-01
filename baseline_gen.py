"""
baseline_gen.py - the bars SmolLM2 must beat for "problem -> fix".

  1. constant      : always answer with the single most common fix in the training data
  2. nearest-match : find the most similar TRAINING problem (TF-IDF) and copy its fix

Scores (on test reports from 2025 onward):
  word-F1    = how many words the answer shares with the real fix
  verb match = does the answer start with the same action word as the real fix (REPLACED / REPAIRED ...)

Usage:
    python baseline_gen.py --in-dir processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from gen_utils import first_word, token_f1


def score(name: str, preds: list[str], golds: list[str]) -> None:
    f1 = np.mean([token_f1(p, g) for p, g in zip(preds, golds)])
    verb = np.mean([first_word(p) == first_word(g) for p, g in zip(preds, golds)])
    print(f"{name:<28} word-F1={f1:.3f}  verb-match={verb:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--n-train", type=int, default=60000)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_dir / "gen_dataset.parquet")
    train = df[df["split"] == "train"].sample(min(args.n_train, (df["split"] == "train").sum()), random_state=0)
    test = df[df["split"] == "test"].sample(args.n_test, random_state=0)
    golds = test["action"].tolist()

    best_const = train["action"].value_counts().idxmax()
    print(f"most common fix in training data: {best_const!r}")
    score("constant (most common fix)", [best_const] * len(test), golds)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vec.fit_transform(train["problem"])
    x_test = vec.transform(test["problem"])
    preds = []
    for start in range(0, x_test.shape[0], 500):   # chunks keep memory small
        sims = linear_kernel(x_test[start:start + 500], x_train)
        preds += train["action"].to_numpy()[sims.argmax(axis=1)].tolist()
    score("nearest similar problem", preds, golds)


if __name__ == "__main__":
    main()