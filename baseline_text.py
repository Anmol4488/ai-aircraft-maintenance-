"""
baseline_text.py - the bar SmolLM2 must beat for Task 1 (text -> system).

TF-IDF turns each report into word-count numbers; logistic regression learns which
words point to which system. No neural network, runs in about a minute on a laptop CPU.

Usage:
    python baseline_text.py --in-dir processed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    args = ap.parse_args()

    df = pd.read_parquet(args.in_dir / "text_dataset.parquet")
    train, test = df[df["split"] == "train"], df[df["split"] == "test"]

    majority = train["label"].value_counts().idxmax()
    print(f"always predict most common system ({majority}): acc={(test['label'] == majority).mean():.3f}")

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    x_train = vec.fit_transform(train["text"])
    x_test = vec.transform(test["text"])

    clf = LogisticRegression(max_iter=300, C=5.0)
    clf.fit(x_train, train["label"])
    pred = clf.predict(x_test)

    print(f"TF-IDF + logistic regression: acc={accuracy_score(test['label'], pred):.3f}  "
          f"macro-F1={f1_score(test['label'], pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()