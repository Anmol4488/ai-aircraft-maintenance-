"""
build_risk_dataset.py - Step 2: turn the cleaned reports into a labelled dataset.

Question the model will answer:
    "Given an aircraft's past, will it have ANOTHER defect in the SAME system
     (ATA chapter) within the next HORIZON days?"

One example = one (aircraft, system, day) that has a report.
Label = 1 if the same aircraft has another report in the same system within HORIZON days.

Key rules (these prevent fake-good results):
  1. Only aircraft with >= MIN_REPORTS reports are used (you chose 10).
  2. Examples whose full HORIZON-day window is not yet observed are DROPPED
     (data ends 2026-09-25, so a report on 2026-08-20 has an unknown answer).
  3. Features use ONLY information from BEFORE the example's date.
  4. Train / validation / test are split by DATE, never randomly.

Usage:
    python build_risk_dataset.py --in-dir processed --horizon 90
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MIN_REPORTS = 10
TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")
DAY_BASE = 100_000  # key = group_code * DAY_BASE + day_number (day_number < 100k)


def count_in_window(keys_sorted: np.ndarray, group: np.ndarray, day: np.ndarray, window: int) -> np.ndarray:
    """
    For every query (group, day): how many reports of that group happened in
    [day - window, day)  -> strictly BEFORE day, so no same-day leakage.
    `keys_sorted` = sorted array of group_code * DAY_BASE + day_number for ALL reports.
    """
    lo = np.searchsorted(keys_sorted, group * DAY_BASE + (day - window), side="left")
    hi = np.searchsorted(keys_sorted, group * DAY_BASE + day, side="left")
    return hi - lo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    ap.add_argument("--horizon", type=int, default=90)
    args = ap.parse_args()

    cols = ["aircraft_id", "date", "ata", "AircraftMake", "AircraftModel", "hours", "cycles"]
    df = pd.read_parquet(args.in_dir / "reports_clean.parquet", columns=cols)

    # Rule 1: enough history
    n = df.groupby("aircraft_id")["date"].transform("size")
    df = df[n >= MIN_REPORTS].copy()
    df = df.sort_values(["aircraft_id", "date"]).reset_index(drop=True)
    data_end = df["date"].max()
    print(f"aircraft kept: {df.aircraft_id.nunique()} | reports: {len(df)} | data ends {data_end.date()}")

    # Numeric codes + day numbers so we can count windows with fast vectorised searchsorted.
    df["ac_code"] = pd.factorize(df["aircraft_id"])[0].astype(np.int64)
    df["ac_ata_code"] = pd.factorize(df["aircraft_id"] + "#" + df["ata"])[0].astype(np.int64)
    df["day"] = (df["date"] - pd.Timestamp("1970-01-01")).dt.days.astype(np.int64)
    # Hours/cycles: fill forward within an aircraft (latest known value from the past).
    df[["hours", "cycles"]] = df.groupby("aircraft_id")[["hours", "cycles"]].ffill()

    keys_ac = np.sort(df["ac_code"].to_numpy() * DAY_BASE + df["day"].to_numpy())
    keys_ac_ata = np.sort(df["ac_ata_code"].to_numpy() * DAY_BASE + df["day"].to_numpy())

    # One example per (aircraft, system, day). Take the first row of each.
    ex = df.drop_duplicates(["aircraft_id", "ata", "date"]).copy()
    ex = ex.sort_values(["aircraft_id", "ata", "date"]).reset_index(drop=True)

    # ---- LABEL: is there another report in the same system within `horizon` days? ----
    next_date = ex.groupby(["aircraft_id", "ata"])["date"].shift(-1)  # strictly later (dates are unique per group)
    days_to_next = (next_date - ex["date"]).dt.days
    ex["label"] = (days_to_next <= args.horizon).astype(int)

    # Rule 2: drop examples whose full window we cannot observe.
    window_complete = ex["date"] + pd.Timedelta(days=args.horizon) <= data_end
    print(f"dropped (window not fully observed): {int((~window_complete).sum())}")
    ex = ex[window_complete].copy()

    # ---- FEATURES (past only) ----
    g_ac = ex["ac_code"].to_numpy()
    g_ata = ex["ac_ata_code"].to_numpy()
    d = ex["day"].to_numpy()
    for w in (30, 90, 365):
        ex[f"n_all_last{w}"] = count_in_window(keys_ac, g_ac, d, w)
        ex[f"n_ata_last{w}"] = count_in_window(keys_ac_ata, g_ata, d, w)
    # All reports of this aircraft before this day (window = d reaches back to day 0 of the key space).
    ex["n_all_before"] = count_in_window(keys_ac, g_ac, d, d)
    ex["days_since_prev_same_ata"] = ex.groupby(["aircraft_id", "ata"])["date"].diff().dt.days
    ex["month"] = ex["date"].dt.month

    # Rule 4: split by date
    ex["split"] = np.where(ex["date"] <= TRAIN_END, "train", np.where(ex["date"] <= VAL_END, "val", "test"))

    out_cols = ["aircraft_id", "date", "ata", "AircraftMake", "AircraftModel", "hours", "cycles",
                "n_all_last30", "n_all_last90", "n_all_last365",
                "n_ata_last30", "n_ata_last90", "n_ata_last365",
                "n_all_before", "days_since_prev_same_ata", "month", "split", "label"]
    out = ex[out_cols]
    out.to_parquet(args.in_dir / "risk_dataset.parquet", index=False)

    print("\nexamples per split and share of label=1 (defect within horizon):")
    print(out.groupby("split")["label"].agg(["size", "mean"]).round(3))
    print(f"\nsaved {args.in_dir / 'risk_dataset.parquet'}")


if __name__ == "__main__":
    main()