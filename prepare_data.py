"""
prepare_data.py - Step 1 of the aircraft maintenance project.

Reads the FAA SDR (Service Difficulty Report) CSVs, cleans them, and builds:
  1. reports_clean.parquet  - one row per cleaned report, sorted per aircraft by date,
                              with per-aircraft "history" columns (gap since last report, etc.)
  2. aircraft.parquet       - one row per aircraft (summary table)
  3. cleaning_report.txt    - how many rows each cleaning rule removed / flagged

Usage (from your project root, on Windows or Linux):
    pip install pandas pyarrow
    python prepare_data.py --data-dir data/faa --out-dir processed

NOTE: use only the `faa/` folder. The `nasa/` folder is an exact duplicate for 2015-2025.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("prepare_data")

# Only the columns we actually need. Loading all 77 columns wastes RAM.
USE_COLS = [
    "OperatorControlNumber", "DifficultyDate", "OperatorDesignator", "SDRType",
    "JASCCode", "NatureOfConditionA", "StageOfOperationCode", "HowDiscoveredCode",
    "RegistryNNumber", "AircraftMake", "AircraftModel", "AircraftSerialNumber",
    "AircraftTotalTime", "AircraftTotalCycles", "PartName", "PartCondition",
    "Discrepancy",
]

# Physical plausibility limits (airframe hours / cycles). Values above these are
# almost certainly typos (we saw hours up to ~10 million in the raw data).
MAX_HOURS = 200_000
MAX_CYCLES = 150_000
# Real aircraft fly at most ~24 h/day. Allow slack for rounding / late reporting.
MAX_HOURS_PER_DAY = 24.0

BAD_SERIALS = {"", "UNKNOWN", "UNK", "NONE", "N/A", "NA", "0", "00", "000", "NULL", "-"}


def load_raw(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate all yearly CSVs (as strings first - safest for messy data)."""
    files = sorted(data_dir.glob("SDR-*.csv"))
    if not files:
        raise FileNotFoundError(f"No SDR-*.csv files found in {data_dir}")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f, usecols=USE_COLS, dtype=str, low_memory=False))
        log.info("loaded %s (%d rows)", f.name, len(frames[-1]))
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Row-level cleaning. Each step records how many rows it affected."""
    n0 = len(df)
    report.append(f"raw rows: {n0}")

    # --- Normalise text columns (strip, uppercase codes) ---
    for col in ["AircraftMake", "AircraftModel", "AircraftSerialNumber", "OperatorDesignator",
                "JASCCode", "NatureOfConditionA", "PartName", "RegistryNNumber"]:
        df[col] = df[col].str.strip().str.upper()

    # Collapse runs of whitespace/newlines inside the free text.
    df["Discrepancy"] = df["Discrepancy"].str.replace(r"\s+", " ", regex=True).str.strip()

    # --- Dates ---
    df["date"] = pd.to_datetime(df["DifficultyDate"], format="%m/%d/%Y", errors="coerce")
    bad_date = df["date"].isna()
    report.append(f"dropped (unparseable date): {int(bad_date.sum())}")
    df = df[~bad_date]

    # --- Aircraft key: make | model | serial. Serial number alone is NOT unique across makes. ---
    df = df[~df["AircraftSerialNumber"].isin(BAD_SERIALS) & df["AircraftSerialNumber"].notna()]
    df = df.dropna(subset=["AircraftMake", "AircraftModel"])
    df["aircraft_id"] = df["AircraftMake"] + "|" + df["AircraftModel"] + "|" + df["AircraftSerialNumber"]
    report.append(f"after dropping rows with no usable aircraft identity: {len(df)}")

    # --- Text: drop empty / placeholder descriptions (useless for the SLM) ---
    text_bad = df["Discrepancy"].isna() | (df["Discrepancy"].str.len() < 15) | (df["Discrepancy"] == "NONE")
    report.append(f"dropped (empty/too-short/placeholder text): {int(text_bad.sum())}")
    df = df[~text_bad]

    # --- Exact duplicates: same aircraft, day, system and text ---
    before = len(df)
    df = df.drop_duplicates(subset=["aircraft_id", "date", "JASCCode", "Discrepancy"])
    report.append(f"dropped (exact duplicate reports): {before - len(df)}")

    # --- Numeric usage columns: implausible -> NaN (keep the row, lose only the bad number) ---
    df["hours"] = pd.to_numeric(df["AircraftTotalTime"], errors="coerce")
    df["cycles"] = pd.to_numeric(df["AircraftTotalCycles"], errors="coerce")
    bad_h = (df["hours"] <= 0) | (df["hours"] > MAX_HOURS)
    bad_c = (df["cycles"] <= 0) | (df["cycles"] > MAX_CYCLES)
    report.append(f"hours set to NaN (<=0 or >{MAX_HOURS}): {int(bad_h.sum())}")
    report.append(f"cycles set to NaN (<=0 or >{MAX_CYCLES}): {int(bad_c.sum())}")
    df.loc[bad_h, "hours"] = np.nan
    df.loc[bad_c, "cycles"] = np.nan

    # --- Derived code columns ---
    df["ata"] = df["JASCCode"].str[:2]  # first two digits = system chapter (e.g. 53 = fuselage)
    return df.reset_index(drop=True)


def _flag_inconsistent_usage(g: pd.DataFrame) -> pd.Series:
    """
    For ONE aircraft (sorted by date): flag rows whose hours are inconsistent with the
    aircraft's own timeline. Airframe hours can only go up, and by at most ~24 h per day.

    We compare each value against the neighbours' median (robust to a single typo),
    then check the physical rate limit against the previous good value.
    """
    h = g["hours"]
    d = (g["date"] - g["date"].iloc[0]).dt.days.astype(float)
    ok = h.notna().to_numpy().copy()
    hv = h.to_numpy(dtype=float)
    dv = d.to_numpy()

    last_h = last_d = None
    bad = np.zeros(len(g), dtype=bool)
    # Reference = median of all valid hours-vs-time fit is overkill; use running check
    # against the previous accepted point, and re-anchor if 3 in a row disagree
    # (meaning the *anchor* was the typo, not the new points).
    streak = 0
    for i in range(len(g)):
        if not ok[i]:
            continue
        if last_h is None:
            last_h, last_d = hv[i], dv[i]
            continue
        dt = dv[i] - last_d
        dh = hv[i] - last_h
        consistent = (dh >= -1.0) and (dh <= MAX_HOURS_PER_DAY * max(dt, 0) + 50.0)
        if consistent:
            last_h, last_d = hv[i], dv[i]
            streak = 0
        else:
            bad[i] = True
            streak += 1
            if streak >= 3:  # anchor was probably wrong -> restart from here
                last_h, last_d = hv[i], dv[i]
                bad[i - 2: i + 1] = False
                streak = 0
    return pd.Series(bad, index=g.index)


def add_history(df: pd.DataFrame, report: list[str]) -> pd.DataFrame:
    """Sort per aircraft and add the sequence ("history") columns."""
    df = df.sort_values(["aircraft_id", "date", "OperatorControlNumber"]).reset_index(drop=True)

    # Fix usage inconsistencies per aircraft (only aircraft with >=2 valid hour values matter).
    df["hours_suspect"] = False
    counts = df.groupby("aircraft_id")["hours"].transform("count")
    multi = df[counts >= 2]
    flags = multi.groupby("aircraft_id", group_keys=False).apply(_flag_inconsistent_usage)
    df.loc[flags.index, "hours_suspect"] = flags
    report.append(f"hours set to NaN (inconsistent with aircraft's own timeline): {int(df['hours_suspect'].sum())}")
    df.loc[df["hours_suspect"], "hours"] = np.nan

    g = df.groupby("aircraft_id", sort=False)
    df["event_idx"] = g.cumcount()                                   # 0,1,2,... per aircraft
    df["days_since_prev"] = g["date"].diff().dt.days                 # NaN for the first report
    df["hours_since_prev"] = g["hours"].transform(lambda s: s.ffill().diff())
    df["days_since_prev_same_ata"] = df.groupby(["aircraft_id", "ata"])["date"].diff().dt.days
    df["n_prev_same_ata"] = df.groupby(["aircraft_id", "ata"]).cumcount()
    return df


def build_aircraft_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per aircraft."""
    g = df.groupby("aircraft_id")
    out = pd.DataFrame({
        "make": g["AircraftMake"].first(),
        "model": g["AircraftModel"].first(),
        "serial": g["AircraftSerialNumber"].first(),
        "operator": g["OperatorDesignator"].agg(lambda s: s.mode().iat[0] if s.notna().any() else np.nan),
        "n_reports": g.size(),
        "first_report": g["date"].min(),
        "last_report": g["date"].max(),
        "max_hours": g["hours"].max(),
        "max_cycles": g["cycles"].max(),
        "n_distinct_ata": g["ata"].nunique(),
    })
    out["span_days"] = (out["last_report"] - out["first_report"]).dt.days
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data/faa"))
    ap.add_argument("--out-dir", type=Path, default=Path("processed"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = []
    df = load_raw(args.data_dir)
    df = clean(df, report)
    df = add_history(df, report)
    aircraft = build_aircraft_table(df)

    report.append(f"FINAL reports: {len(df)} | aircraft: {len(aircraft)}")
    report.append(f"aircraft with >=5 reports: {int((aircraft.n_reports >= 5).sum())}")
    report.append(f"date range: {df.date.min().date()} -> {df.date.max().date()}")
    report.append(f"hours missing: {df.hours.isna().mean():.1%} | cycles missing: {df.cycles.isna().mean():.1%}")

    keep = ["aircraft_id", "date", "OperatorControlNumber", "OperatorDesignator", "AircraftMake",
            "AircraftModel", "JASCCode", "ata", "NatureOfConditionA", "StageOfOperationCode",
            "HowDiscoveredCode", "PartName", "PartCondition", "SDRType", "hours", "cycles",
            "hours_suspect", "event_idx", "days_since_prev", "hours_since_prev",
            "days_since_prev_same_ata", "n_prev_same_ata", "Discrepancy"]
    df[keep].to_parquet(args.out_dir / "reports_clean.parquet", index=False)
    aircraft.to_parquet(args.out_dir / "aircraft.parquet", index=False)
    (args.out_dir / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()