"""
build_gen_dataset.py - Step 6: split each report into PROBLEM and FIX (action).

Rule (simple and inspectable):
  1. Cut the report into sentences.
  2. The first sentence that STARTS with an action word (REPLACED, REMOVED, REPAIRED ...)
     begins the fix. Everything before it is the problem.
  3. Keep the report only if both parts are long enough.

Safety rule: manual references ("IAW AMM 25-55-00", "PER SRM 51-42-11") are removed from the
target text. A language model cannot know the right manual page, so it would INVENT one -
a made-up maintenance reference is worse than none. The raw action is kept in a separate column.

Same date-based split as before; val/test problems that also appear in train are removed.

Usage:
    python build_gen_dataset.py --in-dir processed
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TRAIN_END = pd.Timestamp("2023-12-31")
VAL_END = pd.Timestamp("2024-12-31")

ACTION_VERBS = (
    "REMOVED|REPLACED|REPAIRED|INSTALLED|REINSTALLED|REWORKED|CLEANED|ADJUSTED|TIGHTENED|BLENDED|"
    "FABRICATED|SERVICED|LUBRICATED|RESEALED|RE-SEALED|SEALED|RETORQUED|RE-TORQUED|TORQUED|DRESSED|"
    "POLISHED|PATCHED|SPLICED|RECONNECTED|RESET|REPROGRAMMED|CALIBRATED|STRAIGHTENED|OVERHAULED|"
    "SWAPPED|RERIGGED|RIGGED|DEBURRED|STOP-DRILLED|DRILLED|CHANGED|CORRECTED|APPLIED|REAPPLIED|"
    "REFILLED|RESTORED|REWIRED|REFURBISHED|REROUTED|RE-ROUTED|REFASTENED|SECURED|ACCOMPLISHED"
)
# a sentence that begins with (optional filler) + action verb
ACTION_START = re.compile(rf"^(?:(?:AND|THEN|ALSO|MX|MAINTENANCE|TECH(?:NICIAN)?)\s+)?(?:{ACTION_VERBS})\b")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
# "IAW AMM 25-55-00", "PER SRM 51-42-11 AND DRAWING 14N00021" ... up to end of sentence
REF_CLAUSE = re.compile(r"\s*\b(?:IAW|PER|IAW WITH|ACCORDING TO|REFERENCE|REF\.?)\b[^.]*", flags=re.I)


def split_report(text: str) -> tuple[str, str] | None:
    sents = [s for s in SENT_SPLIT.split(text.strip()) if s]
    for i, s in enumerate(sents):
        if ACTION_START.match(s.upper().lstrip(" ,;:-")):
            if i == 0:
                return None            # nothing before the fix -> no problem statement
            return " ".join(sents[:i]), " ".join(sents[i:])
    return None


def strip_refs(action: str) -> str:
    out = REF_CLAUSE.sub("", action)
    return re.sub(r"\s+", " ", out).strip(" .,;") + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=Path("processed"))
    args = ap.parse_args()

    df = pd.read_parquet(args.in_dir / "reports_clean.parquet", columns=["date", "ata", "Discrepancy"])
    n_all = len(df)

    parts = df["Discrepancy"].map(split_report)
    ok = parts.notna()
    print(f"reports split successfully: {int(ok.sum())} of {n_all} ({ok.mean():.1%})")
    df = df[ok].copy()
    df["problem"] = [p[0] for p in parts[ok]]
    df["action_raw"] = [p[1] for p in parts[ok]]
    df["action"] = df["action_raw"].map(strip_refs)

    good = (df["problem"].str.len() >= 15) & (df["action"].str.len() >= 15) & (df["problem"].str.len() <= 600)
    print(f"after length filter: {int(good.sum())}")
    df = df[good]

    df = df.drop_duplicates(subset=["problem", "action"])
    df["split"] = "train"
    df.loc[df["date"] > TRAIN_END, "split"] = "val"
    df.loc[df["date"] > VAL_END, "split"] = "test"

    seen = set(df.loc[df["split"] == "train", "problem"])
    leak = (df["split"] != "train") & df["problem"].isin(seen)
    print(f"removed val/test rows whose problem text is in train: {int(leak.sum())}")
    df = df[~leak]

    out = df[["date", "ata", "problem", "action", "action_raw", "split"]].reset_index(drop=True)
    out.to_parquet(args.in_dir / "gen_dataset.parquet", index=False)

    print("\nrows per split:")
    print(out["split"].value_counts().to_string())
    print(f"\nmedian chars: problem={out.problem.str.len().median():.0f}  action={out.action.str.len().median():.0f}")
    print(f"actions that had a manual reference removed: {out.action_raw.map(lambda a: bool(REF_CLAUSE.search(a))).mean():.1%}")
    print("\n--- 6 random examples to check by eye ---")
    for _, r in out.sample(6, random_state=7).iterrows():
        print(f"PROBLEM: {r.problem}\nFIX:     {r.action}\n")


if __name__ == "__main__":
    main()