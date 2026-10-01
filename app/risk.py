"""Risk model: turn a request into the SAME features the model was trained on, then predict.

The feature logic mirrors build_risk_dataset.py. Rule: only reports strictly BEFORE `as_of`
are used (windows are [as_of - W days, as_of)). If this file and the training script drift
apart, predictions silently become wrong - the unit tests guard against that.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .schemas import RiskRequest

MIN_HISTORY = 10   # the model was trained only on aircraft with >= 10 reports


def build_features(req: RiskRequest) -> dict[str, float | str | None]:
    as_of = req.as_of

    def count(rows, days: int) -> int:
        lo = as_of - timedelta(days=days)
        return sum(1 for r in rows if lo <= r.date < as_of)

    same = [r for r in req.history if r.ata == req.ata]
    prev_same = [r.date for r in same if r.date < as_of]
    return {
        "ata": req.ata,
        "AircraftMake": req.make,
        "AircraftModel": req.model,
        "hours": req.hours,
        "cycles": req.cycles,
        "n_all_last30": count(req.history, 30),
        "n_all_last90": count(req.history, 90),
        "n_all_last365": count(req.history, 365),
        "n_ata_last30": count(same, 30),
        "n_ata_last90": count(same, 90),
        "n_ata_last365": count(same, 365),
        "n_all_before": sum(1 for r in req.history if r.date < as_of),
        "days_since_prev_same_ata": (as_of - max(prev_same)).days if prev_same else None,
        "month": as_of.month,
    }


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def predict(bundle: dict, req: RiskRequest) -> tuple[float, dict, list[str]]:
    feats = build_features(req)
    warnings: list[str] = []
    if len(req.history) < MIN_HISTORY:
        warnings.append(f"Only {len(req.history)} past reports given; the model was trained on aircraft "
                        f"with at least {MIN_HISTORY}, so treat this score as unreliable.")

    row = {}
    for name in bundle["features"]:
        value = feats[name]
        if name in bundle["categories"]:
            cats = bundle["categories"][name]
            if value not in cats:
                warnings.append(f"{name}='{value}' was not seen in training; treated as unknown.")
                value = None
            row[name] = pd.Categorical([value], categories=cats)   # unseen -> missing
        else:
            row[name] = [np.nan if value is None else float(value)]
    prob = float(bundle["model"].predict_proba(pd.DataFrame(row))[:, 1][0])

    shown = {k: (None if v is None else float(v)) for k, v in feats.items()
             if k not in bundle["categories"]}
    return prob, shown, warnings