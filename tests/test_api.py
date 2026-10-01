"""
Run from the project root:   python -m pytest -q

These tests need neither a GPU nor the SmolLM2 models: the language models are switched off
(LOAD_LLMS=0) and a tiny throw-away risk model is trained just for the test.
"""

from __future__ import annotations

from datetime import date

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.llm import strip_verification_claims
from app.main import app
from app.risk import build_features
from app.schemas import PastReport, RiskRequest

FEATURES = ["ata", "AircraftMake", "AircraftModel", "hours", "cycles", "n_all_last30", "n_all_last90",
            "n_all_last365", "n_ata_last30", "n_ata_last90", "n_ata_last365", "n_all_before",
            "days_since_prev_same_ata", "month"]
CATEGORIES = {"ata": ["25", "53"], "AircraftMake": ["BOEING"], "AircraftModel": ["737"]}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    n = 300
    x = pd.DataFrame({f: rng.integers(0, 50, n).astype(float) for f in FEATURES if f not in CATEGORIES})
    for c, cats in CATEGORIES.items():
        x[c] = pd.Categorical(rng.choice(cats, n), categories=cats)
    y = (x["n_ata_last90"] > 20).astype(int)
    model = lgb.LGBMClassifier(n_estimators=5, min_child_samples=5, verbose=-1).fit(x[FEATURES], y)
    path = tmp_path / "risk.joblib"
    joblib.dump({"model": model, "features": FEATURES, "categories": CATEGORIES, "horizon_days": 90}, path)

    monkeypatch.setenv("RISK_MODEL_PATH", str(path))
    monkeypatch.setenv("LOAD_LLMS", "0")
    with TestClient(app) as c:
        yield c


def req(**over):
    base = {"make": " boeing ", "model": "737", "ata": "53", "as_of": "2025-06-01", "hours": 30000,
            "cycles": 15000, "history": [{"date": "2025-05-20", "ata": "53"}, {"date": "2025-03-01", "ata": "25"}]}
    base.update(over)
    return base


# ---------- feature logic (must match the training script) ----------
def test_window_counts_are_strictly_before_as_of():
    r = RiskRequest(make="b", model="7", ata="53", as_of=date(2025, 6, 1), history=[
        PastReport(date=date(2025, 6, 1), ata="53"),    # same day -> NOT counted
        PastReport(date=date(2025, 5, 2), ata="53"),    # 30 days before -> counted (window is [d-30, d))
        PastReport(date=date(2025, 5, 1), ata="53"),    # 31 days before -> outside 30-day window
        PastReport(date=date(2025, 5, 10), ata="25"),   # other system
    ])
    f = build_features(r)
    assert f["n_all_last30"] == 2 and f["n_ata_last30"] == 1
    assert f["n_all_before"] == 3
    assert f["days_since_prev_same_ata"] == 30


def test_no_previous_same_system_gives_none():
    r = RiskRequest(make="b", model="7", ata="53", as_of=date(2025, 6, 1),
                    history=[PastReport(date=date(2025, 5, 10), ata="25")])
    assert build_features(r)["days_since_prev_same_ata"] is None


# ---------- API ----------
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["risk_model"] is True and body["classifier"] is False


def test_risk_ok_and_normalises_names(client):
    r = client.post("/risk", json=req())
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_repeat_defect"] <= 1.0
    assert body["horizon_days"] == 90 and "disclaimer" in body
    assert any("unreliable" in w for w in body["warnings"])        # only 2 past reports


def test_risk_unknown_model_is_a_warning_not_a_crash(client):
    r = client.post("/risk", json=req(model="NEWPLANE"))
    assert r.status_code == 200
    assert any("not seen in training" in w for w in r.json()["warnings"])


@pytest.mark.parametrize("bad", [
    {"ata": "5"}, {"ata": "ABC"}, {"hours": -5}, {"cycles": 999999}, {"as_of": "not-a-date"},
    {"history": [{"date": "2026-01-01", "ata": "53"}]},            # report from the future
])
def test_risk_rejects_bad_input(client, bad):
    assert client.post("/risk", json=req(**bad)).status_code == 422


def test_llm_endpoints_return_503_when_not_loaded(client):
    text = {"text": "Seat track found corroded at frame 74."}
    assert client.post("/classify", json=text).status_code == 503
    assert client.post("/suggest-fix", json=text).status_code == 503


def test_text_validation(client):
    assert client.post("/classify", json={"text": "short"}).status_code == 422
    assert client.post("/suggest-fix", json={"text": "x" * 3000}).status_code == 422


# ---------- safety filter ----------
def test_verification_claims_are_removed():
    cleaned, removed = strip_verification_claims("REPLACED THE LAMP. OPS CHECKED GOOD.")
    assert cleaned == "REPLACED THE LAMP." and removed


def test_clean_text_is_untouched():
    cleaned, removed = strip_verification_claims("REPLACED HINGE FITTING.")
    assert cleaned == "REPLACED HINGE FITTING." and not removed


def test_only_claims_gives_none():
    cleaned, removed = strip_verification_claims("OPS CHECK GOOD. RETURNED TO SERVICE.")
    assert cleaned is None and removed