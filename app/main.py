"""
Aircraft maintenance decision-support API.

Run from the project root:
    uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000/docs for an interactive page where you can try every endpoint.

Environment variables (all optional):
    RISK_MODEL_PATH   default models/risk_lgbm.joblib
    CLASSIFIER_DIR    default models/smollm2_ata_classifier
    GENERATOR_DIR     default models/smollm2_fix_generator
    LOAD_LLMS         set to 0 to skip loading the two language models (used by the tests)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from . import risk
from .llm import FixGenerator, TextClassifier, strip_verification_claims
from .schemas import (ClassifyResponse, ClassScore, FixResponse, RiskRequest, RiskResponse,
                      TextRequest)

log = logging.getLogger("aircraft_api")
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models ONCE at startup (loading per request would take seconds each time)."""
    app.state.risk = app.state.classifier = app.state.generator = None

    risk_path = Path(os.getenv("RISK_MODEL_PATH", "models/risk_lgbm.joblib"))
    if risk_path.exists():
        app.state.risk = risk.load_bundle(risk_path)
        log.info("risk model loaded from %s", risk_path)
    else:
        log.warning("risk model not found at %s", risk_path)

    if os.getenv("LOAD_LLMS", "1") != "0":
        cls_dir = Path(os.getenv("CLASSIFIER_DIR", "models/smollm2_ata_classifier"))
        gen_dir = Path(os.getenv("GENERATOR_DIR", "models/smollm2_fix_generator"))
        if cls_dir.exists():
            app.state.classifier = TextClassifier(cls_dir)
            log.info("classifier loaded from %s", cls_dir)
        if gen_dir.exists():
            app.state.generator = FixGenerator(gen_dir)
            log.info("fix generator loaded from %s", gen_dir)
    yield


app = FastAPI(
    title="Aircraft Maintenance Decision Support",
    version="0.1.0",
    description="Risk scoring, defect classification and draft-fix suggestions trained on FAA "
                "Service Difficulty Reports. Decision support only - not maintenance instructions.",
    lifespan=lifespan,
)


def _require(component, name: str):
    if component is None:
        raise HTTPException(status_code=503, detail=f"{name} is not loaded on this server")
    return component


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "risk_model": app.state.risk is not None,
        "classifier": app.state.classifier is not None,
        "fix_generator": app.state.generator is not None,
    }


@app.post("/risk", response_model=RiskResponse)
def risk_score(req: RiskRequest) -> RiskResponse:
    bundle = _require(app.state.risk, "risk model")
    prob, feats, warnings = risk.predict(bundle, req)
    return RiskResponse(ata=req.ata, horizon_days=bundle["horizon_days"],
                        probability_repeat_defect=round(prob, 4), features_used=feats, warnings=warnings)


@app.post("/classify", response_model=ClassifyResponse)
def classify(req: TextRequest) -> ClassifyResponse:
    clf = _require(app.state.classifier, "classifier")
    return ClassifyResponse(top=[ClassScore(ata=a, probability=round(p, 4)) for a, p in clf.predict(req.text)])


@app.post("/suggest-fix", response_model=FixResponse)
def suggest_fix(req: TextRequest) -> FixResponse:
    gen = _require(app.state.generator, "fix generator")
    raw = gen.generate(req.text)
    cleaned, removed = strip_verification_claims(raw)
    warnings = ["This is a drafted suggestion learned from past reports, not a diagnosis."]
    if removed:
        warnings.append("Removed statements claiming checks were performed; the model cannot know that.")
    if cleaned is None:
        warnings.append("The model produced no usable suggestion for this input.")
    return FixResponse(suggested_fix=cleaned, warnings=warnings)