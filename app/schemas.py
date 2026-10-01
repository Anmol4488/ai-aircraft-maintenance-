"""Request/response shapes. Pydantic checks every incoming request BEFORE our code runs,
so bad input becomes a clear 422 error instead of a crash deep inside a model."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator

DISCLAIMER = (
    "Decision-support only. Outputs come from statistical models trained on historical reports "
    "and are NOT maintenance instructions or airworthiness decisions. A qualified, certified "
    "person must verify every suggestion against approved maintenance data."
)

ATA_PATTERN = r"^\d{2}$"


class PastReport(BaseModel):
    date: date
    ata: str = Field(pattern=ATA_PATTERN, description="2-digit system chapter, e.g. '53'")


class RiskRequest(BaseModel):
    make: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=40)
    ata: str = Field(pattern=ATA_PATTERN, description="system you want the risk for")
    as_of: date = Field(description="date of the current report")
    hours: float | None = Field(default=None, ge=0, le=200_000)
    cycles: float | None = Field(default=None, ge=0, le=150_000)
    history: list[PastReport] = Field(default_factory=list, max_length=5000,
                                      description="this aircraft's earlier reports")

    @field_validator("make", "model")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().upper()          # training data is upper-case

    @model_validator(mode="after")
    def history_not_in_future(self) -> "RiskRequest":
        if any(r.date > self.as_of for r in self.history):
            raise ValueError("history contains reports dated after as_of")
        return self


class RiskResponse(BaseModel):
    ata: str
    horizon_days: int
    probability_repeat_defect: float
    features_used: dict[str, float | None]
    warnings: list[str]
    disclaimer: str = DISCLAIMER


class TextRequest(BaseModel):
    text: str = Field(min_length=10, max_length=2000)

    @field_validator("text")
    @classmethod
    def strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if len(v) < 10:
            raise ValueError("text too short")
        return v


class ClassScore(BaseModel):
    ata: str
    probability: float


class ClassifyResponse(BaseModel):
    top: list[ClassScore]
    disclaimer: str = DISCLAIMER


class FixResponse(BaseModel):
    suggested_fix: str | None
    warnings: list[str]
    disclaimer: str = DISCLAIMER