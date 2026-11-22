"""Pydantic request/response schemas for the inference API."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """A batch of raw texts to classify."""

    texts: List[str] = Field(..., min_length=1, description="Texts to classify.")

    model_config = {
        "json_schema_extra": {
            "example": {"texts": ["PartyA's economy plan is a real step forward.",
                                    "এই নীতি মোটেও ভালো নয়।"]}
        }
    }


class Prediction(BaseModel):
    """A single classification result."""

    text: str
    label: str
    confidence: float
    probabilities: Dict[str, float]


class PredictResponse(BaseModel):
    """The full batch response."""

    model_name: str
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    labels: List[str]
