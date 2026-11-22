"""FastAPI application exposing real-time political sentiment inference.

Run with::

    uvicorn political_sentiment.api.app:app --reload --port 8000

Then POST to ``/predict``::

    curl -X POST localhost:8000/predict \\
         -H 'Content-Type: application/json' \\
         -d '{"texts": ["PartyA economy plan is a step forward", "এই নীতি ভালো নয়"]}'

The model is discovered from ``artifacts/`` at startup (transformer > BiLSTM >
baseline). Set the ``PS_ARTIFACTS_DIR`` env var to point elsewhere.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ..inference import SentimentService
from ..utils.logging_utils import get_logger
from .schemas import HealthResponse, PredictRequest, PredictResponse

logger = get_logger("api")

# Application state populated at startup.
_state: dict = {"service": None, "model_name": "none"}


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Load the model once at startup so requests are cheap."""
    artifacts_dir = os.environ.get("PS_ARTIFACTS_DIR", "artifacts")
    try:
        service = SentimentService.from_artifacts(artifacts_dir)
        _state["service"] = service
        _state["model_name"] = type(service.model).__name__
        logger.info("Loaded model: %s", _state["model_name"])
    except FileNotFoundError as exc:
        logger.warning("No model available at startup: %s", exc)
    yield
    _state["service"] = None


app = FastAPI(
    title="Bilingual Political Attitude Classification (English & Bangla)",
    description=(
        "Classify political text in English and Bangla as negative, neutral, or positive. "
        "Political attitude is represented by text sentiment."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness probe."""
    service = _state["service"]
    return HealthResponse(
        status="ok",
        model_loaded=service is not None,
        model_name=_state["model_name"],
        labels=service.labels if service else [],
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a batch of texts and return per-class probabilities."""
    service = _state["service"]
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train one first (e.g. `make baseline`).",
        )
    predictions = service.predict(request.texts)
    return PredictResponse(model_name=_state["model_name"], predictions=predictions)


@app.get("/")
def root() -> dict:
    """Minimal landing payload pointing at the docs."""
    return {"service": "bilingual-political-attitude-classification", "docs": "/docs", "health": "/health"}
