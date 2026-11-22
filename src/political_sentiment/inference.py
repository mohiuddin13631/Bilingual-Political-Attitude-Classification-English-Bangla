"""Shared inference helper: locate and load a trained model by type.

Used by both the FastAPI service and the Streamlit dashboard so model discovery
and loading logic lives in exactly one place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .models.base import SentimentModel


def load_model_from_dir(model_dir: str | Path) -> SentimentModel:
    """Load a saved model, dispatching on the ``model_type`` in its meta.json."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model_type = meta["model_type"]

    if model_type == "tfidf_linear":
        from .models.baseline import TfidfLinearModel

        return TfidfLinearModel.load(model_dir)
    if model_type == "bilstm_attention":
        from .models.bilstm import BiLSTMAttentionModel

        return BiLSTMAttentionModel.load(model_dir)
    if model_type == "transformer":
        from .models.transformer import TransformerModel

        return TransformerModel.load(model_dir)
    raise ValueError(f"Unknown model_type in {model_dir}: {model_type}")


def discover_default_model(artifacts_dir: str | Path = "artifacts") -> Optional[Path]:
    """Return the first available trained model directory, preferring the
    transformer, then BiLSTM, then baseline."""
    artifacts_dir = Path(artifacts_dir)
    for exp in ("transformer_bert", "bilstm_attention", "baseline_tfidf"):
        candidate = artifacts_dir / exp / "model"
        if (candidate / "meta.json").exists():
            return candidate
    return None


class SentimentService:
    """Thin, framework-agnostic wrapper exposing a single :meth:`predict`."""

    def __init__(self, model: SentimentModel) -> None:
        self.model = model
        self.labels: List[str] = list(model.labels)

    @classmethod
    def from_artifacts(cls, artifacts_dir: str | Path = "artifacts") -> "SentimentService":
        model_dir = discover_default_model(artifacts_dir)
        if model_dir is None:
            raise FileNotFoundError(
                f"No trained model found under {artifacts_dir}. Run `make baseline` first."
            )
        return cls(load_model_from_dir(model_dir))

    def predict(self, texts: List[str]) -> List[Dict[str, object]]:
        """Return per-text predictions with full probability distributions."""
        proba = np.asarray(self.model.predict_proba(texts))
        results: List[Dict[str, object]] = []
        for row, text in zip(proba, texts):
            idx = int(row.argmax())
            results.append(
                {
                    "text": text,
                    "label": self.labels[idx],
                    "confidence": round(float(row[idx]), 4),
                    "probabilities": {
                        lbl: round(float(p), 4) for lbl, p in zip(self.labels, row)
                    },
                }
            )
        return results
