"""LIME and SHAP explainers that work with *any* :class:`SentimentModel`.

Because every model exposes ``predict_proba(list[str]) -> np.ndarray``, the same
explainers apply uniformly to the TF-IDF baseline, the BiLSTM and the
transformer — important for the EU AI Act interpretability discussion (report
Section VIII), which calls for model-agnostic, post-hoc explanations of
individual high-stakes decisions.

Both libraries are optional dependencies; importing this module never fails, and
each explainer raises a clear, actionable error only if used without its backend
installed.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .. import LABELS
from ..models.base import SentimentModel


def _predict_fn(model: SentimentModel) -> Callable[[Sequence[str]], np.ndarray]:
    """Adapt a model to the ``list[str] -> proba matrix`` callable both libs expect."""

    def fn(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(model.predict_proba(list(texts)))

    return fn


class LimeExplainer:
    """Local explanations via LIME (Ribeiro et al., 2016).

    LIME perturbs the input by removing words and fits a sparse linear surrogate,
    yielding per-word contributions toward each class.
    """

    def __init__(self, model: SentimentModel, class_names: Optional[List[str]] = None) -> None:
        try:
            from lime.lime_text import LimeTextExplainer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LIME is required: `pip install lime`.") from exc
        self.model = model
        self.class_names = class_names or list(model.labels or LABELS)
        self._explainer = LimeTextExplainer(class_names=self.class_names)
        self._predict = _predict_fn(model)

    def explain(self, text: str, num_features: int = 10, num_samples: int = 1000) -> Dict[str, list]:
        """Return per-class word contributions for a single text."""
        pred_idx = int(self._predict([text]).argmax(axis=1)[0])
        exp = self._explainer.explain_instance(
            text,
            self._predict,
            num_features=num_features,
            num_samples=num_samples,
            labels=(pred_idx,),
        )
        return {
            "predicted_class": self.class_names[pred_idx],
            "contributions": exp.as_list(label=pred_idx),
        }


class ShapExplainer:
    """Global/local explanations via SHAP over the model's probability output.

    Uses SHAP's text ``Explainer`` with a partition masker, which is the modern
    recommended path for NLP models and works through the ``predict_proba``
    callable without needing model internals.
    """

    def __init__(self, model: SentimentModel) -> None:
        try:
            import shap
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SHAP is required: `pip install shap`.") from exc
        self._shap = shap
        self.model = model
        self._predict = _predict_fn(model)
        # A text masker tokenizes on whitespace and masks tokens with a neutral
        # placeholder; suitable for a model-agnostic pipeline.
        self._masker = shap.maskers.Text(r"\W+")
        self._explainer = shap.Explainer(
            self._predict, self._masker, output_names=list(model.labels or LABELS)
        )

    def explain(self, texts: Sequence[str]):
        """Return a SHAP ``Explanation`` object for the given texts."""
        return self._explainer(list(texts))


def explain_text(
    model: SentimentModel, text: str, method: str = "lime", **kwargs: object
) -> Dict[str, list]:
    """Convenience entry point returning a JSON-serialisable explanation.

    For the BiLSTM, if ``method == "attention"`` and the model exposes attention,
    token attention weights are returned directly (a faithful, built-in
    explanation with no surrogate model).
    """
    if method == "attention" and hasattr(model, "attention_weights"):
        tokens, weights = model.attention_weights(text)  # type: ignore[attr-defined]
        pred = model.predict([text])[0]
        return {
            "predicted_class": pred,
            "contributions": list(zip(tokens, [round(float(w), 4) for w in weights])),
        }
    if method == "lime":
        return LimeExplainer(model).explain(text, **kwargs)  # type: ignore[arg-type]
    if method == "shap":
        explanation = ShapExplainer(model).explain([text])
        return {"shap_values": explanation}
    raise ValueError(f"Unknown explanation method: {method}")
