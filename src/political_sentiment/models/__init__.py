"""Model zoo: a shared interface plus three concrete classifiers.

All models implement :class:`~.base.SentimentModel` so that training,
evaluation, explainability and serving code is model-agnostic.
"""

from .base import SentimentModel
from .baseline import TfidfLinearModel
from .registry import build_model

__all__ = ["SentimentModel", "TfidfLinearModel", "build_model"]
