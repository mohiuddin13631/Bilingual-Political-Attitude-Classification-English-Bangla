"""Abstract base class defining the common model interface.

Every model — linear baseline, BiLSTM, transformer — exposes the same five
methods. Downstream code (trainer, evaluator, SHAP/LIME wrappers, API) depends
only on this contract, never on a concrete class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd

from .. import LABELS


class SentimentModel(ABC):
    """Common interface for all sentiment classifiers.

    Concrete subclasses must be able to train from a DataFrame of records
    (columns include ``text`` and ``label``) and to produce both hard labels
    and calibrated class probabilities for arbitrary raw text.
    """

    #: Ordered class labels; index == class id.
    labels: List[str] = list(LABELS)

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "SentimentModel":
        """Train the model in place and return ``self``."""

    @abstractmethod
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n_samples, n_classes)`` probability matrix."""

    def predict(self, texts: Sequence[str]) -> List[str]:
        """Return the most likely label string for each input text."""
        proba = self.predict_proba(texts)
        idx = np.asarray(proba).argmax(axis=1)
        return [self.labels[i] for i in idx]

    @abstractmethod
    def save(self, directory: str | Path) -> None:
        """Persist all artifacts needed to reload the model."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "SentimentModel":
        """Reload a model previously written by :meth:`save`."""
