"""TF-IDF + linear classifier baseline.

A strong, fast, fully interpretable baseline: character/word n-gram TF-IDF
features feeding either multinomial Logistic Regression or a Linear SVM. This is
the reference point every deep model must beat (report Section V).

The model owns its own preprocessing (the linear preset with negation marking),
so callers pass *raw* text at both train and inference time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .. import LABEL2ID, LABELS
from ..preprocessing import PreprocessingPipeline
from .base import SentimentModel


class TfidfLinearModel(SentimentModel):
    """TF-IDF vectorizer + (LogisticRegression | calibrated LinearSVC)."""

    def __init__(
        self,
        classifier: str = "logistic_regression",
        ngram_range: tuple = (1, 2),
        max_features: int = 50000,
        min_df: int = 2,
        sublinear_tf: bool = True,
        C: float = 4.0,
        max_iter: int = 2000,
        class_weight: str | None = "balanced",
        calibrate_probabilities: bool = True,
    ) -> None:
        self.classifier_name = classifier
        self.calibrate_probabilities = calibrate_probabilities
        self._pre = PreprocessingPipeline.for_linear()

        vectorizer = TfidfVectorizer(
            ngram_range=tuple(ngram_range),
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b\w+\b|_NEG",  # keep our _NEG tokens
        )

        if classifier == "logistic_regression":
            clf = LogisticRegression(
                C=C, max_iter=max_iter, class_weight=class_weight, n_jobs=None
            )
        elif classifier == "linear_svm":
            base = LinearSVC(C=C, class_weight=class_weight)
            # LinearSVC has no predict_proba; calibrate to expose probabilities
            # (needed for ROC-AUC and the dashboard confidence bars).
            clf = CalibratedClassifierCV(base, cv=3) if calibrate_probabilities else base
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"Unknown classifier: {classifier}")

        self.pipeline: Pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        self.labels = list(LABELS)

    # -- training -----------------------------------------------------------
    def _prep(self, texts: Sequence[str]) -> List[str]:
        return self._pre.transform_batch(list(texts))

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "TfidfLinearModel":
        X = self._prep(train_df["text"].tolist())
        y = train_df["label"].map(LABEL2ID).to_numpy()
        self.pipeline.fit(X, y)
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        X = self._prep(texts)
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(X)
        # Uncalibrated LinearSVC: softmax over decision margins as a fallback.
        margins = self.pipeline.decision_function(X)
        margins = np.atleast_2d(margins)
        exp = np.exp(margins - margins.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def top_features(self, k: int = 15) -> dict:
        """Return the most positive/negative n-grams per class (interpretability).

        Only meaningful for the LogisticRegression variant with linear coefs.
        """
        clf = self.pipeline.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {}
        vocab = np.array(self.pipeline.named_steps["tfidf"].get_feature_names_out())
        out: dict = {}
        for cls_id, label in enumerate(self.labels):
            coefs = clf.coef_[cls_id]
            top = np.argsort(coefs)[-k:][::-1]
            out[label] = list(zip(vocab[top].tolist(), coefs[top].round(3).tolist()))
        return out

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        dump(self.pipeline, directory / "pipeline.joblib")
        meta = {
            "model_type": "tfidf_linear",
            "classifier": self.classifier_name,
            "labels": self.labels,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> "TfidfLinearModel":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        model = cls(classifier=meta["classifier"])
        model.pipeline = load(directory / "pipeline.joblib")
        model.labels = meta["labels"]
        return model
