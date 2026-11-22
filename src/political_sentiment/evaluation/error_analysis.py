"""Qualitative error analysis utilities.

Surfaces the most confident mistakes and slices error rate by language, so the
report's error-analysis section (Section V.D) is generated from data rather than
hand-picked.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def collect_errors(
    frame: pd.DataFrame,
    y_pred: Sequence[str],
    y_proba: np.ndarray,
    labels: Sequence[str],
) -> pd.DataFrame:
    """Return a DataFrame of misclassified examples ordered by confidence.

    The returned frame carries the text, gold/predicted labels, the model's
    confidence in its (wrong) prediction, and any available metadata
    (language/party/topic) for slicing.
    """
    df = frame.reset_index(drop=True).copy()
    df["predicted"] = list(y_pred)
    df["confidence"] = y_proba.max(axis=1)
    errors = df[df["label"] != df["predicted"]].copy()
    keep = [c for c in ("text", "label", "predicted", "confidence", "language", "party", "topic") if c in errors.columns]
    return errors[keep].sort_values("confidence", ascending=False).reset_index(drop=True)


def error_rate_by(frame: pd.DataFrame, y_pred: Sequence[str], by: str) -> pd.DataFrame:
    """Compute error rate grouped by a metadata column (e.g. ``language``)."""
    df = frame.reset_index(drop=True).copy()
    df["predicted"] = list(y_pred)
    df["is_error"] = (df["label"] != df["predicted"]).astype(int)
    grouped = df.groupby(by)["is_error"].agg(["mean", "count"]).rename(
        columns={"mean": "error_rate", "count": "n"}
    )
    return grouped.sort_values("error_rate", ascending=False)
