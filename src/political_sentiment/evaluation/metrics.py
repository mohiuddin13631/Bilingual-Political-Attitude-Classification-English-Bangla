"""Classification metrics for the three-class sentiment task.

Everything downstream (the evaluate CLI, the report tables, the fairness slices)
computes metrics through :func:`compute_metrics` so definitions never drift.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .. import LABELS


def compute_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    y_proba: np.ndarray | None = None,
    labels: List[str] | None = None,
) -> Dict[str, float]:
    """Return a flat dict of headline metrics.

    Parameters
    ----------
    y_true, y_pred:
        Gold and predicted label strings.
    y_proba:
        Optional ``(n, n_classes)`` probability matrix (column order == labels);
        required for ROC-AUC. When omitted, ROC-AUC is reported as NaN.
    labels:
        Class order. Defaults to the project's canonical ``LABELS``.
    """
    labels = labels or list(LABELS)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
    }

    if y_proba is not None and len(set(y_true)) > 1:
        # One-vs-rest macro ROC-AUC over the probability matrix.
        y_true_idx = np.array([labels.index(y) for y in y_true])
        try:
            metrics["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true_idx, y_proba, multi_class="ovr", average="macro", labels=list(range(len(labels))))
            )
        except ValueError:
            metrics["roc_auc_ovr_macro"] = float("nan")
    else:
        metrics["roc_auc_ovr_macro"] = float("nan")

    return metrics


def per_class_report(
    y_true: Sequence[str], y_pred: Sequence[str], labels: List[str] | None = None
) -> Dict[str, Dict[str, float]]:
    """Return sklearn's per-class precision/recall/F1 as a nested dict."""
    labels = labels or list(LABELS)
    return classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )


def confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: List[str] | None = None) -> np.ndarray:
    """Return the confusion matrix with rows=true, cols=pred, fixed label order."""
    labels = labels or list(LABELS)
    return confusion_matrix(y_true, y_pred, labels=labels)
