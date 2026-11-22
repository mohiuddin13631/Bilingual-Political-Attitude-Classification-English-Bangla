"""Matplotlib plotting helpers (confusion matrix, ROC curves, score bars).

All functions write a PNG to ``out_path`` and return that path, so the evaluate
CLI and notebooks can collect figures for the report without duplicating
styling. Uses a non-interactive backend so it runs headless.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .. import LABELS  # noqa: E402


def plot_confusion_matrix(
    matrix: np.ndarray,
    out_path: str | Path,
    labels: List[str] | None = None,
    title: str = "Confusion matrix",
    normalize: bool = True,
) -> Path:
    """Render a (optionally row-normalized) confusion matrix heatmap."""
    labels = labels or list(LABELS)
    mat = matrix.astype(float)
    if normalize:
        row_sums = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else mat.max())
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            ax.text(
                j, i, f"{val:.2f}" if normalize else f"{int(matrix[i, j])}",
                ha="center", va="center",
                color="white" if val > 0.5 else "black", fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_roc_curves(
    y_true: Sequence[str],
    y_proba: np.ndarray,
    out_path: str | Path,
    labels: List[str] | None = None,
    title: str = "ROC curves (one-vs-rest)",
) -> Path:
    """Render one-vs-rest ROC curves for each class."""
    labels = labels or list(LABELS)
    y_idx = np.array([labels.index(y) for y in y_true])

    fig, ax = plt.subplots(figsize=(5, 4.2))
    for cls_id, label in enumerate(labels):
        binary_true = (y_idx == cls_id).astype(int)
        if binary_true.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_true, y_proba[:, cls_id])
        ax.plot(fpr, tpr, label=label, linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_model_comparison(
    scores: Dict[str, Dict[str, float]],
    out_path: str | Path,
    metric_keys: Sequence[str] = ("accuracy", "f1_macro", "roc_auc_ovr_macro"),
    title: str = "Model comparison",
) -> Path:
    """Grouped bar chart comparing models across metrics."""
    models = list(scores.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for i, model in enumerate(models):
        vals = [scores[model].get(k, float("nan")) for k in metric_keys]
        ax.bar(x + i * width, vals, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2, list(metric_keys), rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
