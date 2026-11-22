"""Evaluation: metrics, plots, error analysis."""

from .metrics import compute_metrics, per_class_report
from .plots import plot_confusion_matrix, plot_roc_curves

__all__ = ["compute_metrics", "per_class_report", "plot_confusion_matrix", "plot_roc_curves"]
