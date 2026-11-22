"""Tests for metrics, confusion matrix, plots and fairness diagnostics."""

from __future__ import annotations

import numpy as np

from political_sentiment.evaluation.metrics import compute_metrics, confusion
from political_sentiment.evaluation.plots import plot_confusion_matrix
from political_sentiment.fairness.bias_analysis import (
    group_performance,
    sentiment_distribution_by_group,
    subgroup_fairness_gaps,
)
from political_sentiment.models.baseline import TfidfLinearModel


def test_compute_metrics_perfect():
    y = ["negative", "neutral", "positive"]
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_compute_metrics_with_proba_roc():
    y_true = ["negative", "positive", "neutral", "positive"]
    y_pred = ["negative", "positive", "neutral", "negative"]
    proba = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.2, 0.6, 0.2], [0.5, 0.2, 0.3]])
    m = compute_metrics(y_true, y_pred, proba)
    assert 0.0 <= m["roc_auc_ovr_macro"] <= 1.0
    assert 0.0 < m["accuracy"] < 1.0


def test_confusion_matrix_shape():
    y = ["negative", "neutral", "positive", "negative"]
    p = ["negative", "neutral", "negative", "negative"]
    cm = confusion(y, p)
    assert cm.shape == (3, 3)
    assert cm.sum() == 4


def test_plot_confusion_matrix_writes_file(tmp_path):
    cm = np.array([[5, 1, 0], [0, 4, 1], [1, 0, 6]])
    out = plot_confusion_matrix(cm, tmp_path / "cm.png")
    assert out.exists() and out.stat().st_size > 0


def test_group_performance_and_gaps(small_corpus):
    model = TfidfLinearModel().fit(small_corpus)
    proba = model.predict_proba(small_corpus["text"].tolist())
    preds = [model.labels[i] for i in proba.argmax(axis=1)]
    perf = group_performance(small_corpus, preds, "party", proba, model.labels)
    assert "f1_macro" in perf.columns
    gaps = subgroup_fairness_gaps(perf, "f1_macro")
    assert gaps["gap"] >= 0.0


def test_sentiment_distribution_columns(small_corpus):
    model = TfidfLinearModel().fit(small_corpus)
    preds = model.predict(small_corpus["text"].tolist())
    dist = sentiment_distribution_by_group(small_corpus, preds, "party", model.labels)
    assert {"gold_share", "pred_share", "pred_minus_gold"}.issubset(dist.columns)
