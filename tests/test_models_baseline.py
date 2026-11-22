"""Tests for the TF-IDF linear baseline and the shared model interface."""

from __future__ import annotations

import numpy as np

from political_sentiment.models.baseline import TfidfLinearModel


def test_baseline_fit_predict_shapes(small_corpus):
    model = TfidfLinearModel(classifier="logistic_regression").fit(small_corpus)
    texts = small_corpus["text"].tolist()[:10]
    proba = model.predict_proba(texts)
    assert proba.shape == (10, 3)
    # probabilities sum to ~1
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    preds = model.predict(texts)
    assert all(p in model.labels for p in preds)


def test_baseline_learns_signal(small_corpus):
    model = TfidfLinearModel().fit(small_corpus)
    # Clear-cut cases the model should get right.
    assert model.predict(["This is excellent and inspiring."])[0] == "positive"
    assert model.predict(["This is a disaster and an embarrassment."])[0] == "negative"


def test_baseline_top_features(small_corpus):
    model = TfidfLinearModel().fit(small_corpus)
    feats = model.top_features(5)
    assert set(feats.keys()) == set(model.labels)


def test_baseline_save_load_roundtrip(small_corpus, tmp_path):
    model = TfidfLinearModel().fit(small_corpus)
    model.save(tmp_path / "m")
    reloaded = TfidfLinearModel.load(tmp_path / "m")
    a = model.predict_proba(["a good policy"])
    b = reloaded.predict_proba(["a good policy"])
    assert np.allclose(a, b)


def test_svm_variant_produces_probabilities(small_corpus):
    model = TfidfLinearModel(classifier="linear_svm", calibrate_probabilities=True).fit(small_corpus)
    proba = model.predict_proba(["a reckless and harmful plan"])
    assert proba.shape == (1, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
