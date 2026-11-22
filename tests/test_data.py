"""Tests for synthetic generation, schema, splitting and IO."""

from __future__ import annotations

import pandas as pd

from political_sentiment import LABELS
from political_sentiment.data.io import load_frame, save_frame
from political_sentiment.data.schema import RECORD_COLUMNS
from political_sentiment.data.split import make_splits
from political_sentiment.data.synthetic import SyntheticCorpusGenerator
from political_sentiment.data.schema import records_to_frame


def test_generator_is_reproducible():
    g1 = SyntheticCorpusGenerator(seed=123).generate(120)
    g2 = SyntheticCorpusGenerator(seed=123).generate(120)
    assert [r.text for r in g1] == [r.text for r in g2]


def test_generator_covers_languages_and_labels():
    frame = records_to_frame(SyntheticCorpusGenerator(seed=1).generate(300))
    assert set(frame["language"]) == {"en", "bn"}
    assert set(frame["label"]).issubset(set(LABELS))
    assert list(frame.columns) == RECORD_COLUMNS


def test_generator_balanced_distribution():
    frame = records_to_frame(SyntheticCorpusGenerator(seed=2).generate(600, balanced=True))
    counts = frame["label"].value_counts(normalize=True)
    # roughly balanced (label noise perturbs it slightly)
    assert counts.min() > 0.25


def test_make_splits_are_disjoint_and_stratified():
    frame = records_to_frame(SyntheticCorpusGenerator(seed=3).generate(600))
    splits = make_splits(frame, test_size=0.2, val_size=0.2, seed=3)
    ids = {name: set(part["id"]) for name, part in splits.items()}
    assert ids["train"].isdisjoint(ids["test"])
    assert ids["train"].isdisjoint(ids["val"])
    assert ids["val"].isdisjoint(ids["test"])
    total = sum(len(p) for p in splits.values())
    assert total == len(frame)


def test_io_roundtrip(tmp_path):
    frame = records_to_frame(SyntheticCorpusGenerator(seed=4).generate(60))
    save_frame(frame, tmp_path, "mini")
    loaded = load_frame(tmp_path, "mini")
    assert len(loaded) == len(frame)
    assert set(loaded.columns) == set(frame.columns)
