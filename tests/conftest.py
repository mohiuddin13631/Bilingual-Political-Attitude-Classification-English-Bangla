"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from political_sentiment.data.synthetic import SyntheticCorpusGenerator


@pytest.fixture(scope="session")
def small_corpus() -> pd.DataFrame:
    """A tiny deterministic synthetic corpus for fast tests."""
    gen = SyntheticCorpusGenerator(
        languages=("en", "bn"),
        parties=("PartyA", "PartyB"),
        topics=("economy", "healthcare"),
        seed=7,
    )
    from political_sentiment.data.schema import records_to_frame

    return records_to_frame(gen.generate(300, balanced=True))


@pytest.fixture(scope="session")
def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False
