"""Canonical record schema shared across generators, loaders and models.

Keeping every data source funnel into the same :class:`SentimentRecord`
dataclass means the preprocessing, training and fairness code never has to care
whether an example came from the synthetic generator, a CSV, or a HuggingFace
dataset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Sequence

import pandas as pd

RECORD_COLUMNS: List[str] = [
    "id",
    "text",
    "label",
    "language",
    "party",
    "topic",
    "source",
]


@dataclass(frozen=True)
class SentimentRecord:
    """A single labelled social-media post.

    Attributes
    ----------
    id:
        Stable unique identifier.
    text:
        Raw post text (pre-cleaning).
    label:
        One of ``negative`` | ``neutral`` | ``positive``.
    language:
        ISO code, ``en`` or ``bn``.
    party:
        Political entity the post is *about* — used by the fairness analysis,
        not as a model input.
    topic:
        Policy topic (economy, healthcare, ...), also a fairness slice.
    source:
        Provenance tag (e.g. ``synthetic``, ``semeval``, ``csv``).
    """

    id: str
    text: str
    label: str
    language: str
    party: str
    topic: str
    source: str


def records_to_frame(records: Sequence[SentimentRecord]) -> pd.DataFrame:
    """Convert a sequence of records into a column-ordered DataFrame."""
    frame = pd.DataFrame([asdict(r) for r in records])
    return frame[RECORD_COLUMNS]
