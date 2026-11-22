"""Loaders for real / external datasets, funnelled into :class:`SentimentRecord`.

These are the documented "plug in your own data" hooks. Because most political
social-media corpora cannot be redistributed, the repo ships with the synthetic
generator by default; set ``data.source: csv`` or ``data.source: huggingface``
in the config to use one of these instead.

Recommended real public datasets (see report Section III for licensing notes):

* **SemEval-2016 Task 6 / Task 4** — English tweets with stance & sentiment.
  Academic use; requires accepting the task terms.
* **GermEval / TASS** — German / Spanish sentiment shared tasks (if extending
  beyond EN+BN).
* **SentNoB / BanglaBook** — public Bangla sentiment datasets on HuggingFace
  (CC-BY variants); good fit for the Bangla slice here.
* **EU Parliament debates / ParlSpeech** — long-form political speech, CC-BY.

All loaders normalise arbitrary label spaces to the project's three-class
scheme via :data:`LABEL_ALIASES`.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .schema import SentimentRecord

# Map common external label spellings onto the canonical three classes.
LABEL_ALIASES: Dict[str, str] = {
    "negative": "negative", "neg": "negative", "0": "negative", "-1": "negative",
    "against": "negative", "bad": "negative",
    "neutral": "neutral", "neu": "neutral", "none": "neutral", "2": "neutral",
    "positive": "positive", "pos": "positive", "1": "positive", "4": "positive",
    "favor": "positive", "good": "positive",
}


def normalise_label(raw: object) -> Optional[str]:
    """Map a raw label to the canonical scheme, or None if unknown."""
    key = str(raw).strip().lower()
    return LABEL_ALIASES.get(key)


def load_csv(
    path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
    language: str = "en",
    party_col: Optional[str] = None,
    topic_col: Optional[str] = None,
    source: str = "csv",
) -> List[SentimentRecord]:
    """Load labelled examples from a CSV file.

    Rows whose label cannot be mapped to the canonical scheme are dropped (with
    the count discoverable by comparing input/output lengths).
    """
    frame = pd.read_csv(path)
    records: List[SentimentRecord] = []
    for _, row in frame.iterrows():
        label = normalise_label(row[label_col])
        if label is None:
            continue
        records.append(
            SentimentRecord(
                id=str(uuid.uuid4()),
                text=str(row[text_col]),
                label=label,
                language=str(row.get("language", language)),
                party=str(row[party_col]) if party_col and party_col in row else "unknown",
                topic=str(row[topic_col]) if topic_col and topic_col in row else "unknown",
                source=source,
            )
        )
    return records


def load_huggingface(
    dataset_name: str,
    split: str = "train",
    text_col: str = "text",
    label_col: str = "label",
    language: str = "en",
    source: Optional[str] = None,
) -> List[SentimentRecord]:
    """Load a dataset from the HuggingFace Hub.

    Imported lazily so the base install does not require ``datasets``. Raises a
    clear error if the dependency is missing.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'datasets' package is required for load_huggingface(). "
            "Install with `pip install datasets`."
        ) from exc

    ds = load_dataset(dataset_name, split=split)
    records: List[SentimentRecord] = []
    for row in ds:
        label = normalise_label(row[label_col])
        if label is None:
            continue
        records.append(
            SentimentRecord(
                id=str(uuid.uuid4()),
                text=str(row[text_col]),
                label=label,
                language=str(row.get("language", language)),
                party=str(row.get("party", "unknown")),
                topic=str(row.get("topic", "unknown")),
                source=source or dataset_name,
            )
        )
    return records
