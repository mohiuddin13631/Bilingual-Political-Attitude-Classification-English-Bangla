"""Deterministic, stratified train/validation/test splitting."""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def make_splits(
    frame: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_by: str = "label",
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Split a DataFrame into stratified train/val/test partitions.

    The split is performed in two stages (carve out test, then val from the
    remainder) so that ``val_size`` and ``test_size`` are both expressed as
    fractions of the *original* dataset.

    Returns
    -------
    dict with keys ``train``, ``val``, ``test``.
    """
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be in (0,1) and sum to < 1.")

    strat = frame[stratify_by] if stratify_by in frame.columns else None
    train_val, test = train_test_split(
        frame, test_size=test_size, random_state=seed, shuffle=True, stratify=strat
    )

    # val fraction relative to the remaining train_val partition
    val_relative = val_size / (1.0 - test_size)
    strat_tv = train_val[stratify_by] if stratify_by in train_val.columns else None
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=seed,
        shuffle=True,
        stratify=strat_tv,
    )

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def split_summary(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return a small table of per-split label counts for logging/EDA."""
    rows = []
    for name, part in splits.items():
        counts = part["label"].value_counts().to_dict()
        counts["split"] = name
        counts["n"] = len(part)
        rows.append(counts)
    return pd.DataFrame(rows).set_index("split")
