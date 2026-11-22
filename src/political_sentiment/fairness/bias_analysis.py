"""Bias & fairness analysis across political groups and topics.

Political sentiment models are especially prone to bias: a model may
systematically assign more negative sentiment to one party, or perform worse on
one language, encoding or amplifying real-world asymmetries. This module
quantifies three complementary lenses (report Section VI):

1. **Group performance parity** — is accuracy / F1 comparable across parties,
   topics and languages? Large gaps indicate the model serves some groups worse.
2. **Predicted-sentiment distribution** — does the model assign systematically
   different sentiment *mixes* to each party, beyond what the gold labels
   justify? This is the political-favourability bias that matters most here.
3. **Fairness gap summary** — headline max-min disparities suitable for a table.

These metrics are descriptive diagnostics, not a certificate of fairness; they
are meant to be read alongside the qualitative error analysis.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..evaluation.metrics import compute_metrics


def group_performance(
    frame: pd.DataFrame,
    y_pred: Sequence[str],
    group_col: str,
    y_proba: np.ndarray | None = None,
    labels: List[str] | None = None,
) -> pd.DataFrame:
    """Per-group accuracy / macro-F1 (and ROC-AUC if probabilities given)."""
    df = frame.reset_index(drop=True).copy()
    df["predicted"] = list(y_pred)
    rows = []
    for value, sub in df.groupby(group_col):
        idx = sub.index.to_numpy()
        proba = y_proba[idx] if y_proba is not None else None
        m = compute_metrics(sub["label"].tolist(), sub["predicted"].tolist(), proba, labels)
        m[group_col] = value
        m["n"] = len(sub)
        rows.append(m)
    cols = [group_col, "n", "accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr_macro"]
    out = pd.DataFrame(rows)
    return out[[c for c in cols if c in out.columns]].sort_values("f1_macro", ascending=False)


def sentiment_distribution_by_group(
    frame: pd.DataFrame,
    y_pred: Sequence[str],
    group_col: str,
    labels: List[str] | None = None,
) -> pd.DataFrame:
    """Compare the *predicted* vs *gold* sentiment mix per group.

    A model exhibiting political favourability bias will show predicted-sentiment
    shares for a party that diverge from that party's gold-sentiment shares — for
    example predicting ``negative`` for Party A more often than the labels warrant.
    The returned frame reports, per group and class, gold share, predicted share,
    and their difference.
    """
    from .. import LABELS

    labels = labels or list(LABELS)
    df = frame.reset_index(drop=True).copy()
    df["predicted"] = list(y_pred)

    rows = []
    for value, sub in df.groupby(group_col):
        n = len(sub)
        for label in labels:
            gold_share = float((sub["label"] == label).mean())
            pred_share = float((sub["predicted"] == label).mean())
            rows.append(
                {
                    group_col: value,
                    "sentiment": label,
                    "gold_share": round(gold_share, 4),
                    "pred_share": round(pred_share, 4),
                    "pred_minus_gold": round(pred_share - gold_share, 4),
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def subgroup_fairness_gaps(perf: pd.DataFrame, metric: str = "f1_macro") -> Dict[str, float]:
    """Summarise disparity for a group-performance table.

    Returns the max, min, absolute gap and standard deviation of ``metric``
    across groups — the headline numbers for the fairness table.
    """
    values = perf[metric].dropna().to_numpy()
    if values.size == 0:
        return {"max": float("nan"), "min": float("nan"), "gap": float("nan"), "std": float("nan")}
    return {
        "max": float(values.max()),
        "min": float(values.min()),
        "gap": float(values.max() - values.min()),
        "std": float(values.std()),
    }
