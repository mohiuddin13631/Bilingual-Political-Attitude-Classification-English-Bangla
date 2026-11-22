"""Model factory: build a concrete model from a resolved config.

Neural models are imported lazily so that the pure-baseline path never requires
PyTorch / Transformers to be installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..utils.config import Config
from .base import SentimentModel
from .baseline import TfidfLinearModel

if TYPE_CHECKING:  # pragma: no cover
    pass


def build_model(cfg: Config) -> SentimentModel:
    """Instantiate a model given the experiment config's ``model`` block."""
    model_cfg = cfg.model
    model_type = model_cfg.type

    if model_type == "tfidf_linear":
        clf = model_cfg.classifier
        tfidf = model_cfg.tfidf
        clf_params = model_cfg.get(clf, Config({})).to_dict()
        return TfidfLinearModel(
            classifier=clf,
            ngram_range=tfidf.ngram_range,
            max_features=tfidf.max_features,
            min_df=tfidf.min_df,
            sublinear_tf=tfidf.sublinear_tf,
            calibrate_probabilities=cfg.get("train", Config({})).get(
                "calibrate_probabilities", True
            ),
            **clf_params,
        )

    if model_type == "bilstm_attention":
        from .bilstm import BiLSTMAttentionModel  # lazy: requires torch

        return BiLSTMAttentionModel.from_config(cfg)

    if model_type == "transformer":
        from .transformer import TransformerModel  # lazy: requires torch+transformers

        return TransformerModel.from_config(cfg)

    raise ValueError(f"Unknown model.type: {model_type}")
