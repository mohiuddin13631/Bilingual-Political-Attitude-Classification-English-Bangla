"""Full evaluation entry point: metrics, figures, fairness, error analysis.

Loads a trained model from its artifacts directory, evaluates it on the test
split, and writes a complete evaluation bundle:

    <artifacts>/<experiment>/eval/
        metrics.json
        confusion_matrix.png
        roc_curves.png
        per_class_report.csv
        fairness_by_party.csv
        fairness_by_language.csv
        sentiment_distribution_by_party.csv
        top_errors.csv

Usage
-----
    python -m political_sentiment.evaluation.evaluate --config configs/baseline_tfidf.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..data.io import load_frame
from ..fairness.bias_analysis import (
    group_performance,
    sentiment_distribution_by_group,
    subgroup_fairness_gaps,
)
from ..utils import get_logger, load_config, seed_everything
from .error_analysis import collect_errors, error_rate_by
from .metrics import compute_metrics, confusion, per_class_report
from .plots import plot_confusion_matrix, plot_roc_curves


def _load_model(cfg):  # noqa: ANN001
    """Load the trained model based on its declared type (lazy imports)."""
    model_type = cfg.model.type
    model_dir = Path(cfg.paths.artifacts_dir) / cfg.experiment_name / "model"
    if model_type == "tfidf_linear":
        from ..models.baseline import TfidfLinearModel

        return TfidfLinearModel.load(model_dir)
    if model_type == "bilstm_attention":
        from ..models.bilstm import BiLSTMAttentionModel

        return BiLSTMAttentionModel.load(model_dir)
    if model_type == "transformer":
        from ..models.transformer import TransformerModel

        return TransformerModel.load(model_dir)
    raise ValueError(f"Unknown model.type: {model_type}")


def evaluate(config_path: str) -> dict:
    cfg = load_config(config_path)
    seed_everything(cfg.seed)
    exp_name = cfg.experiment_name
    out_dir = Path(cfg.paths.artifacts_dir) / exp_name / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"eval.{exp_name}", cfg.logging.level)

    model = _load_model(cfg)
    test_df = load_frame(cfg.paths.processed_dir, "test")
    y_true = test_df["label"].tolist()

    proba = model.predict_proba(test_df["text"].tolist())
    preds = [model.labels[i] for i in proba.argmax(axis=1)]

    # Headline metrics + per-class report ------------------------------- #
    metrics = compute_metrics(y_true, preds, proba, labels=model.labels)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    report = per_class_report(y_true, preds, labels=model.labels)
    pd.DataFrame(report).transpose().to_csv(out_dir / "per_class_report.csv")

    # Figures ------------------------------------------------------------ #
    cm = confusion(y_true, preds, labels=model.labels)
    plot_confusion_matrix(cm, out_dir / "confusion_matrix.png", labels=model.labels,
                          title=f"{exp_name}: confusion matrix")
    plot_roc_curves(y_true, proba, out_dir / "roc_curves.png", labels=model.labels,
                    title=f"{exp_name}: ROC (one-vs-rest)")

    # Fairness ----------------------------------------------------------- #
    if "party" in test_df.columns:
        perf_party = group_performance(test_df, preds, "party", proba, model.labels)
        perf_party.to_csv(out_dir / "fairness_by_party.csv", index=False)
        sentiment_distribution_by_group(test_df, preds, "party", model.labels).to_csv(
            out_dir / "sentiment_distribution_by_party.csv", index=False
        )
        gaps = subgroup_fairness_gaps(perf_party, "f1_macro")
        logger.info("Party F1 fairness gap: %.4f (max=%.4f min=%.4f)",
                    gaps["gap"], gaps["max"], gaps["min"])
    if "language" in test_df.columns:
        group_performance(test_df, preds, "language", proba, model.labels).to_csv(
            out_dir / "fairness_by_language.csv", index=False
        )
        error_rate_by(test_df, preds, "language").to_csv(out_dir / "error_rate_by_language.csv")

    # Error analysis ----------------------------------------------------- #
    errors = collect_errors(test_df, preds, proba, model.labels)
    errors.head(50).to_csv(out_dir / "top_errors.csv", index=False)

    logger.info("Evaluation bundle written to %s", out_dir)
    logger.info("Metrics: %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sentiment model.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
