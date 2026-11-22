"""Unified training entry point for every model.

Usage
-----
    python -m political_sentiment.training.train --config configs/baseline_tfidf.yaml
    python -m political_sentiment.training.train --config configs/bilstm_attention.yaml
    python -m political_sentiment.training.train --config configs/transformer_bert.yaml

For reproducibility every run:
  * seeds all RNGs from ``cfg.seed``;
  * writes the *resolved* config next to the artifacts;
  * evaluates on the held-out test split and writes ``metrics.json``.

Artifacts are written to ``<artifacts_dir>/<experiment_name>/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.io import load_frame
from ..evaluation.metrics import compute_metrics, per_class_report
from ..models.registry import build_model
from ..utils import get_logger, load_config, seed_everything


def run_training(config_path: str) -> dict:
    """Train, evaluate and persist a model. Returns the test-set metrics dict."""
    cfg = load_config(config_path)
    seed = seed_everything(cfg.seed)

    exp_name = cfg.get("experiment_name", "experiment")
    out_dir = Path(cfg.paths.artifacts_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"train.{exp_name}", cfg.logging.level, log_file=str(out_dir / "train.log"))
    logger.info("=== Training '%s' (seed=%d) ===", exp_name, seed)

    # 1. Data ------------------------------------------------------------- #
    processed = cfg.paths.processed_dir
    train_df = load_frame(processed, "train")
    val_df = load_frame(processed, "val")
    test_df = load_frame(processed, "test")
    logger.info("Loaded splits: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # 2. Persist the resolved config (reproducibility log) --------------- #
    cfg.to_yaml(out_dir / "resolved_config.yaml")

    # 3. Build + fit ------------------------------------------------------ #
    model = build_model(cfg)
    logger.info("Built model: %s", cfg.model.type)
    model.fit(train_df, val_df)
    logger.info("Training complete.")

    # 4. Persist model ---------------------------------------------------- #
    model_dir = out_dir / "model"
    model.save(model_dir)
    logger.info("Saved model to %s", model_dir)

    # 5. Evaluate on test ------------------------------------------------- #
    proba = model.predict_proba(test_df["text"].tolist())
    preds = [model.labels[i] for i in proba.argmax(axis=1)]
    metrics = compute_metrics(test_df["label"].tolist(), preds, proba, labels=model.labels)
    report = per_class_report(test_df["label"].tolist(), preds, labels=model.labels)

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "classification_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Test metrics: %s", json.dumps({k: round(v, 4) for k, v in metrics.items()}))

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a sentiment model.")
    parser.add_argument("--config", required=True, help="Path to an experiment config YAML.")
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
