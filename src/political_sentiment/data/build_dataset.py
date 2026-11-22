"""Entry point: build the dataset (generate or load) and write splits to disk.

Usage
-----
    python -m political_sentiment.data.build_dataset --config configs/base.yaml

Writes:
    data/synthetic/corpus.csv      (when source == synthetic)
    data/processed/train.parquet
    data/processed/val.parquet
    data/processed/test.parquet
    data/processed/split_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..utils import get_logger, load_config, seed_everything
from .io import save_frame
from .loaders import load_csv, load_huggingface
from .schema import records_to_frame
from .split import make_splits, split_summary
from .synthetic import SyntheticCorpusGenerator


def build(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = seed_everything(cfg.seed)
    logger = get_logger(
        "data.build", cfg.logging.level, log_file=f"{cfg.logging.log_dir}/build_data.log"
    )
    logger.info("Building dataset (seed=%s, source=%s)", seed, cfg.data.source)

    processed_dir = Path(cfg.paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1. Obtain records from the configured source ------------------------- #
    source = cfg.data.source
    if source == "synthetic":
        gen = SyntheticCorpusGenerator(
            languages=cfg.data.languages,
            parties=cfg.data.parties,
            topics=cfg.data.topics,
            labels=cfg.data.labels,
            seed=seed,
            sarcasm_rate=cfg.data.get("sarcasm_rate", 0.12),
            negation_rate=cfg.data.get("negation_rate", 0.18),
            hedge_rate=cfg.data.get("hedge_rate", 0.25),
            contrast_rate=cfg.data.get("contrast_rate", 0.22),
            label_noise=cfg.data.get("label_noise", 0.06),
        )
        balanced = cfg.data.class_balance == "balanced"
        records = gen.generate(cfg.data.n_samples, balanced=balanced)
        frame = records_to_frame(records)
        syn_dir = Path(cfg.paths.synthetic_dir)
        syn_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(syn_dir / "corpus.csv", index=False)
        logger.info("Generated %d synthetic records -> %s", len(frame), syn_dir / "corpus.csv")
    elif source == "csv":
        records = load_csv(cfg.data.csv_path)
        frame = records_to_frame(records)
        logger.info("Loaded %d records from CSV", len(frame))
    elif source == "huggingface":
        records = load_huggingface(cfg.data.hf_name, split=cfg.data.get("hf_split", "train"))
        frame = records_to_frame(records)
        logger.info("Loaded %d records from HuggingFace", len(frame))
    else:  # pragma: no cover - guarded by config
        raise ValueError(f"Unknown data.source: {source}")

    # 2. Stratified split -------------------------------------------------- #
    splits = make_splits(
        frame,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        stratify_by=cfg.split.stratify_by,
        seed=seed,
    )
    for name, part in splits.items():
        out = save_frame(part, processed_dir, name)
        logger.info("Wrote %s (%d rows)", out, len(part))

    summary = split_summary(splits)
    summary.to_csv(processed_dir / "split_summary.csv")
    logger.info("Split summary:\n%s", summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the sentiment dataset.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to config YAML.")
    args = parser.parse_args()
    build(args.config)


if __name__ == "__main__":
    main()
