"""Train several model configs and emit a side-by-side comparison.

Usage
-----
    python -m political_sentiment.training.compare \\
        --configs configs/baseline_tfidf.yaml configs/bilstm_attention.yaml configs/transformer_bert.yaml

Each config is trained and evaluated on the shared test split; results are
collected into ``artifacts/comparison/summary.csv`` and a grouped-bar figure.
Configs whose backend is unavailable (e.g. a transformer config when PyTorch is
not installed) are skipped with a warning, so the script still produces a
partial comparison in constrained environments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from ..evaluation.plots import plot_model_comparison
from ..utils import get_logger
from .train import run_training

logger = get_logger("compare")


def compare(config_paths: List[str], out_dir: str = "artifacts/comparison") -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores: Dict[str, Dict[str, float]] = {}
    for path in config_paths:
        name = Path(path).stem
        try:
            metrics = run_training(path)
            scores[name] = metrics
            logger.info("%s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})
        except ImportError as exc:
            logger.warning("Skipping %s (missing backend): %s", name, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed %s: %s", name, exc)

    if not scores:
        logger.error("No models trained successfully.")
        return pd.DataFrame()

    df = pd.DataFrame(scores).transpose()
    df.to_csv(out / "summary.csv")
    (out / "summary.json").write_text(json.dumps(scores, indent=2))
    plot_model_comparison(scores, out / "model_comparison.png",
                          title="Model comparison (test set)")
    logger.info("Comparison written to %s", out)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple model configs.")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", default="artifacts/comparison")
    args = parser.parse_args()
    df = compare(args.configs, args.out)
    if not df.empty:
        print(df.round(4).to_string())


if __name__ == "__main__":
    main()
