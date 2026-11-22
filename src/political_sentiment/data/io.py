"""Frame persistence helpers with a graceful parquet -> CSV fallback.

Parquet is preferred (typed, compact) but requires ``pyarrow``/``fastparquet``.
When neither is installed we transparently fall back to UTF-8 CSV, which fully
preserves the Bangla text. Downstream code addresses splits by *stem* (e.g.
``train``) and lets these helpers resolve the concrete file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parquet_available() -> bool:
    for mod in ("pyarrow", "fastparquet"):
        try:
            __import__(mod)
            return True
        except ImportError:
            continue
    return False


def save_frame(frame: pd.DataFrame, directory: str | Path, stem: str) -> Path:
    """Save ``frame`` as ``<stem>.parquet`` if possible, else ``<stem>.csv``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if _parquet_available():
        path = directory / f"{stem}.parquet"
        frame.to_parquet(path, index=False)
    else:
        path = directory / f"{stem}.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
    return path


def load_frame(directory: str | Path, stem: str) -> pd.DataFrame:
    """Load ``<stem>.parquet`` if present, else ``<stem>.csv``."""
    directory = Path(directory)
    parquet_path = directory / f"{stem}.parquet"
    csv_path = directory / f"{stem}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, encoding="utf-8")
    raise FileNotFoundError(
        f"No dataset found for stem '{stem}' in {directory} (tried .parquet and .csv)."
    )
