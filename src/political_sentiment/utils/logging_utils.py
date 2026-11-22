"""Project-wide logging helper.

Provides a single :func:`get_logger` factory so every module logs with a
consistent format and, optionally, to a run-specific file. Logged config +
metrics form part of the reproducibility trail.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name:
        Logger name, conventionally ``__name__`` of the calling module.
    level:
        Logging level as a string (e.g. ``"INFO"``, ``"DEBUG"``).
    log_file:
        Optional path; when given, logs are also written there.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    logger.propagate = False

    if logger.handlers:  # already configured
        return logger

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
