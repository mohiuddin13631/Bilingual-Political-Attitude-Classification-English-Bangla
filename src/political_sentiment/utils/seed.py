"""Deterministic seeding across Python, NumPy and PyTorch.

Reproducibility is a first-class requirement of this project (see report
Section VII). :func:`seed_everything` should be called once, early, in every
entry point (training scripts, notebooks, tests).
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all relevant RNGs and return the seed used.

    Parameters
    ----------
    seed:
        The integer seed applied to Python's ``random``, NumPy and (if
        available) PyTorch, including CUDA.
    deterministic:
        When True, request deterministic cuDNN kernels. This can slow training
        slightly but removes a major source of run-to-run variance.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for the pure-baseline path
        pass

    return seed
