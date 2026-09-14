"""
Global reproducibility helper.

`random_state` was already threaded through every sklearn estimator and every
split in benchmark.py -- that part was fine. What was missing is a single
place that also seeds Python's `random`, NumPy's global RNG, and (once
installed) PyTorch/CUDA, so that a full run -- including any deep-learning
models -- is reproducible end to end, not just the sklearn pieces.

`benchmark()` calls this internally using the existing `random_state`
parameter, so nothing changes for callers who don't touch torch. It's also
exported as a public function for anyone seeding their own script/notebook
before calling into mlbench-lite.
"""
from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python's `random`, NumPy, and (if installed) PyTorch/CUDA.

    Parameters
    ----------
    seed : int
        The seed to use everywhere.
    deterministic : bool, optional
        If True and PyTorch is installed, also forces cuDNN into
        deterministic mode (`torch.backends.cudnn.deterministic = True`,
        `benchmark = False`). This can noticeably slow down GPU training, so
        it's opt-in -- most people just want "same seed -> same numbers" on
        CPU/Colab, which doesn't need this flag.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
