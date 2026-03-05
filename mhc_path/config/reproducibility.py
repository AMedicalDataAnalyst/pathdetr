"""Reproducibility & Seed Management.

Centralized seed management and determinism configuration for all
training engines and the experiment runner in the mHC-Path pipeline.
"""

from __future__ import annotations

import os
import random
from typing import Any, Callable

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set seeds for Python, NumPy, PyTorch, and CUDA.

    Parameters
    ----------
    seed : int
        The random seed to set across all libraries.
    deterministic : bool, default=True
        When True, also enables full deterministic mode:
        ``torch.use_deterministic_algorithms(True)``,
        ``cudnn.deterministic = True``, ``cudnn.benchmark = False``,
        and ``CUBLAS_WORKSPACE_CONFIG=":4096:8"``.

    Notes
    -----
    CUBLAS_WORKSPACE_CONFIG must be set to handle non-determinism in
    cuBLAS GEMM operations on Ampere+ GPUs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Env var must be set before any cuBLAS call to take effect
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_run_seed(base_seed: int, run_id: int) -> int:
    """Derive a deterministic per-run seed via Knuth multiplicative hash.

    Parameters
    ----------
    base_seed : int
        The base seed shared across all runs of an experiment.
    run_id : int
        The unique identifier for this particular run.

    Returns
    -------
    int
        A deterministic seed in the range ``[0, 2**32)``.

    Notes
    -----
    Uses the Knuth multiplicative hash constant 2654435761, which is
    the closest prime to ``2**32 * (sqrt(5) - 1) / 2`` (the golden
    ratio). Collision-free for run_id in ``[0, 1000]``.
    """
    return (base_seed * 2654435761 + run_id) % (2**32)


def check_determinism(
    fn: Callable[..., Any],
    *args: Any,
    n_trials: int = 3,
    **kwargs: Any,
) -> bool:
    """Run *fn* multiple times from the same seed and verify bitwise identity.

    Parameters
    ----------
    fn : Callable
        The function to test for determinism.
    *args : Any
        Positional arguments forwarded to *fn*.
    n_trials : int, default=3
        Number of repeated invocations to compare.
    **kwargs : Any
        Keyword arguments forwarded to *fn*.

    Returns
    -------
    bool
        True if every trial produced bitwise-identical outputs,
        False otherwise.
    """
    reference: bytes | None = None

    for _ in range(n_trials):
        seed_everything(42, deterministic=True)
        result = fn(*args, **kwargs)

        # Serialise to bytes for bitwise comparison
        if isinstance(result, torch.Tensor):
            trial_bytes = result.detach().cpu().numpy().tobytes()
        elif isinstance(result, np.ndarray):
            trial_bytes = result.tobytes()
        else:
            trial_bytes = bytes(str(result), "utf-8")

        if reference is None:
            reference = trial_bytes
        elif trial_bytes != reference:
            return False

    return True
