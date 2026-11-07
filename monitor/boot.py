"""Vectorised stationary bootstrap (Politis & Romano 1994) used by the monitor and vendor test.

stationary_indices(n, mean_block, n_boot, rng, length=None) -> int array (n_boot, length)
Each path starts at a uniform random row; at every step, with prob 1/mean_block it jumps to a
new uniform random row, otherwise it moves to the next row (wrapping circularly).
"""
from __future__ import annotations

import math

import numpy as np


def stationary_indices(n: int, mean_block: float, n_boot: int, rng: np.random.Generator,
                       length: int | None = None) -> np.ndarray:
    length = n if length is None else length
    p = 1.0 / float(mean_block)
    idx = np.empty((n_boot, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, n_boot)
    jumps = rng.random((n_boot, length)) < p
    fresh = rng.integers(0, n, (n_boot, length))
    for j in range(1, length):
        nxt = (idx[:, j - 1] + 1) % n
        idx[:, j] = np.where(jumps[:, j], fresh[:, j], nxt)
    return idx


def p_block_fallback(x, n: int = 1000, seed: int = 42) -> float:
    """One-sided stationary-bootstrap p of mean > 0 (same recipe as M1.5). Used only if
    engine.nulls is not importable."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 4:
        return 1.0
    try:
        from arch.bootstrap import optimal_block_length
        b = float(optimal_block_length(x)["stationary"].iloc[0])
        b = math.ceil(b) if np.isfinite(b) else math.ceil(T ** (1 / 3))
    except Exception:
        b = math.ceil(T ** (1 / 3))
    b = int(min(max(b, 2), max(2, T // 4)))
    x0 = x - x.mean()
    idx = stationary_indices(T, b, n, np.random.default_rng(seed))
    m = x0[idx].mean(axis=1)
    return float((1 + np.sum(m >= x.mean())) / (n + 1))


def get_p_block():
    try:
        from engine.nulls import p_block  # type: ignore
        return p_block
    except Exception:
        return p_block_fallback
