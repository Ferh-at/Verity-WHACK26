"""Null distributions (M1.5): stationary bootstrap (Politis & Romano 1994) and circular shift.

The resampling is a vectorised numpy implementation of the same algorithm as
arch.bootstrap.StationaryBootstrap (geometric block lengths with mean `block_len`, wrap-around);
the per-replicate Python loop in arch is too slow for the benchmark."""
import math

import numpy as np
import pandas as pd

from config import MIN_SHIFT, N_BOOT, SEED
from engine.stats import nw_t


def block_len(x) -> int:
    x = np.asarray(x, dtype=float)
    T = len(x)
    hi = max(2, T // 4)
    try:
        from arch.bootstrap import optimal_block_length
        b = float(optimal_block_length(x)["stationary"].iloc[0])
        if not math.isfinite(b):
            raise ValueError
        b = math.ceil(b)
    except Exception:
        b = math.ceil(T ** (1 / 3))
    return int(min(max(b, 2), hi))


def stationary_indices(T: int, n: int, block: float, rng: np.random.Generator) -> np.ndarray:
    """(n, T) index matrix of stationary-bootstrap resamples."""
    starts = rng.integers(0, T, size=(n, T))
    new_block = rng.random((n, T)) < 1.0 / block
    new_block[:, 0] = True
    t = np.arange(T)
    last_start = np.maximum.accumulate(np.where(new_block, t, 0), axis=1)
    rows = np.arange(n)[:, None]
    return (starts[rows, last_start] + (t - last_start)) % T


def p_block(x, n: int = N_BOOT, seed: int = SEED, block: int | None = None) -> float:
    """One-sided stationary-bootstrap p of mean > 0, studentized:
    p = (1 + #{t*_b >= t_obs}) / (n + 1), t = Newey-West t of the mean, t*_b from resamples of the
    demeaned series. Studentizing keeps the size at 5-7% (the raw-mean version over-rejects:
    7.6% on white noise and 9.4% on AR(1) phi=0.5 with T=144)."""
    x = np.asarray(pd.Series(x).dropna(), dtype=float)
    T = len(x)
    if T < 8 or x.std() == 0:
        return 1.0
    x0 = x - x.mean()
    idx = stationary_indices(T, n, block or block_len(x), np.random.default_rng(seed))
    t_star = nw_t(x0[idx].T)
    return float((1 + np.sum(t_star >= nw_t(x))) / (n + 1))


def draw_shift(T: int, rng: np.random.Generator) -> int:
    return int(rng.integers(MIN_SHIFT, max(MIN_SHIFT, T - MIN_SHIFT) + 1))


def circular_shift(p: pd.Series, k: int) -> pd.Series:
    return pd.Series(np.roll(p.values, k), index=p.index)
