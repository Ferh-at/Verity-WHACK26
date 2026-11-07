"""Sharpe, Newey-West t, naive t-test (M1.4)."""
import math

import numpy as np
from scipy import stats as sps


def sharpe(x, annual: bool = True) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return float("nan")
    s = x.mean() / sd
    return float(s * math.sqrt(12)) if annual else float(s)


def nw_lag(T: int) -> int:
    """Newey-West (1994) rule: L = floor(4 * (T/100)^(2/9))."""
    return int(math.floor(4 * (T / 100) ** (2 / 9)))


def nw_t(x, lags: int | None = None):
    """Newey-West t of the mean (Bartlett kernel). x: 1-D (T,) -> float, or 2-D (T, K) -> (K,) array,
    vectorised over columns. NaN-free input expected."""
    a = np.asarray(x, dtype=float)
    one_d = a.ndim == 1
    if one_d:
        a = a[:, None]
    T = a.shape[0]
    if T < 3:
        out = np.full(a.shape[1], np.nan)
        return float(out[0]) if one_d else out
    L = nw_lag(T) if lags is None else lags
    m = a.mean(axis=0)
    u = a - m
    omega = (u * u).sum(axis=0) / T
    for l in range(1, L + 1):
        g = (u[l:] * u[:-l]).sum(axis=0) / T
        omega = omega + 2 * (1 - l / (L + 1)) * g
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / np.sqrt(omega / T)
    return float(t[0]) if one_d else t


def naive_p(x) -> float:
    """One-sided iid t-test p (mean > 0). Naive baseline only."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or x.std() == 0:
        return 1.0
    return float(sps.ttest_1samp(x, 0.0, alternative="greater").pvalue)
