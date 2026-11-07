"""M2.7 Lower one-sided CUSUM (Page 1954) on a rule's strength series.

z_t = (s_t - mu0) / sd0
C_0 = 0,  C_t = max(0, C_{t-1} - z_t - k);  alarm_t = C_t > h
A missing s_t leaves C unchanged. C keeps evolving after an alarm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import CUSUM_K


def run_cusum(s: pd.Series, mu0: float, sd0: float, h: float, k: float = CUSUM_K) -> pd.DataFrame:
    z = ((s - mu0) / sd0).to_numpy(dtype=float)
    C = np.zeros(len(z))
    c = 0.0
    for i, zi in enumerate(z):
        if np.isfinite(zi):
            c = max(0.0, c - zi - k)
        C[i] = c
    out = pd.DataFrame({"C": C, "alarm": C > h, "z": z}, index=s.index)
    return out


def cusum_paths(Z: np.ndarray, k: float = CUSUM_K) -> np.ndarray:
    """Vectorised over simulations. Z shape (S, T) -> C shape (S, T)."""
    Z = np.atleast_2d(Z)
    C = np.zeros_like(Z, dtype=float)
    c = np.zeros(Z.shape[0])
    for j in range(Z.shape[1]):
        zj = np.nan_to_num(Z[:, j], nan=-k)   # NaN -> no change
        c = np.maximum(0.0, c - zj - k)
        C[:, j] = c
    return C


def onsets(alarm) -> np.ndarray:
    """Boolean array marking the first month of each alarm episode (rising edges)."""
    a = np.asarray(alarm, dtype=bool)
    prev = np.zeros_like(a)
    prev[..., 1:] = a[..., :-1]
    return a & ~prev


def n_episodes(alarm) -> np.ndarray:
    return onsets(alarm).sum(axis=-1)
