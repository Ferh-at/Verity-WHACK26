"""Luck hurdle: the whole search re-run on null data (White 2000 style).

For replicate b one uniform draw u_b (seed SEED + b) is shared by every hypothesis, giving the
circular shift k_i = MIN_SHIFT + floor(u_b * (T_i - 2*MIN_SHIFT + 1)); with equal history lengths
every hypothesis gets the same k_b, which keeps their cross-correlation."""
import numpy as np
import pandas as pd

from config import COST_BPS, HURDLE_PCT, MIN_SHIFT, N_HURDLE, SEED
from engine.backtest import strategy_returns
from engine.stats import nw_t
from schemas import Hypothesis


def shift_draws(n: int, seed: int) -> np.ndarray:
    return np.array([np.random.default_rng(seed + b).random() for b in range(1, n + 1)])


def shifts_for(T: int, u: np.ndarray) -> np.ndarray:
    span = T - 2 * MIN_SHIFT + 1
    if span >= 1:
        return MIN_SHIFT + np.floor(u * span).astype(int)
    return np.floor(u * T).astype(int)


def null_stats(held: np.ndarray, r: np.ndarray, u: np.ndarray, cost_bps: float = COST_BPS) -> np.ndarray:
    """NW t of net returns for each circular shift of the held position against the returns. (n,)"""
    T = len(held)
    if T < 3:
        return np.full(len(u), np.nan)
    k = shifts_for(T, u)
    idx = (np.arange(T)[None, :] - k[:, None]) % T
    ps = held[idx]
    turnover = np.abs(ps - np.roll(ps, 1, axis=1))
    net = ps * r[None, :] - cost_bps / 1e4 * turnover
    return nw_t(net.T)


def discovery_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(held position, asset return, net) over eligible discovery months."""
    e = df[df["pos"].notna() & df["net"].notna()]
    return e["pos"].to_numpy(float), e["r"].to_numpy(float), e["net"].to_numpy(float)


def hurdle_from(stats: dict[str, np.ndarray], observed: dict[str, float]) -> dict:
    """stats: {hid: (n,) null statistics}. Returns {hurdle, null_max, p_search, observed}."""
    if not stats:
        return {"hurdle": float("nan"), "null_max": np.array([]), "p_search": {}, "observed": {}}
    M = np.nanmax(np.vstack(list(stats.values())), axis=0)
    n = len(M)
    hurdle = float(np.percentile(M, HURDLE_PCT))
    p_search = {h: float((1 + np.sum(M >= s)) / (n + 1)) if np.isfinite(s) else 1.0
                for h, s in observed.items()}
    return {"hurdle": hurdle, "null_max": M, "p_search": p_search, "observed": dict(observed)}


def luck_hurdle(hyps: list[Hypothesis], panel: pd.DataFrame, n: int = N_HURDLE, seed: int = SEED,
                splits: dict | None = None, cache: dict | None = None) -> dict:
    """hyps: cluster representatives of the whole session ledger. `cache` ({spec_hash: (S, null)})
    lets the hurdle be recomputed incrementally as batches are added (must be per panel)."""
    cache = {} if cache is None else cache
    u = shift_draws(n, seed)
    stats, observed = {}, {}
    for h in hyps:
        key = (h.spec_hash, n, seed)
        if key not in cache:
            held, r, net = discovery_arrays(strategy_returns(h.spec, panel, "discovery", splits))
            s = nw_t(net) if len(net) >= 3 else float("nan")
            cache[key] = (s, null_stats(held, r, u))
        observed[h.id], stats[h.id] = cache[key]
    return hurdle_from(stats, observed)
