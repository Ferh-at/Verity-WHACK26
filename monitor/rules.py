"""M2.6 Monitor rules: strength series, strength with bootstrap range, integrity.

Every alarm rule is a pair of monthly inputs (a, b) and a window statistic; higher = rule working.
  negcorr: s = -corr_W(a, b)
  negbeta: s = -slope of a on b over W months  (gold return per pp change in real yield)
curve_recession is display only: share of inversion spells followed by a known recession start
within 24 months.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from config import ROLL_WIN, ROLL_WIN_BTC, STRENGTH_BOOT, STRENGTH_BLOCK, SEED
from data import pit
from data.store import as_ts, month_end
from data.vendor import vendor_check, worst
from data.nber import recession_starts_known
from monitor.boot import stationary_indices

RULES: dict[str, dict] = {
    "bonds_protect":   dict(statement="Bonds protect stocks", a=("SPX", "ret"), b=("TLT", "ret"),
                            stat="negcorr", window=ROLL_WIN, data_start="2005-07-31", alarm=True),
    "dollar_haven":    dict(statement="The dollar is a safe haven", a=("SPX", "ret"), b=("DXY", "ret"),
                            stat="negcorr", window=ROLL_WIN, data_start="2003-01-31", alarm=True),
    "gold_realyield":  dict(statement="Gold rises when real yields fall", a=("GOLD", "ret"), b=("DFII10", "diff"),
                            stat="negbeta", window=ROLL_WIN, data_start="2006-01-31", alarm=True),
    "btc_diversifier": dict(statement="Bitcoin doesn't move with stocks", a=("BTC", "ret"), b=("NASDAQ", "ret"),
                            stat="negcorr", window=ROLL_WIN_BTC, data_start="2016-09-30", alarm=True),
    "curve_recession": dict(statement="Curve inversion means recession", a=("T10Y3M", "level"), b=None,
                            stat="spells", window=24, data_start="2003-01-31", alarm=False),
    "vix_control":     dict(statement="Fear spikes when stocks fall (control)", a=("SPX", "ret"), b=("VIX", "dlog"),
                            stat="negcorr", window=ROLL_WIN, data_start="2003-01-31", alarm=True, control=True),
}
ALARM_RULES = [k for k, v in RULES.items() if v["alarm"]]
TRADABLE = {"SPX", "NASDAQ", "TLT", "GOLD", "DXY", "BTC"}


def _transform(x: pd.Series, kind: str) -> pd.Series:
    if kind == "ret":
        return x.pct_change(fill_method=None)
    if kind == "diff":
        return x.diff()
    if kind == "dlog":
        return np.log(x).diff()
    return x


def pairs(rule_id: str, panel: pd.DataFrame) -> pd.DataFrame:
    r = RULES[rule_id]
    out = pd.DataFrame({"a": _transform(panel[r["a"][0]], r["a"][1])})
    if r["b"] is not None:
        out["b"] = _transform(panel[r["b"][0]], r["b"][1])
    return out


def window_stat(A: np.ndarray, B: np.ndarray, stat: str) -> np.ndarray:
    """Statistic over the last axis. A, B shape (..., W)."""
    a = A - A.mean(axis=-1, keepdims=True)
    b = B - B.mean(axis=-1, keepdims=True)
    sab = (a * b).sum(axis=-1)
    sbb = (b * b).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "negcorr":
            return -sab / np.sqrt((a * a).sum(axis=-1) * sbb)
        if stat == "negbeta":
            return -sab / sbb
    raise ValueError(stat)


def rolling_stat(A: np.ndarray, B: np.ndarray, W: int, stat: str) -> np.ndarray:
    """Vectorised rolling statistic. A, B shape (S, T) -> (S, T-W+1); window j ends at row j+W-1."""
    A, B = np.atleast_2d(A).astype(float), np.atleast_2d(B).astype(float)

    def cs(x):
        c = np.cumsum(x, axis=1)
        c = np.concatenate([np.zeros((x.shape[0], 1)), c], axis=1)
        return c[:, W:] - c[:, :-W]

    Sa, Sb, Saa, Sbb, Sab = cs(A), cs(B), cs(A * A), cs(B * B), cs(A * B)
    cab = Sab - Sa * Sb / W
    vaa = Saa - Sa * Sa / W
    vbb = Sbb - Sb * Sb / W
    with np.errstate(invalid="ignore", divide="ignore"):
        if stat == "negcorr":
            return -cab / np.sqrt(vaa * vbb)
        if stat == "negbeta":
            return -cab / vbb
    raise ValueError(stat)


def strength_series(rule_id: str, panel: pd.DataFrame) -> pd.Series:
    """s_t for every month whose W-month window is complete (and on/after the rule's data start)."""
    r = RULES[rule_id]
    if r["stat"] == "spells":
        raise ValueError("curve_recession has no rolling strength series; use strength()")
    P, W = pairs(rule_id, panel), r["window"]
    a, b = P["a"], P["b"]
    ok = (a.notna() & b.notna()).astype(float).rolling(W, min_periods=W).sum() == W
    if r["stat"] == "negcorr":
        s = -a.rolling(W, min_periods=W).corr(b)
    else:
        s = -a.rolling(W, min_periods=W).cov(b) / b.rolling(W, min_periods=W).var()
    s = s.where(ok)
    s = s[s.index >= as_ts(r["data_start"])]
    return s.rename(rule_id)


def _wilson(k: int, n: int, z: float = 1.6448536) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def curve_spells(as_of, panel: pd.DataFrame | None = None) -> list[dict]:
    t = as_ts(as_of)
    p = panel if panel is not None else pit.panel(t)
    x = p["T10Y3M"].loc[:t].dropna()
    inv = x < 0
    starts = inv & ~inv.shift(1, fill_value=False)
    rec = recession_starts_known(t)
    out = []
    for d in x.index[starts.values]:
        horizon = d + pd.offsets.MonthEnd(24)
        eligible = horizon <= month_end(t)
        hit = any(d < r <= horizon for r in rec)
        out.append({"start": str(d.date()), "eligible": bool(eligible), "recession_within_24m": bool(hit)})
    return out


def strength(rule_id: str, as_of, panel: pd.DataFrame | None = None, n_boot: int = STRENGTH_BOOT,
             seed: int | None = None) -> dict:
    t = as_ts(as_of)
    p = panel if panel is not None else pit.panel(t)
    r = RULES[rule_id]
    if r["stat"] == "spells":
        sp = curve_spells(t, p)
        el = [s for s in sp if s["eligible"]]
        k = sum(s["recession_within_24m"] for s in el)
        lo, hi = _wilson(k, len(el))
        est = k / len(el) if el else float("nan")
        return {"est": _clean(est), "lo": _clean(lo), "hi": _clean(hi), "n_spells": len(el), "hits": k}
    P = pairs(rule_id, p).loc[:t]
    W = r["window"]
    last = P.iloc[-W:]
    if len(last) < W or last.isna().any().any() or t < as_ts(r["data_start"]):
        return {"est": None, "lo": None, "hi": None}
    A, B = last["a"].to_numpy(), last["b"].to_numpy()
    est = float(window_stat(A, B, r["stat"]))
    rng = np.random.default_rng(SEED + (seed if seed is not None else t.year * 12 + t.month))
    idx = stationary_indices(W, STRENGTH_BLOCK, n_boot, rng)
    bs = window_stat(A[idx], B[idx], r["stat"])
    bs = bs[np.isfinite(bs)]
    lo, hi = (np.percentile(bs, [5, 95]) if len(bs) else (np.nan, np.nan))
    return {"est": _clean(est), "lo": _clean(lo), "hi": _clean(hi)}


def _clean(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 6)


def rule_assets(rule_id: str) -> list[str]:
    r = RULES[rule_id]
    return [x[0] for x in (r["a"], r["b"]) if x is not None and x[0] in TRADABLE]


def integrity(rule_id: str, as_of) -> dict:
    W = RULES[rule_id]["window"]
    res = [vendor_check(a, as_of, window=W) for a in rule_assets(rule_id)]
    w = worst(res)
    return {"vendors_agree": w["vendors_agree"], "max_gap_pct": w["max_gap_pct"], "status": w["status"]}
