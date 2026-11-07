"""Reference implementation of the M1.2 feature/position table.

M1 owns engine/features.py. This copy exists so M2's audit, vendor test and their unit tests run
before (or without) M1's code. `get_feature_fns()` returns M1's functions when importable,
otherwise these. Keep the formulas identical to M1.2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import ZWIN, ZMIN, ZCLIP

PRICE_IDS = {"SPX", "NASDAQ", "TLT", "GOLD", "DXY", "BTC"}


def _is_price(series_id: str) -> bool:
    return series_id in PRICE_IDS or series_id.startswith("BENCH_A")


def _delta1(x: pd.Series, series_id: str) -> pd.Series:
    return x.pct_change(fill_method=None) if _is_price(series_id) else x.diff()


def feature(spec, panel: pd.DataFrame) -> pd.Series:
    f, L = spec.feature, int(spec.lookback)
    x = panel[spec.series].astype(float)
    if f == "mom":
        out = x / x.shift(L) - 1.0
    elif f == "level":
        out = x.copy()
    elif f == "change":
        out = (np.log(x) - np.log(x.shift(L))) if _is_price(spec.series) else (x - x.shift(L))
    elif f == "zscore":
        r = x.rolling(L, min_periods=L)
        out = (x - r.mean()) / r.std(ddof=1)
    elif f == "corr":
        a = _delta1(x, spec.series)
        b = _delta1(panel[spec.series2].astype(float), spec.series2)
        out = a.rolling(L, min_periods=L).corr(b)
    elif f == "centered_ma":          # LEAKY CONTROL
        out = _delta1(x, spec.series).rolling(L, center=True, min_periods=L).mean()
    elif f == "fullsample_z":         # LEAKY CONTROL
        out = (x - x.mean()) / x.std(ddof=1)
    else:
        raise ValueError(f"unknown feature {f}")
    return out.replace([np.inf, -np.inf], np.nan).rename(f"{f}:{spec.series}")


def position(spec, panel: pd.DataFrame) -> pd.Series:
    f = feature(spec, panel)
    if spec.transform == "sign":
        p = spec.direction * np.sign(f)
    else:
        r = f.rolling(ZWIN, min_periods=ZMIN)
        z = (f - r.mean()) / r.std(ddof=1)
        p = spec.direction * z.clip(-ZCLIP, ZCLIP) / ZCLIP
    p = p.where(f.notna())
    return p.rename("pos")


def get_feature_fns():
    """(feature, position) — M1's engine.features when available, else this reference copy."""
    try:
        from engine import features as ef  # type: ignore
        fe, po = getattr(ef, "feature"), getattr(ef, "position", None)
        return fe, (po or position)
    except Exception:
        return feature, position
