"""Features and positions (M1.2). Every feature at month-end t uses rows <= t only,
except the two leaky controls (centered_ma, fullsample_z)."""
import numpy as np
import pandas as pd

from catalog import CATALOG
from config import ZCLIP, ZMIN, ZWIN
from schemas import Spec


def is_price(series_id: str) -> bool:
    if series_id in CATALOG:
        return CATALOG[series_id]["type"] == "price"
    return series_id.startswith(("BENCH_A", "BENCH_MKT"))


def delta1(x: pd.Series, series_id: str) -> pd.Series:
    """One-month change: returns for price series, differences for level series."""
    return x.pct_change(fill_method=None) if is_price(series_id) else x.diff()


def feature(spec: Spec, panel: pd.DataFrame) -> pd.Series:
    x = panel[spec.series].astype(float)
    L = spec.lookback
    f = spec.feature
    if f == "mom":
        out = x / x.shift(L) - 1
    elif f == "level":
        out = x.copy()
    elif f == "change":
        out = np.log(x).diff(L) if is_price(spec.series) else x.diff(L)
    elif f == "zscore":
        out = (x - x.rolling(L).mean()) / x.rolling(L).std(ddof=1)
    elif f == "corr":
        d1 = delta1(x, spec.series)
        d2 = delta1(panel[spec.series2].astype(float), spec.series2)
        out = d1.rolling(L).corr(d2)
    elif f == "centered_ma":
        out = delta1(x, spec.series).rolling(L, center=True).mean()
    elif f == "fullsample_z":
        out = (x - x.mean()) / x.std(ddof=1)
    else:
        raise ValueError(f"unknown feature {f}")
    return out.replace([np.inf, -np.inf], np.nan).rename(None)


def to_position(f: pd.Series, transform: str, direction: int) -> pd.Series:
    if transform == "sign":
        p = np.sign(f)
    elif transform == "clip_z":
        m = f.rolling(ZWIN, min_periods=ZMIN).mean()
        s = f.rolling(ZWIN, min_periods=ZMIN).std(ddof=1)
        z = ((f - m) / s).replace([np.inf, -np.inf], np.nan)
        p = z.clip(-ZCLIP, ZCLIP) / ZCLIP
    else:
        raise ValueError(f"unknown transform {transform}")
    return direction * p


def position(spec: Spec, panel: pd.DataFrame) -> pd.Series:
    """Position in [-1, 1] set at month-end t. NaN = ineligible month."""
    return to_position(feature(spec, panel), spec.transform, spec.direction)
