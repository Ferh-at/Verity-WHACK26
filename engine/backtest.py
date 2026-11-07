"""Backtest with no look-ahead (M1.3).

Timing: the position p_t set at month-end t (data <= t) earns r_{t+1}.
    gross_{t+1} = p_t * r_{t+1}
    net_{t+1}   = gross_{t+1} - COST_BPS/1e4 * |p_t - p_{t-1}|   (p_{-1} = 0, NaN -> 0 for turnover)
All output frames are indexed by the RETURN date t+1; split slicing uses that date."""
import numpy as np
import pandas as pd

from config import COST_BPS, EMBARGO_MONTHS, POST_CUTOFF_START, SHORT_HISTORY_RULE, SPLITS
from engine.features import position
from schemas import Spec

MIN_DISCOVERY_MONTHS = 60


def default_splits() -> dict:
    s = dict(SPLITS)
    s["post_cutoff"] = (POST_CUTOFF_START, SPLITS["holdout"][1])
    return s


def asset_returns(panel: pd.DataFrame, asset: str) -> pd.Series:
    return panel[asset].astype(float).pct_change(fill_method=None)


def asset_availability(panel: pd.DataFrame, asset: str) -> pd.Series:
    """Monthly presence flags for split sizing without computing sealed returns."""
    present = panel[asset].notna()
    return pd.Series(np.where(present & present.shift(fill_value=False), 1.0, np.nan), index=panel.index)


def net_returns(p: pd.Series, r: pd.Series, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """p and r share one month-end index. Returns DataFrame[gross, net, pos, r] indexed by return date;
    `pos` is the position held over that return month (p_{t} for row t+1), `r` the asset return."""
    p0 = p.fillna(0.0)
    turnover = (p0 - p0.shift(1).fillna(0.0)).abs()
    held = p.shift(1)
    gross = held * r
    net = gross - cost_bps / 1e4 * turnover.shift(1)
    return pd.DataFrame({"gross": gross, "net": net, "pos": held, "r": r})


def split_bounds(split: str, splits: dict | None = None, r: pd.Series | None = None) -> tuple:
    """(start, end) return dates of a split. Assets with < 60 discovery months of returns use
    SHORT_HISTORY_RULE on their own available history (post_cutoff keeps calendar dates)."""
    splits = splits or default_splits()
    start, end = (pd.Timestamp(d) for d in splits[split])
    if split == "post_cutoff":
        hold_start, hold_end = split_bounds("holdout", splits, r)
        return max(start, hold_start), min(end, hold_end)
    if r is None or "discovery" not in splits:
        return start, end
    d0, d1 = (pd.Timestamp(d) for d in splits["discovery"])
    valid = r.dropna()
    if (valid.index.to_series().between(d0, d1)).sum() >= MIN_DISCOVERY_MONTHS:
        return start, end
    last = max(pd.Timestamp(v[1]) for k, v in splits.items() if k != "post_cutoff")
    hist = valid.index[valid.index <= last]
    if len(hist) == 0:
        return start, end
    n = len(hist)
    a = int(round(SHORT_HISTORY_RULE[0] * n))
    b = int(round((SHORT_HISTORY_RULE[0] + SHORT_HISTORY_RULE[1]) * n))
    cuts = {"discovery": (0, a), "validation": (a, b), "holdout": (b, n)}
    i, j = cuts[split]
    if j <= i:
        return hist[0], hist[0] - pd.offsets.MonthEnd(1)
    return hist[i], hist[j - 1]


def slice_split(df: pd.DataFrame | pd.Series, split: str, splits: dict | None = None,
                r: pd.Series | None = None, embargo: int = EMBARGO_MONTHS):
    start, end = split_bounds(split, splits, r)
    out = df.loc[(df.index >= start) & (df.index <= end)]
    return out.iloc[embargo:] if split != "post_cutoff" else out


def strategy_returns(spec: Spec, panel: pd.DataFrame, split: str | None = None,
                     splits: dict | None = None, pos: pd.Series | None = None) -> pd.DataFrame:
    """DataFrame[gross, net, pos] for one split (or the full sample if split is None).
    Rows with NaN `pos` are ineligible months (they count against coverage)."""
    if split is None:
        p = position(spec, panel) if pos is None else pos
        return net_returns(p, asset_returns(panel, spec.asset))
    availability = asset_availability(panel, spec.asset)
    start, end = split_bounds(split, splits, availability)
    prefix = panel.loc[:end]
    if prefix.empty:
        return pd.DataFrame(columns=["gross", "net", "pos", "r"], index=prefix.index, dtype=float)
    p = position(spec, prefix) if pos is None else pos.reindex(prefix.index)
    df = net_returns(p, asset_returns(prefix, spec.asset))
    eligible = df.loc[start:end]
    if split == "post_cutoff":
        hold_start, hold_end = split_bounds("holdout", splits, availability)
        hold_dates = prefix.loc[hold_start:hold_end].index[EMBARGO_MONTHS:]
        return eligible.loc[eligible.index.intersection(hold_dates)]
    return eligible.iloc[EMBARGO_MONTHS:]


def eligible_net(df: pd.DataFrame) -> pd.Series:
    return df["net"][df["pos"].notna()].dropna()
