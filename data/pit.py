"""M2.3 Point-in-time loader.

    as_of("UNRATE", "2008-11-30")   -> values published on or before that date, indexed by `date`
    panel()                          -> full wide month-end panel of all C5 ids + RF
    panel("2022-06-30")              -> same, but every cell whose `published` > t is NaN and
                                        rows after t are dropped ("as known on t")

RF = DGS3MO / 1200 (monthly decimal), shifted one month (the rate known at t-1 applies to month t).
Prices stay as levels; returns are computed downstream.

Monthly macro series have published = next month-end, so at t the value dated t is missing
and the value dated t-1 is present (one-month release lag).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from config import DATA_START, DATA_END, FREQ
from catalog import CATALOG, ids as catalog_ids, primary_source
from data import store
from data.store import as_ts


@lru_cache(maxsize=None)
def _long(series_id: str, source: str | None = None, directory: str | None = None) -> pd.DataFrame | None:
    src = source or primary_source(series_id)
    return store.read_cache(src, series_id, None if directory is None else store.Path(directory))


@lru_cache(maxsize=4)
def _wide(directory: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(values, published) wide frames on the full month-end index, one column per id."""
    idx = pd.date_range(DATA_START, DATA_END, freq=FREQ)
    vals, pubs = {}, {}
    for sid in catalog_ids():
        df = _long(sid, None, directory)
        if df is None:
            vals[sid] = pd.Series(np.nan, index=idx)
            pubs[sid] = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
            continue
        d = df.drop_duplicates("date", keep="last").set_index("date")
        vals[sid] = d["value"].reindex(idx)
        pubs[sid] = d["published"].reindex(idx)
    V = pd.DataFrame(vals, index=idx)
    P = pd.DataFrame(pubs, index=idx)
    return V, P


def clear_cache() -> None:
    _long.cache_clear()
    _wide.cache_clear()


def as_of(series_id: str, t) -> pd.Series:
    """Values of `series_id` with published <= t, indexed by date (month-end)."""
    df = _long(series_id, None, str(store.cache_dir()))
    if df is None:
        return pd.Series(dtype=float, name=series_id)
    t = as_ts(t)
    d = df[df["published"] <= t]
    return pd.Series(d["value"].values, index=pd.DatetimeIndex(d["date"].values), name=series_id)


def published_frame(t=None, ids: list[str] | None = None) -> pd.DataFrame:
    """Wide frame of `published` timestamps aligned to panel(t) (for audits/tests)."""
    _, P = _wide(str(store.cache_dir()))
    cols = ids or list(P.columns)
    P = P[cols]
    if t is not None:
        P = P.loc[: as_ts(t)]
    return P.copy()


def panel(t=None, ids: list[str] | None = None) -> pd.DataFrame:
    """Wide month-end panel of C5 ids + RF, cut to published <= t when t is given."""
    V, P = _wide(str(store.cache_dir()))
    if t is not None:
        tt = as_ts(t)
        known = P.notna() & (P <= tt)
        V = V.where(known).loc[:tt]
    out = V.copy()
    out["RF"] = (out["DGS3MO"] / 1200.0).shift(1)
    if ids is not None:
        keep = [c for c in ids if c in out.columns]
        if "RF" not in keep:
            keep.append("RF")
        out = out[keep]
    out.index.name = "date"
    return out


def vendor_series(series_id: str, source: str, t=None) -> pd.Series | None:
    """Month-end price series from a specific vendor (yahoo/stooq), published <= t. None if absent."""
    df = _long(series_id, source, str(store.cache_dir()))
    if df is None:
        return None
    if t is not None:
        df = df[df["published"] <= as_ts(t)]
    return pd.Series(df["value"].values, index=pd.DatetimeIndex(df["date"].values), name=f"{source}:{series_id}")
def decision_panel() -> pd.DataFrame:
    """Values aligned to their first usable decision month for M1 backtests."""
    V, P = _wide(str(store.cache_dir()))
    out = V.copy()
    for column in V:
        events = pd.DataFrame({"value": V[column], "published": P[column]}).dropna()
        if events.empty:
            continue
        first_usable = pd.DatetimeIndex(events["published"]) + pd.offsets.MonthEnd(0)
        first_usable = first_usable.where(first_usable >= events.index, events.index)
        # If several releases land together, keep the latest observation date.
        released = events["value"].groupby(first_usable).last()
        out[column] = released.reindex(out.index)
    out["RF"] = (out["DGS3MO"] / 1200.0).shift(1)
    out.index.name = "date"
    return out
