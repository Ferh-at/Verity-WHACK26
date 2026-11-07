"""Cache I/O shared by fetch, pit and vendor.

File format (C4): data/cache/{source}_{id}.parquet with columns
  date (month-end, tz-naive), value (float), published (date the value became knowable)
plus a CSV mirror with the same stem. Readers try parquet, then CSV.

Cache directory resolution:
  1. $WHACK_CACHE_DIR if set
  2. data/cache/ if it holds at least one file
  3. mocks/cache/ (SYNTHETIC data, a loud warning is logged once)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = ROOT / "data" / "cache"
MOCK_CACHE = ROOT / "mocks" / "cache"
log = logging.getLogger("whack.data")
_warned = False


def cache_dir() -> Path:
    global _warned
    env = os.environ.get("WHACK_CACHE_DIR")
    if env:
        return Path(env)
    if DATA_CACHE.exists() and (any(DATA_CACHE.glob("*.parquet")) or any(DATA_CACHE.glob("*.csv"))):
        return DATA_CACHE
    if not _warned:
        log.warning("data/cache is empty -> using SYNTHETIC mocks/cache. Run `python -m data.fetch`.")
        _warned = True
    return MOCK_CACHE


def using_mock() -> bool:
    return cache_dir().resolve() == MOCK_CACHE.resolve()


def cache_source() -> str:
    """Identify the configured input cache for generated monitor artefacts."""
    directory = cache_dir().resolve()
    if directory == MOCK_CACHE.resolve():
        return "synthetic_cache"
    return "market_cache" if directory == DATA_CACHE.resolve() else "external_cache"


def cache_path(source: str, series_id: str, directory: Path | None = None) -> Path:
    return (directory or cache_dir()) / f"{source}_{series_id}.parquet"


def write_cache(df: pd.DataFrame, source: str, series_id: str, directory: Path | None = None) -> Path:
    d = directory or DATA_CACHE
    d.mkdir(parents=True, exist_ok=True)
    out = df[["date", "value", "published"]].copy()
    out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
    out["published"] = pd.to_datetime(out["published"]).astype("datetime64[ns]")
    out["value"] = out["value"].astype(float)
    p = cache_path(source, series_id, d)
    out.to_parquet(p, index=False)
    out.to_csv(p.with_suffix(".csv"), index=False, date_format="%Y-%m-%d")
    return p


def read_cache(source: str, series_id: str, directory: Path | None = None) -> pd.DataFrame | None:
    p = cache_path(source, series_id, directory)
    df = None
    if p.exists():
        try:
            df = pd.read_parquet(p)
        except Exception as e:  # corrupt parquet -> try CSV mirror
            log.warning("parquet read failed for %s (%s); trying CSV", p.name, e)
    if df is None and p.with_suffix(".csv").exists():
        df = pd.read_csv(p.with_suffix(".csv"))
    if df is None:
        return None
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df["published"] = pd.to_datetime(df["published"]).astype("datetime64[ns]")
    df["value"] = df["value"].astype(float)
    return df.sort_values("date").reset_index(drop=True)


def month_end(t) -> pd.Timestamp:
    """Any date-like -> the month-end Timestamp of its month (tz-naive)."""
    ts = pd.Timestamp(t)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return (ts.normalize() + pd.offsets.MonthEnd(0))


def as_ts(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize(None) if ts.tzinfo is not None else ts
