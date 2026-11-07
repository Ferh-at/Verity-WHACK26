"""M2.1 Fetch and cache every catalog series.

    python -m data.fetch            # download everything to data/cache/
    python -m data.fetch --check    # print id, first, last, n, NaN table
    OFFLINE=1 python -m data.fetch  # never touches the network (just --check)

Vendors:
  yahoo  chart API adjusted close (yfinance fallback) -> month-end last
  stooq  pandas_datareader 'stooq' daily Close (fallback: direct CSV URL) -> month-end last
  fred   fredapi with FRED_API_KEY (fallback: keyless fredgraph.csv)
         daily series: ffill <= 5 business days, then month-end last
         monthly series: moved to month-end, published = next month-end [TA]
  nber   hard-coded table (data/nber.py); date = event month-end, value = +1 peak / -1 trough,
         published = announcement date
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_START, DATA_END, FREQ  # noqa: E402
from catalog import CATALOG  # noqa: E402
from data import store  # noqa: E402

log = logging.getLogger("whack.fetch")

try:
    from dotenv import load_dotenv
    load_dotenv(store.ROOT / ".env")
except Exception:
    pass


def offline() -> bool:
    return os.environ.get("OFFLINE", "0") == "1"


# ---------------------------------------------------------------- helpers
def _to_month_end(daily: pd.Series, ffill_limit: int | None = None) -> pd.Series:
    s = daily.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    s = pd.to_numeric(s, errors="coerce").sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if ffill_limit:
        bidx = pd.bdate_range(s.index.min(), s.index.max())
        s = s.reindex(s.index.union(bidx)).ffill(limit=ffill_limit)
    return s.resample(FREQ).last()     # NaN-skipping last; empty months stay NaN


def _frame(m: pd.Series, monthly_lag: bool) -> pd.DataFrame:
    df = pd.DataFrame({"date": m.index, "value": m.values})
    if monthly_lag:
        df["published"] = df["date"] + pd.offsets.MonthEnd(1)
    else:
        df["published"] = df["date"]
    return df


def _retry(fn, what: str, tries: int = 3, backoff: float = 2.0):
    err = None
    for i in range(tries):
        try:
            out = fn()
            if out is None or len(out) == 0:
                raise ValueError("empty result")
            return out
        except Exception as e:  # network, parse, empty
            err = e
            log.info("%s attempt %d failed: %s", what, i + 1, e)
            time.sleep(backoff)
    raise RuntimeError(f"{what}: {err}")


# ---------------------------------------------------------------- vendors
def _yahoo_chart(symbol: str, start: str, end: str) -> pd.Series:
    """Adjusted daily close from Yahoo's chart endpoint.

    The pinned yfinance client is rejected by Yahoo (empty body, then
    'no timezone found'). A browser-style request to the same vendor still
    returns the adjusted series the cache is specified to store.
    """
    import json
    import urllib.parse
    import urllib.request
    start_ts = int(pd.Timestamp(start).tz_localize("UTC").timestamp())
    end_ts = int((pd.Timestamp(end) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp())
    quoted = urllib.parse.quote(symbol, safe="")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{quoted}?period1={start_ts}&period2={end_ts}&interval=1d&includeAdjustedClose=true"
    )
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode())
    result = (payload.get("chart") or {}).get("result") or []
    if not result:
        raise ValueError((payload.get("chart") or {}).get("error") or "empty chart")
    block = result[0]
    stamps = block.get("timestamp") or []
    indicators = block.get("indicators") or {}
    adjusted = (indicators.get("adjclose") or [{}])[0].get("adjclose")
    close = (indicators.get("quote") or [{}])[0].get("close")
    values = adjusted if adjusted and any(v is not None for v in adjusted) else close
    if not stamps or not values:
        raise ValueError("empty chart")
    index = pd.to_datetime(stamps, unit="s", utc=True).tz_convert(None).normalize()
    return pd.Series(values, index=index, dtype=float).dropna()


def fetch_yahoo(symbol: str, start: str, end: str) -> pd.Series:
    try:
        return _yahoo_chart(symbol, start, end)
    except Exception as e:
        log.info("yahoo chart failed for %s (%s); trying yfinance", symbol, e)
    import yfinance as yf
    end_plus = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    df = yf.download(symbol, start=start, end=end_plus, interval="1d", auto_adjust=True, progress=False)
    close = df["Close"]
    if isinstance(close, pd.DataFrame):      # newer yfinance returns MultiIndex columns
        close = close.iloc[:, 0]
    return close.dropna()


def fetch_stooq(symbol: str, start: str, end: str) -> pd.Series:
    import socket
    previous = socket.getdefaulttimeout()
    socket.setdefaulttimeout(10)
    try:
        try:
            import pandas_datareader.data as web
            df = web.DataReader(symbol, "stooq", start, end)
            return df["Close"].sort_index()
        except Exception as e:
            log.info("pandas_datareader stooq failed for %s (%s); trying CSV URL", symbol, e)
            import urllib.request
            url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
            raw = urllib.request.urlopen(url, timeout=10).read().decode()
            df = pd.read_csv(io.StringIO(raw), parse_dates=["Date"], index_col="Date")
            return df["Close"].sort_index().loc[start:end]
    finally:
        socket.setdefaulttimeout(previous)


def fetch_fred(series_id: str, start: str, end: str) -> pd.Series:
    key = os.environ.get("FRED_API_KEY")
    if key:
        from fredapi import Fred
        return Fred(api_key=key).get_series(series_id, observation_start=start, observation_end=end)
    import urllib.request
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start}&coed={end}"
    raw = urllib.request.urlopen(url, timeout=30).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = ["date", "value"]
    s = pd.to_numeric(df["value"], errors="coerce")  # FRED uses "." for missing
    s.index = pd.to_datetime(df["date"])
    return s


def nber_frame() -> pd.DataFrame:
    from data.nber import NBER_TABLE
    rows = [{"date": store.month_end(m + "-01"), "value": 1.0 if ev == "peak" else -1.0,
             "published": pd.Timestamp(ann)} for ev, m, ann in NBER_TABLE]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main
def fetch_one(series_id: str, source: str, symbol: str, start: str, end: str) -> pd.DataFrame:
    meta = CATALOG[series_id]
    if source == "yahoo":
        m = _to_month_end(_retry(lambda: fetch_yahoo(symbol, start, end), f"yahoo {symbol}"))
    elif source == "stooq":
        m = _to_month_end(_retry(lambda: fetch_stooq(symbol, start, end), f"stooq {symbol}", tries=1))
    elif source == "fred":
        raw = _retry(lambda: fetch_fred(symbol, start, end), f"fred {symbol}")
        if meta["freq"] == "monthly":
            raw.index = pd.to_datetime(raw.index)
            m = _to_month_end(raw)
        else:
            m = _to_month_end(raw, ffill_limit=5)
    else:
        raise ValueError(source)
    m = m.loc[store.month_end(start):store.month_end(end)]
    # trim leading NaN months (series not yet started); internal gaps stay NaN
    first = m.first_valid_index()
    m = m.loc[first:] if first is not None else m
    return _frame(m, monthly_lag=(meta["freq"] == "monthly"))


def fetch_all(start: str = DATA_START, end: str = DATA_END) -> dict:
    """Write data/cache/{source}_{id}.parquet (+ .csv) for every catalog entry and vendor."""
    report = {}
    if offline():
        log.warning("OFFLINE=1: nothing fetched; using existing cache")
        return report
    for sid, meta in CATALOG.items():
        for source, symbol in meta["sources"].items():
            key = f"{source}_{sid}"
            try:
                df = nber_frame() if source == "nber" else fetch_one(sid, source, symbol, start, end)
                store.write_cache(df, source, sid, store.DATA_CACHE)
                report[key] = f"ok ({len(df)} rows)"
            except Exception as e:
                existing = store.read_cache(source, sid, store.DATA_CACHE)
                if existing is not None:
                    report[key] = f"FAILED, kept existing cache ({len(existing)} rows): {e}"
                else:
                    report[key] = f"FAILED, no cache: {e}"
                log.warning("%s: %s", key, report[key])
    return report


def check_table(directory: Path | None = None) -> pd.DataFrame:
    rows = []
    d = directory or store.cache_dir()
    for sid, meta in CATALOG.items():
        for source in meta["sources"]:
            df = store.read_cache(source, sid, d)
            if df is None:
                rows.append(dict(id=sid, source=source, first=None, last=None, n=0, nan=None, status="MISSING"))
                continue
            v = df.dropna(subset=["value"])
            rows.append(dict(id=sid, source=source,
                             first=v["date"].min().date() if len(v) else None,
                             last=v["date"].max().date() if len(v) else None,
                             n=int(len(v)), nan=int(df["value"].isna().sum()), status="ok"))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print the cache table only")
    ap.add_argument("--start", default=DATA_START)
    ap.add_argument("--end", default=DATA_END)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not a.check:
        for k, v in fetch_all(a.start, a.end).items():
            print(f"{k:22s} {v}")
    print(f"cache dir: {store.cache_dir()}" + ("   (SYNTHETIC MOCK DATA)" if store.using_mock() else ""))
    with pd.option_context("display.width", 140, "display.max_rows", 100):
        print(check_table().to_string(index=False))


if __name__ == "__main__":
    main()
