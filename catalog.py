"""Series catalog (Common contract C5). Owner: M2. Read by everyone.

Each entry:
  sources : {vendor: symbol}. The first vendor listed is the PRIMARY one used by pit.panel().
            A second price vendor ("stooq") is used only by the integrity / vendor check.
  type    : "price" | "level" | "events"
  tradable: can be the `asset` of a Spec
  freq    : "daily" (resample to month-end last) | "monthly" (published = next month-end)
  desc    : short human description (fed to the LLM generator by M3)
"""

CATALOG: dict[str, dict] = {
    "SPX":    dict(sources={"yahoo": "^GSPC", "stooq": "^SPX"}, type="price", tradable=True, freq="daily",
                   desc="S&P 500 index level"),
    "NASDAQ": dict(sources={"yahoo": "^IXIC", "stooq": "^NDQ"}, type="price", tradable=True, freq="daily",
                   desc="Nasdaq Composite index level"),
    "TLT":    dict(sources={"yahoo": "TLT", "stooq": "TLT.US"}, type="price", tradable=True, freq="daily",
                   desc="iShares 20+ year Treasury bond ETF (total-return adjusted); starts Jul 2002"),
    "GOLD":   dict(sources={"yahoo": "GC=F", "stooq": "XAUUSD"}, type="price", tradable=True, freq="daily",
                   desc="Gold, front-month future (Yahoo) / spot (Stooq), USD per oz"),
    "DXY":    dict(sources={"yahoo": "DX-Y.NYB"}, type="price", tradable=True, freq="daily",
                   desc="US dollar index (single vendor)"),
    "BTC":    dict(sources={"yahoo": "BTC-USD", "stooq": "BTCUSD"}, type="price", tradable=True, freq="daily",
                   desc="Bitcoin in USD; starts Sep 2014 (short history)"),
    "VIX":    dict(sources={"yahoo": "^VIX"}, type="level", tradable=False, freq="daily",
                   desc="CBOE VIX implied volatility index, points"),
    "DGS10":  dict(sources={"fred": "DGS10"}, type="level", tradable=False, freq="daily",
                   desc="10-year Treasury constant-maturity yield, %"),
    "DGS2":   dict(sources={"fred": "DGS2"}, type="level", tradable=False, freq="daily",
                   desc="2-year Treasury constant-maturity yield, %"),
    "DGS3MO": dict(sources={"fred": "DGS3MO"}, type="level", tradable=False, freq="daily",
                   desc="3-month Treasury bill yield, %"),
    "T10Y3M": dict(sources={"fred": "T10Y3M"}, type="level", tradable=False, freq="daily",
                   desc="10y minus 3m Treasury spread, pp (negative = inverted curve)"),
    "T10Y2Y": dict(sources={"fred": "T10Y2Y"}, type="level", tradable=False, freq="daily",
                   desc="10y minus 2y Treasury spread, pp"),
    "DFII10": dict(sources={"fred": "DFII10"}, type="level", tradable=False, freq="daily",
                   desc="10-year TIPS real yield, %; from Jan 2003"),
    "UNRATE": dict(sources={"fred": "UNRATE"}, type="level", tradable=False, freq="monthly",
                   desc="US unemployment rate, % (known one month later)"),
    "CPIAUCSL": dict(sources={"fred": "CPIAUCSL"}, type="level", tradable=False, freq="monthly",
                   desc="US CPI, all items, index level (known one month later)"),
    "FEDFUNDS": dict(sources={"fred": "FEDFUNDS"}, type="level", tradable=False, freq="monthly",
                   desc="Effective federal funds rate, monthly average % (known one month later)"),
    "NBER":   dict(sources={"nber": "table"}, type="events", tradable=False, freq="events",
                   desc="NBER business-cycle peaks/troughs with announcement dates"),
}

PRICE_VENDORS = ("yahoo", "stooq")


def ids(include_events: bool = False) -> list[str]:
    return [k for k, v in CATALOG.items() if include_events or v["type"] != "events"]


def tradable_ids() -> list[str]:
    return [k for k, v in CATALOG.items() if v["tradable"]]


def primary_source(series_id: str) -> str:
    return next(iter(CATALOG[series_id]["sources"]))


def is_tradable(series_id: str) -> bool:
    if series_id.startswith("BENCH_A"):
        return True
    return bool(CATALOG.get(series_id, {}).get("tradable", False))


def describe(series_id: str) -> str:
    """'T10Y3M: 10y minus 3m Treasury spread, pp' — used for LLM prompts."""
    return f"{series_id}: {CATALOG[series_id]['desc']}"
