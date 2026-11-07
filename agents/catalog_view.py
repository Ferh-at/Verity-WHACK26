"""C5 catalog view for the standalone M3 package and the team catalog."""
from importlib import import_module

# This view is needed before Member 2's catalog module is merged.
FALLBACK = {
    "SPX": ("price", "S&P 500 index"),
    "NASDAQ": ("price", "Nasdaq Composite"),
    "TLT": ("price", "Long-duration US Treasury ETF"),
    "GOLD": ("price", "Gold"),
    "DXY": ("price", "US dollar index"),
    "BTC": ("price", "Bitcoin in USD"),
    "VIX": ("level", "Equity implied volatility"),
    "DGS10": ("level", "10y Treasury yield, percent"),
    "DGS2": ("level", "2y Treasury yield, percent"),
    "DGS3MO": ("level", "3m Treasury yield, percent"),
    "T10Y3M": ("level", "10y minus 3m Treasury spread, percentage points"),
    "T10Y2Y": ("level", "10y minus 2y Treasury spread, percentage points"),
    "DFII10": ("level", "10y real Treasury yield, percent"),
    "UNRATE": ("level", "US unemployment rate"),
    "CPIAUCSL": ("level", "US consumer price index"),
    "FEDFUNDS": ("level", "Effective federal funds rate"),
    "NBER": ("events", "US business cycle announcements"),
}


def catalog():
    try:
        module = import_module("catalog")
    except ModuleNotFoundError as exc:
        if exc.name != "catalog":
            raise
        return {
            key: {"type": kind, "tradable": kind == "price", "description": description}
            for key, (kind, description) in FALLBACK.items()
        }
    return module.CATALOG


def field(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)
