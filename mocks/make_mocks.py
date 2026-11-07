"""Build SYNTHETIC mock data so every member can work before real data is cached.

    python -m mocks.make_mocks

Writes:
  mocks/cache/{source}_{id}.parquet/.csv   same format as data/cache (synthetic, NOT real prices)
  mocks/m2_panel.parquet                      wide panel of all C5 ids + RF (= data.pit.panel() on mocks)
  mocks/sample_m2_frames.jsonl                bonds_protect frames 2015-01..2023-12, fake numbers, alarm 2022-06

The synthetic world is shaped like the real one (stock-bond correlation negative until 2021,
positive from 2022; curve inversions in 2000, 2006-07, 2019, 2022-24) so demos look right,
but every number is made up.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import DATA_START, DATA_END, SEED  # noqa: E402
from catalog import CATALOG  # noqa: E402
from data import store  # noqa: E402
from data.fetch import nber_frame  # noqa: E402
from schemas import ReplayFrame  # noqa: E402

MOCK = ROOT / "mocks"


def _yr(idx):
    return idx.year + (idx.month - 0.5) / 12


def build(seed: int = SEED) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(DATA_START, DATA_END, freq="ME")
    T, y = len(idx), _yr(idx)
    e = lambda: rng.standard_normal(T)  # noqa: E731
    r_spx = 0.006 + 0.043 * e()
    # stocks-bonds: negative until 2021, positive from 2022
    rho = np.where(y < 2022.0, -0.40, 0.55)
    z_tlt = rho * (r_spx - 0.006) / 0.043 + np.sqrt(1 - rho ** 2) * e()
    r_tlt = 0.004 + 0.035 * z_tlt
    r_dxy = 0.000 + 0.020 * (-0.3 * (r_spx - 0.006) / 0.043 + np.sqrt(1 - 0.09) * e())
    r_nas = 0.007 + 1.2 * (r_spx - 0.006) + 0.025 * e()
    dlog_vix = -4.0 * (r_spx - 0.006) + 0.12 * e()
    # rates (interpolated anchors + noise)
    a3 = [(2000, 5.8), (2001, 5.5), (2002, 1.7), (2003.5, .9), (2004.5, 1.3), (2006.5, 5.0), (2007.5, 4.8),
          (2008.5, 1.5), (2009, .1), (2015.9, .1), (2016.9, .5), (2018.9, 2.4), (2019.6, 2.0), (2020.3, .1),
          (2021.9, .05), (2022.9, 4.3), (2023.6, 5.4), (2024.6, 5.3), (2025.5, 4.3), (2026.9, 3.8)]
    ac = [(2000, .5), (2000.6, -.5), (2001.2, .3), (2002, 3.5), (2004, 3.5), (2005.5, 1.0), (2006.5, -.4),
          (2007.4, -.2), (2007.9, 1.0), (2009, 3.0), (2014, 2.5), (2018, 1.0), (2019.4, -.2), (2019.8, .2),
          (2020.3, .6), (2021, 1.5), (2022.5, .5), (2022.85, -.5), (2023.5, -1.5), (2024.5, -1.3),
          (2024.95, .2), (2026.9, .5)]
    ip = lambda a: np.interp(y, [p[0] for p in a], [p[1] for p in a])  # noqa: E731
    dgs3 = np.maximum(0.01, ip(a3) + 0.05 * e())
    curve = ip(ac) + 0.08 * e()
    dgs10 = dgs3 + curve
    dgs2 = dgs3 + 0.45 * curve + 0.05 * e()
    real = dgs10 - 2.2 + np.cumsum(0.03 * e()) * 0.3
    d_real = np.diff(real, prepend=real[0])
    r_gold = 0.006 - 0.08 * d_real / 0.25 * 0.045 + 0.04 * e()
    lev = lambda r, p0: p0 * np.cumprod(1 + r)  # noqa: E731
    s = {"SPX": lev(r_spx, 1400), "TLT": lev(r_tlt, 80), "DXY": lev(r_dxy, 105), "NASDAQ": lev(r_nas, 4000),
         "GOLD": lev(r_gold, 290), "VIX": 20 * np.exp(np.cumsum(dlog_vix) - np.cumsum(dlog_vix).mean() * 0),
         "DGS3MO": dgs3, "DGS10": dgs10, "DGS2": dgs2, "T10Y3M": dgs10 - dgs3, "T10Y2Y": dgs10 - dgs2,
         "DFII10": real}
    s["VIX"] = np.clip(s["VIX"] / np.median(s["VIX"]) * 18, 9, 80)
    # BTC from Sep 2014; uncorrelated with Nasdaq until 2020, correlated after
    rb = np.where(y < 2020, 0.0, 0.5)
    zb = rb * (r_nas - 0.007) / np.std(r_nas) + np.sqrt(1 - rb ** 2) * e()
    btc = lev(0.04 + 0.20 * zb, 1.0)
    btc = np.where(idx >= pd.Timestamp("2014-09-30"), btc / btc[idx.get_loc(pd.Timestamp("2014-09-30"))] * 390, np.nan)
    s["BTC"] = btc
    s["UNRATE"] = np.clip(4 + 2 * np.sin((y - 2000) / 4) + 0.1 * e(), 3, 10)
    s["CPIAUCSL"] = 170 * np.cumprod(1 + 0.002 + 0.002 * e())
    s["FEDFUNDS"] = np.maximum(0.05, dgs3 + 0.1)
    out = {k: pd.Series(v, index=idx, name=k) for k, v in s.items()}
    out["TLT"] = out["TLT"].where(idx >= pd.Timestamp("2002-07-31"))
    out["DFII10"] = out["DFII10"].where(idx >= pd.Timestamp("2003-01-31"))
    return out


def write_cache(series: dict[str, pd.Series], seed: int = SEED) -> None:
    rng = np.random.default_rng(seed + 1)
    d = MOCK / "cache"
    d.mkdir(parents=True, exist_ok=True)
    for sid, meta in CATALOG.items():
        for source in meta["sources"]:
            if source == "nber":
                store.write_cache(nber_frame(), "nber", sid, d)
                continue
            x = series[sid].dropna()
            if source == "stooq":   # second vendor: same prices, tiny independent noise
                x = x * (1 + 0.0005 * rng.standard_normal(len(x)))
            lag = meta["freq"] == "monthly"
            df = pd.DataFrame({"date": x.index, "value": x.values,
                               "published": x.index + pd.offsets.MonthEnd(1) if lag else x.index})
            store.write_cache(df, source, sid, d)


def sample_frames() -> list[ReplayFrame]:
    rng = np.random.default_rng(7)
    months = pd.date_range("2015-01-31", "2023-12-31", freq="ME")
    out, c, first = [], 0.0, None
    for t in months:
        est = 0.35 + 0.05 * rng.standard_normal() if t < pd.Timestamp("2021-10-31") else \
            0.35 - 0.06 * ((t - pd.Timestamp("2021-10-31")).days / 30) + 0.03 * rng.standard_normal()
        est = max(-0.8, est)
        c = max(0.0, c - (est - 0.35) / 0.15 - 0.5)
        if t == pd.Timestamp("2022-06-30"):
            c = max(c, 6.0)
        alarm = t >= pd.Timestamp("2022-06-30") and c > 5.0
        if alarm and first is None:
            first = t.date()
        out.append(ReplayFrame(rule_id="bonds_protect", as_of=t.date(),
                               integrity={"vendors_agree": True, "max_gap_pct": 0.3, "status": "ok"},
                               strength={"est": round(est, 3), "lo": round(est - 0.15, 3), "hi": round(est + 0.12, 3)},
                               change={"cusum": round(c, 3), "threshold": 5.0, "alarm": bool(alarm)},
                               first_alarm=first))
    return out


def main():
    series = build()
    write_cache(series)
    import os
    os.environ["WHACK_CACHE_DIR"] = str(MOCK / "cache")
    from data import pit
    pit.clear_cache()
    pit.panel().to_parquet(MOCK / "m2_panel.parquet")
    (MOCK / "sample_m2_frames.jsonl").write_text("\n".join(f.model_dump_json() for f in sample_frames()) + "\n")
    print("mock cache:", len(list((MOCK / "cache").glob("*.parquet"))), "files; m2_panel + sample_m2_frames written")


if __name__ == "__main__":
    main()
