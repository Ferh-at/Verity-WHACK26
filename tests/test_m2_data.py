"""M2.11 unit tests: fetch/cache, NBER, point-in-time loader, vendor check, mocks."""
import json

import numpy as np
import pandas as pd
import pytest

import schemas as S
from catalog import CATALOG, ids
from data import fetch, nber, pit, store, vendor


# ---------------------------------------------------------------- mocks parse (C8)
def test_mock_frames_parse():
    lines = (store.ROOT / "mocks" / "sample_m2_frames.jsonl").read_text().splitlines()
    frames = [S.ReplayFrame.model_validate_json(l) for l in lines if l.strip()]
    assert frames[0].as_of.isoformat() == "2015-01-31" and frames[-1].as_of.isoformat() == "2023-12-31"
    assert next(f for f in frames if f.change["alarm"]).as_of.isoformat() == "2022-06-30"


def test_mock_panel_columns():
    p = pd.read_parquet(store.ROOT / "mocks" / "m2_panel.parquet")
    assert set(ids()) | {"RF"} <= set(p.columns)
    assert p.index.min() == pd.Timestamp("2000-01-31") and p.index.max() == pd.Timestamp("2026-10-31")


# ---------------------------------------------------------------- fetch
def test_check_table_counts():
    tab = fetch.check_table()
    prim = tab.groupby("id").first()
    for sid in ids():
        n = int(tab[(tab.id == sid)].n.max())
        assert n > (120 if sid == "BTC" else 200), (sid, n)


def test_fetch_all_with_fake_vendors(tmp_path, monkeypatch):
    """Exercise fetch_all end to end without the network: resampling, release lag, ffill limit."""
    monkeypatch.setattr(store, "DATA_CACHE", tmp_path)
    monkeypatch.setenv("OFFLINE", "0")
    days = pd.bdate_range("2000-01-03", "2001-12-31")
    daily = pd.Series(np.arange(len(days), dtype=float), index=days)
    monkeypatch.setattr(fetch, "fetch_yahoo", lambda s, a, b: daily)
    monkeypatch.setattr(fetch, "fetch_stooq", lambda s, a, b: daily[::-1])   # newest first, like Stooq

    def fake_fred(sid, a, b):
        if CATALOG[sid]["freq"] == "monthly":
            return pd.Series(1.0, index=pd.date_range("2000-01-01", "2001-12-01", freq="MS"))
        x = daily.copy()
        x.iloc[5:8] = np.nan                       # 3-day holiday gap -> forward-filled
        return x
    monkeypatch.setattr(fetch, "fetch_fred", fake_fred)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    rep = fetch.fetch_all("2000-01-31", "2001-12-31")
    assert all(v.startswith("ok") for v in rep.values()), rep
    spx = store.read_cache("yahoo", "SPX", tmp_path)
    assert (spx["date"] == spx["date"] + pd.offsets.MonthEnd(0)).all()       # month-ends
    assert spx["value"].iloc[0] == daily.loc[:"2000-01-31"].iloc[-1]          # last day of month
    assert (spx["published"] == spx["date"]).all()
    st = store.read_cache("stooq", "SPX", tmp_path)
    assert st["value"].tolist() == spx["value"].tolist()                       # sorted correctly
    un = store.read_cache("fred", "UNRATE", tmp_path)
    assert (un["published"] == un["date"] + pd.offsets.MonthEnd(1)).all()     # one-month release lag
    assert (tmp_path / "fred_UNRATE.csv").exists()                              # CSV mirror


def test_fetch_failure_keeps_existing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_CACHE", tmp_path)
    monkeypatch.setenv("OFFLINE", "0")
    old = pd.DataFrame({"date": [pd.Timestamp("2000-01-31")], "value": [1.0], "published": [pd.Timestamp("2000-01-31")]})
    store.write_cache(old, "yahoo", "SPX", tmp_path)

    def boom(*a):
        raise ConnectionError("down")
    for f in ("fetch_yahoo", "fetch_stooq", "fetch_fred"):
        monkeypatch.setattr(fetch, f, boom)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    rep = fetch.fetch_all("2000-01-31", "2000-12-31")
    assert "kept existing cache" in rep["yahoo_SPX"]
    assert store.read_cache("yahoo", "SPX", tmp_path)["value"].tolist() == [1.0]


def test_offline_fetches_nothing(monkeypatch):
    monkeypatch.setenv("OFFLINE", "1")
    monkeypatch.setattr(fetch, "fetch_yahoo", lambda *a: (_ for _ in ()).throw(AssertionError("network")))
    assert fetch.fetch_all() == {}


# ---------------------------------------------------------------- NBER
def test_nber_point_in_time():
    before = nber.recession_known("2008-11-30")
    assert not before.loc["2008"].any()
    assert before.loc["2001-04":"2001-11"].all() and not before.loc["2001-03"].any()
    after = nber.recession_known("2008-12-31")
    assert after.loc["2008-01":"2008-12"].all()
    assert not after.loc["2007-12"].any()
    # trough of 2009-06 announced 2010-09-20: before that, the recession runs to as_of
    mid = nber.recession_known("2010-06-30")
    assert mid.loc["2010-06"].all()
    late = nber.recession_known("2010-12-31")
    assert late.loc["2009-06"].all() and not late.loc["2009-07":].any()


# ---------------------------------------------------------------- PIT property test
def test_pit_property_1000_random_t():
    full = pit.panel()
    rng = np.random.default_rng(0)
    days = pd.date_range("2001-01-01", "2026-10-31", freq="D")
    market = [c for c in ids() if CATALOG[c]["freq"] == "daily"]
    for t in days[rng.integers(0, len(days), 1000)]:
        p = pit.panel(t)
        pub = pit.published_frame(t)
        cols = [c for c in pub.columns]
        vals = p[cols]
        # no value whose published date is after t
        leak = vals.notna() & ~(pub.notna() & (pub <= t))
        assert not leak.values.any(), t
        assert p.index.max() <= t
        prev = store.month_end(t) - pd.offsets.MonthEnd(1)
        pd.testing.assert_frame_equal(p.loc[:prev, market], full.loc[:prev, market])


def test_pit_macro_release_lag():
    p = pit.panel("2010-06-30")
    assert np.isnan(p.loc["2010-06-30", "UNRATE"])          # June value not yet published
    assert not np.isnan(p.loc["2010-05-31", "UNRATE"])      # May value published at end of June
    s = pit.as_of("UNRATE", "2010-06-30")
    assert s.index.max() == pd.Timestamp("2010-05-31")


def test_pit_rf():
    p = pit.panel()
    assert np.isclose(p.loc["2010-06-30", "RF"], p.loc["2010-05-31", "DGS3MO"] / 1200)


# ---------------------------------------------------------------- vendor
def test_vendor_injected_error_flags():
    idx = pd.date_range("2010-01-31", periods=48, freq="ME")
    py = pd.Series(100 * np.cumprod(1 + 0.01 * np.random.default_rng(1).standard_normal(48)), index=idx)
    ps = py.copy()
    assert vendor.vendor_check_series(py, ps, idx[-1])["status"] == "ok"
    bad = py.copy()
    for i in (20, 30, 40):          # 2% error in 3 price months -> 3+ bad return months
        bad.iloc[i] *= 1.02
    res = vendor.vendor_check_series(py, bad, idx[-1])
    assert res["status"] == "flag" and res["n_bad"] > 2


def test_vendor_single_vendor():
    assert vendor.vendor_check("DXY", "2020-12-31")["status"] == "single_vendor"
    assert vendor.vendor_check("BENCH_A3", "2020-12-31")["status"] == "single_vendor"


def test_vendor_test_returns_testresult():
    h = S.Hypothesis(id="hy_x", thesis_id=None, source="thesis",
                     spec=S.Spec(feature="mom", series="SPX", lookback=12, transform="sign", asset="SPX", direction=1),
                     rationale="", spec_hash="x", created_at="2026-11-07T00:00:00Z")
    r = vendor.vendor_test(h, pit.panel())
    assert r.test == "vendor" and r.kill is True and r.passed is True
