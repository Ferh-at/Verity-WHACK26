"""Leak audit: leaky controls are flagged and clean features are not."""
import numpy as np
import pandas as pd
import pytest

import schemas as S
from data import pit
from data.audit import leak_audit, leak_audit_spec, edit_future_demo
from data.features_ref import get_feature_fns


@pytest.fixture(scope="module")
def panel():
    return pit.panel()


def spec(**kw):
    base = dict(feature="mom", series="SPX", series2=None, lookback=12, transform="sign", asset="SPX", direction=1)
    base.update(kw)
    return S.Spec(**base)


def test_trailing_mean_passes(panel):
    r = leak_audit(lambda p: p["SPX"].pct_change(fill_method=None).rolling(3).mean(), panel, write_evidence=False)
    assert r.passed and r.kill and r.test == "leak_audit"
    assert "no dependence" in r.detail


def test_centered_mean_flagged(panel):
    r = leak_audit(lambda p: p["SPX"].pct_change(fill_method=None).rolling(3, center=True).mean(), panel,
                   write_evidence=False)
    assert not r.passed and "leak at feature date" in r.detail


def test_fullsample_z_flagged(panel):
    r = leak_audit(lambda p: (p["T10Y3M"] - p["T10Y3M"].mean()) / p["T10Y3M"].std(), panel, write_evidence=False)
    assert not r.passed


CLEAN = [spec(feature="mom", series=s, lookback=L) for s, L in
         [("SPX", 1), ("SPX", 12), ("NASDAQ", 6), ("TLT", 3), ("GOLD", 12), ("DXY", 24)]] + \
        [spec(feature="level", series=s, lookback=1) for s in ["T10Y3M", "VIX", "DFII10", "UNRATE"]] + \
        [spec(feature="change", series=s, lookback=L) for s, L in [("DGS10", 3), ("SPX", 6), ("CPIAUCSL", 12)]] + \
        [spec(feature="zscore", series=s, lookback=L) for s, L in [("T10Y2Y", 36), ("VIX", 12), ("FEDFUNDS", 24)]] + \
        [spec(feature="corr", series=a, series2=b, lookback=L) for a, b, L in
         [("SPX", "TLT", 36), ("GOLD", "DFII10", 24), ("SPX", "VIX", 12), ("BTC", "NASDAQ", 12)]]
LEAKY = [spec(feature="centered_ma", series=s, lookback=L) for s, L in
         [("SPX", 3), ("TLT", 3), ("GOLD", 6), ("NASDAQ", 12), ("DXY", 3), ("VIX", 6), ("T10Y3M", 3),
          ("DGS10", 12), ("UNRATE", 6), ("DFII10", 24)]] + \
        [spec(feature="fullsample_z", series=s, lookback=1) for s in
         ["SPX", "TLT", "GOLD", "T10Y3M", "VIX", "DGS10", "DFII10", "UNRATE", "CPIAUCSL", "BTC"]]


def test_must_have_3_clean_and_leaky(panel):
    assert len(CLEAN) >= 20 and len(LEAKY) >= 20
    clean_flagged = [s for s in CLEAN if not leak_audit_spec(s, panel, write_evidence=False).passed]
    leaky_missed = [s for s in LEAKY if leak_audit_spec(s, panel, write_evidence=False).passed]
    assert clean_flagged == [], clean_flagged
    assert leaky_missed == [], leaky_missed


def test_clip_z_position_is_clean(panel):
    _, po = get_feature_fns()
    s = spec(feature="zscore", series="T10Y3M", lookback=24, transform="clip_z")
    assert leak_audit(lambda p: po(s, p), panel, write_evidence=False).passed


def test_evidence_written(panel, tmp_path, monkeypatch):
    import data.audit as A
    monkeypatch.setattr(A, "EVIDENCE", tmp_path / "ev.jsonl")
    r = leak_audit_spec(spec(feature="centered_ma", series="SPX", lookback=3), panel, hypothesis_id="hy_demo")
    line = (tmp_path / "ev.jsonl").read_text().strip()
    assert r.evidence_ref.endswith(line.split('"id": "')[1].split('"')[0])
    assert '"counterexample"' in line


def test_edit_future_demo(panel):
    leaky = spec(feature="centered_ma", series="SPX", lookback=3)
    out = edit_future_demo(leaky, panel, "2008-09-30", 1.3)
    assert out["passed"] is False and out["before"] != out["after"]
    clean = spec(feature="mom", series="SPX", lookback=12)
    out = edit_future_demo(clean, panel, "2008-09-30", 1.3)
    assert out["passed"] is True and out["before"] == out["after"]
    assert set(out) >= {"before", "after", "position_before", "position_after", "passed"}
