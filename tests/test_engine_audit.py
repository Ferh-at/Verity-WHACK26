"""Leak-audit controls against data/audit.py."""
import numpy as np
import pandas as pd

from data.audit import edit_future_demo, leak_audit
from data.pit import panel as mock_panel
from engine.features import feature
from schemas import Spec


def test_controls():
    p = mock_panel()
    assert leak_audit(lambda d: d["SPX"].rolling(3).mean(), p).passed
    assert not leak_audit(lambda d: d["SPX"].rolling(3, center=True).mean(), p).passed
    assert not leak_audit(lambda d: (d["SPX"] - d["SPX"].mean()) / d["SPX"].std(), p).passed


def test_clean_features_pass():
    p = mock_panel()
    for s in [Spec(feature="mom", series="SPX", lookback=12, transform="sign", asset="SPX", direction=1),
              Spec(feature="level", series="T10Y3M", lookback=1, transform="sign", asset="SPX", direction=1),
              Spec(feature="change", series="DGS10", lookback=3, transform="clip_z", asset="TLT", direction=1),
              Spec(feature="zscore", series="VIX", lookback=24, transform="sign", asset="SPX", direction=1),
              Spec(feature="corr", series="SPX", series2="TLT", lookback=36, transform="sign", asset="GOLD",
                   direction=1)]:
        assert leak_audit(lambda d, s=s: feature(s, d), p).passed, s.feature


def test_edit_future_demo():
    p = mock_panel()
    leaky = Spec(feature="centered_ma", series="SPX", lookback=3, transform="sign", asset="SPX", direction=1)
    clean = Spec(feature="mom", series="SPX", lookback=3, transform="sign", asset="SPX", direction=1)
    t = pd.Timestamp("2010-06-30")
    assert not edit_future_demo(leaky, p, t, 1.3)["passed"]
    assert edit_future_demo(clean, p, t, 1.3)["passed"]
