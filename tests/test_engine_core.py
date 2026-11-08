"""DSL, features, backtest, stats, power, multiple testing and dedupe."""
import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from engine import backtest, dsl, features, multitest, power, stats
from engine.dedupe import cluster_returns, effective_n
from schemas import Spec


def spec(**kw):
    base = dict(feature="mom", series="SPX", lookback=12, transform="sign", asset="SPX", direction=1)
    base.update(kw)
    return Spec(**base)


# --- DSL
def test_validate_ok():
    dsl.validate_spec(spec())
    dsl.validate_spec(spec(feature="corr", series="SPX", series2="TLT", lookback=36, asset="GOLD"))


@pytest.mark.parametrize("kw,msg", [
    (dict(lookback=5), "lookback"),
    (dict(feature="level", lookback=3), "level"),
    (dict(feature="zscore", lookback=6), "zscore"),
    (dict(series="NOPE"), "unknown series"),
    (dict(feature="corr", lookback=12), "series2"),
    (dict(feature="corr", series2="SPX", lookback=12), "differ"),
    (dict(series2="TLT"), "only allowed"),
    (dict(asset="VIX"), "tradable"),
    (dict(feature="centered_ma", lookback=3), "leaky"),
])
def test_validate_rejects(kw, msg):
    with pytest.raises(ValueError, match=msg):
        dsl.validate_spec(spec(**kw))


def test_leaky_allowed_when_planted():
    dsl.validate_spec(spec(feature="centered_ma", lookback=3), source="planted")


def test_spec_hash_order_invariant():
    a = Spec(feature="mom", series="SPX", lookback=12, transform="sign", asset="SPX", direction=1)
    b = Spec(direction=1, asset="SPX", transform="sign", lookback=12, series="SPX", feature="mom")
    assert dsl.spec_hash(a) == dsl.spec_hash(b)
    h = dsl.make_hypothesis(a, "thesis", "x")
    assert h.id == "hy_" + h.spec_hash[:10] and len(h.spec_hash) == 64


# --- features
def _panel(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame({"SPX": 100 * np.cumprod(1 + rng.normal(0.01, 0.04, n)),
                         "TLT": 100 * np.cumprod(1 + rng.normal(0.0, 0.03, n)),
                         "T10Y3M": rng.normal(1, 1, n)}, index=idx)


def test_feature_formulas():
    p = _panel()
    x, P = p["T10Y3M"], p["SPX"]
    f = lambda **kw: features.feature(spec(**kw), p)
    pd.testing.assert_series_equal(f(feature="mom", lookback=3), (P / P.shift(3) - 1), check_names=False)
    pd.testing.assert_series_equal(f(feature="level", series="T10Y3M", lookback=1), x, check_names=False)
    pd.testing.assert_series_equal(f(feature="change", series="T10Y3M", lookback=6), x.diff(6), check_names=False)
    pd.testing.assert_series_equal(f(feature="change", lookback=6), np.log(P).diff(6), check_names=False)
    z = f(feature="zscore", series="T10Y3M", lookback=12)
    t = p.index[30]
    w = x.loc[:t].iloc[-12:]
    assert z[t] == pytest.approx((x[t] - w.mean()) / w.std(ddof=1))
    c = f(feature="corr", series2="TLT", lookback=12)
    rs, rt = P.pct_change().loc[:t].iloc[-12:], p["TLT"].pct_change().loc[:t].iloc[-12:]
    assert c[t] == pytest.approx(np.corrcoef(rs, rt)[0, 1])


def test_positions_in_range():
    p = _panel(120)
    for tr in ("sign", "clip_z"):
        pos = features.position(spec(transform=tr, lookback=3), p).dropna()
        assert len(pos) > 0 and pos.abs().max() <= 1.0


# --- backtest
def test_backtest_unit_example():
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    df = backtest.net_returns(pd.Series([1.0, 1.0, -1.0], idx), pd.Series([0.0, 0.10, -0.10], idx), cost_bps=2.0)
    assert df["net"].iloc[1] == pytest.approx(0.0998, abs=1e-12)
    assert df["net"].iloc[2] == pytest.approx(-0.1000, abs=1e-12)


def test_split_slicing_and_embargo():
    p = _panel(300)
    splits = {"discovery": ("2000-01-31", "2009-12-31"), "validation": ("2010-01-31", "2014-12-31"),
              "holdout": ("2015-01-31", "2024-12-31")}
    df = backtest.strategy_returns(spec(lookback=1), p, "validation", splits)
    assert df.index[0] == pd.Timestamp("2010-02-28")
    assert df.index[-1] == pd.Timestamp("2014-12-31")


def test_short_history_rule():
    p = _panel(300)
    p.loc[p.index < pd.Timestamp("2015-01-31"), "SPX"] = np.nan
    splits = {"discovery": ("2000-01-31", "2016-12-31"), "validation": ("2017-01-31", "2019-12-31"),
              "holdout": ("2020-01-31", "2024-12-31")}
    r = backtest.asset_returns(p, "SPX")
    d = backtest.split_bounds("discovery", splits, r)
    h = backtest.split_bounds("holdout", splits, r)
    n = r.dropna().shape[0]
    months = lambda a, b: (b.year - a.year) * 12 + b.month - a.month + 1
    assert months(*d) == round(0.6 * n) and h[1] == r.dropna().index[-1]


# --- stats
def test_sharpe_unit():
    assert stats.sharpe([0.01, 0.03]) == pytest.approx(4.898979, abs=1e-6)


def test_nw_lag():
    assert stats.nw_lag(144) == 4


def test_nw_matches_statsmodels():
    rng = np.random.default_rng(1)
    X = rng.normal(0.005, 0.04, (144, 5))
    ours = stats.nw_t(X)
    for j in range(5):
        ref = sm.OLS(X[:, j], np.ones(144)).fit(cov_type="HAC", cov_kwds={"maxlags": 4}).tvalues[0]
        assert ours[j] == pytest.approx(ref, abs=1e-6)
        assert stats.nw_t(X[:, j]) == pytest.approx(ref, abs=1e-6)


# --- power
def test_power_numbers():
    assert power.mde80(6) == pytest.approx(1.0151, abs=1e-3)
    assert power.power(0.5, 6) == pytest.approx(0.3372, abs=1e-3)
    m = power.mintrl_months(1 / math.sqrt(12), 0.0, 0.0, 3.0)
    assert m == pytest.approx(34.82, abs=0.01) and m / 12 == pytest.approx(2.90, abs=0.01)
    assert power.mintrl_months(-0.1) is None


def test_dsr_monotone_and_n1():
    trials = list(np.random.default_rng(0).normal(0, 0.1, 50))
    vals = [power.dsr(0.3, 144, 0.0, 3.0, trials, n) for n in (1, 2, 10, 100)]
    assert vals[0] == pytest.approx(power.psr(0.3, 0.0, 144, 0.0, 3.0))
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_power_line():
    x = np.random.default_rng(0).normal(0.01, 0.04, 144)
    pl = power.power_line(x)
    assert pl.years == pytest.approx(12.0) and pl.mde_sharpe_80 == pytest.approx(power.mde80(12))


# --- multiple testing
def test_by_vs_bh_and_small_p():
    p = {f"h{i}": v for i, v in enumerate([0.001] * 5 + [0.5] * 95)}
    by, bh = multitest.by_qvalues(p), multitest.bh_qvalues(p)
    assert all(by[k] >= bh[k] - 1e-15 for k in p)
    # BY q = p * m * c(m) / rank = 0.001 * 100 * H(100) / 5 = 0.1037 (the spec's "< 0.1" is off by c(m))
    c100 = sum(1 / j for j in range(1, 101))
    assert all(by[f"h{i}"] == pytest.approx(0.001 * 100 * c100 / 5) for i in range(5))
    assert all(bh[f"h{i}"] == pytest.approx(0.02) for i in range(5))
    p2 = {f"h{i}": v for i, v in enumerate([0.0005] * 5 + [0.5] * 95)}
    assert all(multitest.by_qvalues(p2)[f"h{i}"] < 0.1 for i in range(5))


# --- dedupe
def test_dedupe_copies_and_independent():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=144, freq="ME")
    base = rng.normal(0, 0.04, 144)
    nets = {f"c{i}": pd.Series(base + rng.normal(0, 0.05 * 0.04, 144), idx) for i in range(20)}
    m = cluster_returns(list(nets), nets)
    assert effective_n(m) == 1 and set(m.values()) == {"c0"}
    ind = {f"n{i}": pd.Series(rng.normal(0, 0.04, 144), idx) for i in range(10)}
    assert effective_n(cluster_returns(list(ind), ind)) == 10
