"""M2.6-M2.10 monitor: unit tests and must-haves 8 & 9."""
import json

import numpy as np
import pandas as pd
import pytest

import schemas as S
from config import N_CALIB_SIMS, N_BREAK_SIMS, ROLL_WIN
from data import pit, store
from monitor.cusum import run_cusum, cusum_paths, n_episodes
from monitor.rules import strength, strength_series, rolling_stat, window_stat, RULES
from monitor import calibrate as cal
from monitor import breaks, replay as rp


# ---------------------------------------------------------------- CUSUM
def test_cusum_constant_is_zero():
    s = pd.Series(0.3, index=pd.date_range("2010-01-31", periods=50, freq="ME"))
    out = run_cusum(s, 0.3, 0.1, h=3)
    assert (out["C"] == 0).all() and not out["alarm"].any()


def test_cusum_step_down_2sd():
    s = pd.Series([0.3] * 10 + [0.1] * 10, index=pd.date_range("2010-01-31", periods=20, freq="ME"))
    out = run_cusum(s, 0.3, 0.1, h=3)
    np.testing.assert_allclose(out["C"].iloc[10:13].values, [1.5, 3.0, 4.5])
    assert out["alarm"].values.argmax() == 12          # 3rd month after the step


def test_cusum_vectorised_matches_loop():
    rng = np.random.default_rng(3)
    Z = rng.standard_normal((5, 100)) - 0.3
    C = cusum_paths(Z)
    for i in range(5):
        s = pd.Series(Z[i], index=pd.date_range("2000-01-31", periods=100, freq="ME"))
        np.testing.assert_allclose(run_cusum(s, 0, 1, 3)["C"].values, C[i])


# ---------------------------------------------------------------- strength
def _exact_pair(n, rho, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n); x = (x - x.mean()) / x.std()
    e = rng.standard_normal(n); e = e - e.mean(); e = e - (e @ x) / (x @ x) * x; e = e / e.std()
    y = rho * x + np.sqrt(1 - rho ** 2) * e
    return 0.01 + 0.04 * x, 0.005 + 0.03 * y


def test_strength_synthetic_corr_minus_half():
    n = ROLL_WIN
    ra, rb = _exact_pair(n, -0.5)
    idx = pd.date_range("2009-12-31", periods=n + 1, freq="ME")
    p = pd.DataFrame({"SPX": 100 * np.r_[1, np.cumprod(1 + ra)], "TLT": 100 * np.r_[1, np.cumprod(1 + rb)]}, index=idx)
    st = strength("bonds_protect", idx[-1], panel=p)
    assert abs(st["est"] - 0.5) <= 0.15
    assert st["lo"] <= st["est"] <= st["hi"]


def test_rolling_stat_matches_pandas():
    p = pit.panel()
    for r in ["bonds_protect", "gold_realyield"]:
        s = strength_series(r, p).dropna()
        from monitor.rules import pairs
        P = pairs(r, p).loc[s.index[0] - pd.offsets.MonthEnd(RULES[r]["window"] - 1): s.index[-1]]
        v = rolling_stat(P["a"].values[None], P["b"].values[None], RULES[r]["window"], RULES[r]["stat"])[0]
        np.testing.assert_allclose(v[:50], s.values[:50], rtol=1e-8)


def test_strength_uses_only_panel_as_of():
    a = strength("bonds_protect", "2018-06-30")
    p = pit.panel()
    p.loc["2018-07-31":, "SPX"] *= 3.0                 # wreck the future
    b = strength("bonds_protect", "2018-06-30", panel=p.loc[:"2018-06-30"])
    assert a == b


def test_curve_recession_strength():
    st = strength("curve_recession", "2026-10-31")
    assert st["n_spells"] >= 1 and 0 <= st["est"] <= 1


# ---------------------------------------------------------------- calibration (must-have 8)
@pytest.fixture(scope="module")
def calib(tmp_path_factory):
    path = tmp_path_factory.mktemp("c") / "calib.json"
    res = {r: cal.calibrate(r, n=N_CALIB_SIMS, checkpoint=False) for r in ["bonds_protect", "vix_control"]}
    cal.save_calib(res, path)
    return path, res


def test_calibration_false_alarm_rate(calib):
    _, res = calib
    for r, c in res.items():
        assert c["n_sims"] >= 1000
        # must-have: <= 1 false alarm per 20 years per rule (spec acceptance band is [0.5, 1.5]; see README)
        assert c["fa_rate_per_20y"] <= 1.0, (r, c["fa_rate_per_20y"])
        assert c["h"] > 0 and len(c["max_c"]) == c["n_sims"]


def test_calibration_reproducible():
    a = cal.calibrate("bonds_protect", n=600, checkpoint=False)
    b = cal.calibrate("bonds_protect", n=600, checkpoint=False)
    assert a["h"] == b["h"] and a["max_c"] == b["max_c"]


def test_calibration_checkpoint_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CKPT_DIR", tmp_path)
    a = cal.calibrate("vix_control", n=1000, checkpoint=True)
    b = cal.calibrate("vix_control", n=1000, checkpoint=True)      # loads from checkpoint
    assert a["max_c"] == b["max_c"] and any(tmp_path.iterdir())


# ---------------------------------------------------------------- breaks (must-have 9)
def test_break_sims_report(calib):
    _, res = calib
    runs = breaks.break_sims("bonds_protect", N_BREAK_SIMS, params=res["bonds_protect"])
    assert len(runs) >= 200 and all(isinstance(r, S.BenchmarkRun) and r.kind == "break" for r in runs)
    m = breaks.metrics(runs)
    assert m["median_delay_m"] is not None and 0 <= m["share_detected_within_24m"] <= 1
    print("\nbonds_protect planted breaks:", {k: v for k, v in m.items() if k != "delays"})


# ---------------------------------------------------------------- replay (must-have 8)
@pytest.fixture(scope="module")
def frames(calib, tmp_path_factory):
    path, _ = calib
    out = tmp_path_factory.mktemp("f") / "frames.jsonl"
    rp.replay(["bonds_protect", "vix_control", "curve_recession"], "2003-01-31", out=out, calib_path=path)
    return out, path


def test_replay_frames_valid(frames):
    out, _ = frames
    fr = rp.load_frames(out)
    months = {f.as_of for f in fr if f.rule_id == "bonds_protect"}
    assert len(months) >= 280            # 2003-01 .. 2026-10
    assert all(isinstance(f, S.ReplayFrame) for f in fr)


def test_vix_control_never_alarms_real_data(frames):
    if store.using_mock():
        pytest.skip("VIX-never-alarms is a statement about REAL data; mock cache in use")
    out, _ = frames
    assert not any(f.change.get("alarm") for f in rp.load_frames(out, "vix_control"))


def test_replay_same_seed_same_first_alarm(frames, tmp_path):
    out, path = frames
    out2 = tmp_path / "f2.jsonl"
    rp.replay(["bonds_protect"], "2003-01-31", out=out2, calib_path=path)
    a = rp.load_frames(out, "bonds_protect")[-1].first_alarm
    b = rp.load_frames(out2, "bonds_protect")[-1].first_alarm
    assert a == b


def test_replay_is_point_in_time(frames, tmp_path, monkeypatch):
    """A frame at t must not change when data after t is edited."""
    out, path = frames
    t = pd.Timestamp("2016-06-30")
    before = [f for f in rp.load_frames(out, "bonds_protect") if pd.Timestamp(f.as_of) == t][0]
    real_panel = pit.panel

    def edited(tt=None, ids=None):
        p = real_panel(None, ids)
        p.loc[p.index > t, ["SPX", "TLT"]] *= 1.7
        return p if tt is None else p.loc[:pd.Timestamp(tt)]
    monkeypatch.setattr(pit, "panel", edited)
    monkeypatch.setattr(rp.pit, "panel", edited)
    out2 = tmp_path / "f3.jsonl"
    rp.replay(["bonds_protect"], "2016-01-31", "2016-12-31", out=out2, calib_path=path)
    after = [f for f in rp.load_frames(out2, "bonds_protect") if pd.Timestamp(f.as_of) == t][0]
    assert before.strength == after.strength and before.change == after.change


def test_freeze_frame(frames):
    out, _ = frames
    ff = rp.freeze_frame("bonds_protect", out)
    if ff is not None:
        assert ff.as_of == ff.first_alarm and ff.change["alarm"]
        assert 1 <= len(ff.strength["last12"]) <= 12
        facts = rp.explainer_facts(ff, out)
        assert set(facts) >= {"rule", "as_of", "est", "lo", "hi", "cusum", "h", "alarm", "integrity",
                              "months_since_calib_end"}
