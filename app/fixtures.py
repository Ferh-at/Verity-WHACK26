"""Explicitly synthetic demo inputs. Production artefacts always come from the team."""
import math
from datetime import date

import numpy as np
import pandas as pd

from agents.generator import make_hypothesis
from agents.llm import now
from config import AUDIT_TOL, CALIB_WINDOW, DATA_END, DATA_START, REPLAY_START, SEED
from schemas import ReplayFrame, Spec

RULES = {
    "bonds_protect": "Bonds protect stocks",
    "dollar_haven": "The dollar is a safe haven",
    "gold_realyield": "Gold rises when real yields fall",
    "btc_diversifier": "Bitcoin doesn't move with stocks",
    "curve_recession": "Curve inversion means recession",
    "vix_control": "Fear spikes when stocks fall",
}
CONTROL = make_hypothesis(
    Spec(feature="centered_ma", series="SPX", lookback=3, transform="sign", asset="SPX", direction=1),
    "planted", "Centred averages consume the following month.",
)


def sample_panel():
    rng = np.random.default_rng(SEED)
    index = pd.date_range(DATA_START, DATA_END, freq="ME")
    prices = ("SPX", "NASDAQ", "TLT", "GOLD", "DXY", "BTC")
    levels = ("VIX", "DGS10", "DGS2", "DGS3MO", "T10Y3M", "T10Y2Y", "DFII10",
              "UNRATE", "CPIAUCSL", "FEDFUNDS")
    panel = pd.DataFrame(index=index)
    for i, col in enumerate(prices):
        panel[col] = (100 + i * 15) * (1 + rng.normal(.005, .035, len(index))).cumprod()
    for i, col in enumerate(levels):
        panel[col] = 4 + i * .2 + rng.normal(0, .7, len(index)).cumsum() / 8
    panel["CPIAUCSL"] = 100 * (1 + rng.uniform(.001, .004, len(index))).cumprod()
    panel["NBER"] = 0
    panel["RF"] = panel["DGS3MO"].shift() / 1200
    return panel


def _sample_feature(spec, panel):
    x = panel[spec.series]
    n = spec.lookback
    prices = {"SPX", "NASDAQ", "TLT", "GOLD", "DXY", "BTC"}
    if spec.feature == "level":
        return x
    if spec.feature == "mom":
        return x.pct_change(n, fill_method=None)
    if spec.feature == "change":
        return np.log(x).diff(n) if spec.series in prices else x.diff(n)
    if spec.feature == "zscore":
        window = x.rolling(n)
        return (x - window.mean()) / window.std()
    if spec.feature == "corr":
        y = panel[spec.series2]
        dx = x.pct_change(fill_method=None) if spec.series in prices else x.diff()
        dy = y.pct_change(fill_method=None) if spec.series2 in prices else y.diff()
        return dx.rolling(n).corr(dy)
    if spec.feature == "centered_ma":
        return x.diff().rolling(n, center=True).mean()
    if spec.feature == "fullsample_z":
        return (x - x.mean()) / x.std()
    raise ValueError("Unsupported fixture feature")


def sample_edit(spec, panel, t, multiplier):
    t = pd.Timestamp(t)
    changed = panel.copy()
    changed.loc[changed.index > t] *= multiplier

    def read(frame):
        f = _sample_feature(spec, frame)
        p = np.sign(f)
        if spec.transform == "clip_z":
            from config import ZCLIP, ZMIN, ZWIN
            trailing = f.rolling(ZWIN, min_periods=ZMIN)
            p = ((f - trailing.mean()) / trailing.std()).clip(-ZCLIP, ZCLIP) / ZCLIP
        return float(f.loc[t]), float(spec.direction * p.loc[t])

    before, p_before = read(panel)
    after, p_after = read(changed)
    return {
        "before": before, "after": after, "position_before": p_before, "position_after": p_after,
        "passed": abs(before - after) <= AUDIT_TOL and abs(p_before - p_after) <= AUDIT_TOL,
    }


def sample_hypotheses():
    t_id = "th_00000001"
    specs = [
        ("level", "T10Y3M", None, 1, "sign", "SPX", 1),
        ("change", "T10Y2Y", None, 6, "clip_z", "NASDAQ", 1),
        ("zscore", "T10Y3M", None, 24, "clip_z", "SPX", 1),
        ("mom", "TLT", None, 6, "sign", "SPX", 1),
        ("corr", "SPX", "TLT", 12, "sign", "SPX", -1),
        ("change", "DGS2", None, 3, "sign", "SPX", -1),
        ("change", "DGS10", None, 12, "sign", "TLT", -1),
        ("level", "T10Y2Y", None, 1, "clip_z", "TLT", -1),
        ("zscore", "DFII10", None, 36, "sign", "GOLD", -1),
        ("mom", "DXY", None, 3, "clip_z", "SPX", -1),
        ("level", "VIX", None, 1, "clip_z", "SPX", -1),
        ("corr", "SPX", "DXY", 36, "clip_z", "GOLD", 1),
        ("change", "UNRATE", None, 12, "sign", "SPX", -1),
        ("level", "T10Y3M", None, 1, "sign", "NASDAQ", 1),
        ("mom", "NASDAQ", None, 12, "clip_z", "TLT", -1),
        ("change", "T10Y3M", None, 3, "sign", "TLT", 1),
        ("zscore", "FEDFUNDS", None, 24, "sign", "NASDAQ", -1),
        ("corr", "T10Y3M", "SPX", 24, "clip_z", "SPX", 1),
        ("level", "DFII10", None, 1, "clip_z", "GOLD", -1),
        ("change", "CPIAUCSL", None, 12, "clip_z", "SPX", -1),
    ]
    return [
        make_hypothesis(Spec(**dict(zip(Spec.model_fields, values))), "thesis",
                        "A candidate macro transmission channel for the illustrative thesis.", t_id)
        for values in specs
    ]


def sample_replay():
    rng = np.random.default_rng(SEED)
    index = pd.date_range(REPLAY_START, DATA_END, freq="ME")
    threshold = 9.2
    baseline = date.fromisoformat(CALIB_WINDOW[1])
    frames = []
    for name in RULES:
        first = None
        for i, stamp in enumerate(index):
            day = stamp.date()
            if name == "btc_diversifier" and day < date(2016, 9, 30):
                continue
            drift = max(0, i - 83) if name == "bonds_protect" else 0
            est = .38 + .08 * math.sin(i / 8) - min(.75, drift * .042) + rng.normal(0, .012)
            if name == "curve_recession":
                est = max(0, min(1, est))
            c = max(0, .8 * math.sin(i / 8)) + drift * 1.6 if day > baseline else 0
            alarm = c > threshold and name != "curve_recession"
            if alarm and first is None:
                first = day
            frames.append(ReplayFrame(
                rule_id=name, as_of=day,
                integrity={"status": "single_vendor" if name in {"dollar_haven", "vix_control"} else "ok",
                           "vendors_agree": None if name in {"dollar_haven", "vix_control"} else True},
                strength={"est": est, "lo": est - .13, "hi": est + .13},
                change={} if name == "curve_recession" else {"cusum": c, "threshold": threshold, "alarm": alarm},
                first_alarm=first,
            ))
    return frames
