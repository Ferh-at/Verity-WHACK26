"""Factor check: is any alpha left after market, trend and carry?"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from config import FACTOR_T_MIN
from engine.stats import nw_lag
from schemas import TestResult

TREND_ASSETS = ["SPX", "TLT", "GOLD", "DXY"]


def factor_proxies(panel: pd.DataFrame) -> pd.DataFrame:
    """Monthly factor returns indexed by return date t+1 (real data)."""
    rf = panel["RF"] if "RF" in panel else pd.Series(0.0, index=panel.index)
    ret = {a: panel[a].pct_change(fill_method=None) for a in TREND_ASSETS if a in panel}
    out = {}
    if "SPX" in ret:
        out["MKT"] = ret["SPX"] - rf
    if ret:
        legs = [np.sign(panel[a] / panel[a].shift(12) - 1).shift(1) * ret[a] for a in ret]
        out["TREND"] = pd.concat(legs, axis=1).mean(axis=1, skipna=True)
    if "T10Y3M" in panel and "TLT" in ret:
        out["CARRY"] = np.sign(panel["T10Y3M"]).shift(1) * (ret["TLT"] - rf)
    return pd.DataFrame(out)


def factor_test(net: pd.Series, proxies: pd.DataFrame, hypothesis_id: str = "",
                kill: bool = True) -> TestResult:
    """OLS of net on [1, proxies] with HAC errors; passes if the alpha NW t > FACTOR_T_MIN."""
    df = pd.concat([net.rename("net"), proxies], axis=1, join="inner").dropna()
    if len(df) < 24:
        return TestResult(hypothesis_id=hypothesis_id, test="factor", passed=True, kill=False, value=None,
                          detail=f"untestable (n={len(df)} months with factor data)", evidence_ref="factor")
    X = sm.add_constant(df[list(proxies.columns)], has_constant="add")
    fit = sm.OLS(df["net"], X).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag(len(df))})
    t = float(fit.tvalues["const"])
    alpha_ann = float(fit.params["const"]) * 12
    passed = t > FACTOR_T_MIN
    return TestResult(hypothesis_id=hypothesis_id, test="factor", passed=passed, kill=kill, value=t,
                      detail=f"alpha after {'/'.join(proxies.columns)}: {alpha_ann:.2%}/yr, NW t={t:.2f}"
                             f" ({'>' if passed else '<='} {FACTOR_T_MIN})",
                      evidence_ref="factor")
