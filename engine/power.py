"""Power, PSR, MinTRL and deflated Sharpe.
SR_m = monthly Sharpe, n = months, g3 = skewness, g4 = Pearson kurtosis (normal = 3)."""
import math

import numpy as np
from scipy import stats as sps
from scipy.stats import norm

from config import ALPHA, POWER_TARGET
from schemas import Power

EULER_GAMMA = 0.5772156649
Z_ALPHA = norm.ppf(1 - ALPHA)


def mde80(years: float) -> float:
    """Annual Sharpe detectable with 80% power at one-sided ALPHA."""
    if years <= 0:
        return float("inf")
    return float((Z_ALPHA + norm.ppf(POWER_TARGET)) / math.sqrt(years))


def power(sr_ann: float, years: float) -> float:
    return float(norm.cdf(sr_ann * math.sqrt(max(years, 0.0)) - Z_ALPHA))


def _sr_var_term(sr_m: float, g3: float, g4: float) -> float:
    return max(1 - g3 * sr_m + (g4 - 1) / 4 * sr_m ** 2, 1e-12)


def psr(sr_m: float, sr_star_m: float, n: int, g3: float, g4: float) -> float:
    """Probabilistic Sharpe ratio (Bailey & Lopez de Prado 2012)."""
    if n < 2:
        return float("nan")
    return float(norm.cdf((sr_m - sr_star_m) * math.sqrt(n - 1) / math.sqrt(_sr_var_term(sr_m, g3, g4))))


def mintrl_months(sr_m: float, sr_star_m: float = 0.0, g3: float = 0.0, g4: float = 3.0) -> float | None:
    """Minimum track record length in months; None unless sr_m > sr_star_m."""
    if not (sr_m > sr_star_m):
        return None
    return float(1 + _sr_var_term(sr_m, g3, g4) * (Z_ALPHA / (sr_m - sr_star_m)) ** 2)


def expected_max_sr(sr_ms_of_trials, n_eff: int) -> float:
    """SR0 of the deflated Sharpe: expected max monthly Sharpe of N_eff unskilled trials."""
    if n_eff < 2 or len(sr_ms_of_trials) < 2:
        return 0.0
    sd = float(np.std(np.asarray(sr_ms_of_trials, dtype=float), ddof=1))
    N = n_eff
    return sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / N) + EULER_GAMMA * norm.ppf(1 - 1 / (N * math.e)))


def dsr(sr_m: float, n: int, g3: float, g4: float, sr_ms_of_trials, N_eff: int) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014)."""
    return psr(sr_m, expected_max_sr(sr_ms_of_trials, N_eff), n, g3, g4)


def moments(x) -> tuple[float, float, float]:
    """(monthly Sharpe, skewness, Pearson kurtosis)."""
    x = np.asarray(x, dtype=float)
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    if len(x) < 4 or sd == 0:
        return 0.0, 0.0, 3.0
    return float(x.mean() / sd), float(sps.skew(x)), float(sps.kurtosis(x, fisher=False))


def power_line(net_returns) -> Power:
    """Power line from the discovery-split net returns (years = n/12)."""
    x = np.asarray(net_returns, dtype=float)
    years = len(x) / 12
    sr_m, g3, g4 = moments(x)
    mt = mintrl_months(sr_m, 0.0, g3, g4)
    return Power(years=round(years, 4), mde_sharpe_80=mde80(max(years, 1 / 12)),
                 mintrl_years=None if mt is None else mt / 12, power_at_05=power(0.5, years))
