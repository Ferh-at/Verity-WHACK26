"""Vendor integrity: Yahoo vs Stooq monthly returns.

vendor_check(asset_id, end, window=36) -> {status, max_gap_pct, n_bad, vendors_agree, n}
  status "single_vendor" when there is no Stooq copy (never a failure),
         "flag" when more than VENDOR_MAX_BAD months differ by more than VENDOR_GAP,
         "ok" otherwise.
vendor_test(h, panel) -> TestResult(test="vendor", kill=True)
  fails only if the discovery window is flagged AND the strategy re-run on Stooq returns flips
  the sign of the mean net return or pushes p_block from < ALPHA to >= ALPHA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import VENDOR_GAP, VENDOR_MAX_BAD, ALPHA, COST_BPS, SPLITS, SEED, N_BOOT
from schemas import TestResult
from data.store import as_ts
from data import pit

_STATUS_RANK = {"ok": 0, "single_vendor": 1, "flag": 2}


def compare_returns(r_a: pd.Series, r_b: pd.Series) -> dict:
    j = pd.concat([r_a, r_b], axis=1).dropna()
    if len(j) == 0:
        return {"status": "single_vendor", "max_gap_pct": None, "n_bad": 0, "vendors_agree": None, "n": 0}
    gap = (j.iloc[:, 0] - j.iloc[:, 1]).abs()
    n_bad = int((gap > VENDOR_GAP).sum())
    status = "flag" if n_bad > VENDOR_MAX_BAD else "ok"
    return {"status": status, "max_gap_pct": round(float(gap.max()) * 100, 3), "n_bad": n_bad,
            "vendors_agree": status == "ok", "n": int(len(j))}


def vendor_check_series(p_yahoo: pd.Series, p_stooq: pd.Series | None, end, window: int = 36,
                        start=None) -> dict:
    if p_stooq is None or p_stooq.dropna().empty:
        return {"status": "single_vendor", "max_gap_pct": None, "n_bad": 0, "vendors_agree": None, "n": 0}
    end = as_ts(end)
    ry = p_yahoo.pct_change(fill_method=None).loc[:end]
    rs = p_stooq.pct_change(fill_method=None).loc[:end]
    j = pd.concat([ry, rs], axis=1).dropna()
    j = j.loc[as_ts(start):] if start is not None else j.iloc[-window:]
    return compare_returns(j.iloc[:, 0], j.iloc[:, 1])


def vendor_check(asset_id: str, end, window: int = 36, start=None) -> dict:
    if asset_id.startswith("BENCH_"):
        return {"status": "single_vendor", "max_gap_pct": None, "n_bad": 0, "vendors_agree": None, "n": 0}
    py = pit.vendor_series(asset_id, "yahoo", end)
    ps = pit.vendor_series(asset_id, "stooq", end)
    if py is None:
        return {"status": "single_vendor", "max_gap_pct": None, "n_bad": 0, "vendors_agree": None, "n": 0}
    return vendor_check_series(py, ps, end, window, start)


def worst(results: list[dict]) -> dict:
    if not results:
        return {"status": "single_vendor", "max_gap_pct": None, "n_bad": 0, "vendors_agree": None, "n": 0}
    w = max(results, key=lambda r: _STATUS_RANK[r["status"]])
    gaps = [r["max_gap_pct"] for r in results if r["max_gap_pct"] is not None]
    out = dict(w)
    out["max_gap_pct"] = max(gaps) if gaps else None
    out["vendors_agree"] = None if w["status"] == "single_vendor" else (w["status"] == "ok")
    return out


def _net_returns(pos: pd.Series, price: pd.Series) -> pd.Series:
    """net_{t+1} = p_t * r_{t+1} - cost * |p_t - p_{t-1}|, indexed by the return date t+1."""
    r = price.pct_change(fill_method=None)
    p = pos.reindex(price.index)
    turn = (p.fillna(0) - p.fillna(0).shift(1).fillna(0)).abs()
    net = p.shift(1) * r - COST_BPS / 1e4 * turn.shift(1)
    return net


def vendor_test(h, panel: pd.DataFrame, *, dates=None) -> TestResult:
    asset = h.spec.asset
    d0, d1 = (dates[0], dates[-1]) if dates is not None and len(dates) else map(as_ts, SPLITS["discovery"])
    chk = vendor_check(asset, d1, start=d0)
    hid = getattr(h, "id", "unknown")
    if chk["status"] == "single_vendor":
        return TestResult(hypothesis_id=hid, test="vendor", passed=True, kill=True, value=None,
                          detail="single vendor", evidence_ref=f"vendor:{asset}")
    if chk["status"] == "ok":
        return TestResult(hypothesis_id=hid, test="vendor", passed=True, kill=True, value=chk["max_gap_pct"],
                          detail=f"Yahoo and Stooq agree ({chk['n_bad']} of {chk['n']} months differ > 1pp)",
                          evidence_ref=f"vendor:{asset}")
    # flagged: re-run the discovery mean on Stooq prices
    from data.features_ref import get_feature_fns
    from monitor.boot import get_p_block
    _, po = get_feature_fns()
    p_block = get_p_block()
    prefix = panel.loc[:d1]
    pos = po(h.spec, prefix)
    py = prefix[asset]
    ps = pit.vendor_series(asset, "stooq", d1).reindex(prefix.index)
    ny = _net_returns(pos, py).loc[d0:d1].dropna()
    ns = _net_returns(pos, ps).loc[d0:d1].dropna()
    my, ms = float(ny.mean()), float(ns.mean())
    pyv, psv = p_block(ny.values, n=N_BOOT, seed=SEED), p_block(ns.values, n=N_BOOT, seed=SEED)
    flips = np.sign(my) != np.sign(ms)
    pushed = (pyv < ALPHA) and (psv >= ALPHA)
    passed = not (flips or pushed)
    detail = (f"vendors disagree in {chk['n_bad']} months; Stooq re-run mean {ms:.4f} vs {my:.4f}, "
              f"p {psv:.3f} vs {pyv:.3f}" + (" -> result depends on vendor" if not passed else " -> robust"))
    return TestResult(hypothesis_id=hid, test="vendor", passed=bool(passed), kill=True, value=float(psv),
                      detail=detail, evidence_ref=f"vendor:{asset}")
