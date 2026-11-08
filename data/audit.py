"""Leak audit (prefix + future-edit) and the "edit the future" demo.

leak_audit(feature_fn, panel) checks, at AUDIT_N_DATES cut dates t:
  (a) prefix:      feature_fn(panel.loc[:t]) == feature_fn(panel) at every date <= t
  (b) future-edit: multiply every value after t by (1 + 0.05*N(0,1)) (seeded), recompute,
                   compare at every date <= t
A difference above tolerance (or a value that only exists when the future is visible) is a leak.

Tolerance is relative: |a-b| > AUDIT_TOL * max(1, |a|), so index levels in the thousands do not
trip on float rounding. Evidence (the first counterexample) goes to runs/audit_evidence.jsonl;
the returned TestResult's evidence_ref points to it.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

import numpy as np
import pandas as pd

from config import AUDIT_N_DATES, AUDIT_TOL, AUDIT_EDIT_SCALE, SEED, SPLITS
from schemas import TestResult
from data.store import ROOT, as_ts

EVIDENCE = ROOT / "runs" / "audit_evidence.jsonl"


def _default_dates(f_full: pd.Series, panel: pd.DataFrame, n: int) -> list[pd.Timestamp]:
    lo, hi = as_ts(SPLITS["discovery"][0]), as_ts(SPLITS["validation"][1])
    valid = f_full.dropna()
    valid = valid[(valid.index >= lo) & (valid.index <= hi) & (valid.index < panel.index.max())]
    if len(valid) == 0:
        valid = f_full.dropna()
        valid = valid[valid.index < panel.index.max()]
    if len(valid) == 0:
        return []
    # skip a warm-up year after the first valid value so trailing windows are full
    first = valid.index[0] + pd.offsets.MonthEnd(12)
    cand = valid[valid.index >= first].index
    if len(cand) == 0:
        cand = valid.index
    pos = np.unique(np.linspace(0, len(cand) - 1, min(n, len(cand))).round().astype(int))
    return list(cand[pos])


def _compare(a: pd.Series, b: pd.Series, t: pd.Timestamp):
    """Return (max_rel_diff, first_bad_date, a_val, b_val) over dates <= t."""
    a = a.loc[:t]
    b = b.reindex(a.index)
    both = a.notna() & b.notna()
    # availability leak: value exists with the full panel but not without the future
    avail = a.notna() & b.isna()
    diff = (a - b).abs() / np.maximum(1.0, a.abs())
    diff = diff.where(both, 0.0)
    worst = float(diff.max()) if len(diff) else 0.0
    bad = diff[diff > AUDIT_TOL]
    if len(bad):
        d = bad.index[0]
        return worst, d, float(a[d]), float(b[d]), "value"
    if avail.any():
        d = avail[avail].index[0]
        return max(worst, np.inf), d, float(a[d]), None, "availability"
    return worst, None, None, None, None


def edit_future(panel: pd.DataFrame, t, scale: float = AUDIT_EDIT_SCALE, seed: int = SEED,
                multiplier: float | None = None) -> pd.DataFrame:
    """Copy of panel with every numeric value after t perturbed (or multiplied by `multiplier`)."""
    t = as_ts(t)
    p = panel.copy()
    after = p.index > t
    num = p.select_dtypes("number").columns
    if multiplier is not None:
        p.loc[after, num] = p.loc[after, num] * multiplier
    else:
        rng = np.random.default_rng(seed)
        noise = 1.0 + scale * rng.standard_normal((int(after.sum()), len(num)))
        p.loc[after, num] = p.loc[after, num].values * noise
    return p


def leak_audit(feature_fn: Callable[[pd.DataFrame], pd.Series], panel: pd.DataFrame,
               dates: list | None = None, seed: int = SEED, hypothesis_id: str = "unknown",
               write_evidence: bool = True, evidence_path=None) -> TestResult:
    f_full = feature_fn(panel)
    dates = [as_ts(d) for d in dates] if dates is not None else _default_dates(f_full, panel, AUDIT_N_DATES)
    max_diff, cex = 0.0, None
    for i, t in enumerate(dates):
        # (a) prefix
        f_pre = feature_fn(panel.loc[:t])
        d, when, a, b, kind = _compare(f_full, f_pre, t)
        max_diff = max(max_diff, d)
        if when is not None and cex is None:
            cex = dict(check="prefix", kind=kind, t=t, feature_date=when, before=a, after=b)
        # (b) future edit
        f_ed = feature_fn(edit_future(panel, t, seed=seed + i))
        d, when, a, b, kind = _compare(f_full, f_ed, t)
        max_diff = max(max_diff, d)
        if when is not None and cex is None:
            cex = dict(check="future_edit", kind=kind, t=t, feature_date=when, before=a, after=b)
        if cex is not None:
            break                      # one counterexample is enough to kill
    leak = cex is not None
    eid = "ae_" + uuid.uuid4().hex[:10]
    if leak:
        detail = (f"leak at feature date {cex['feature_date'].date()} when "
                  f"{'editing data after' if cex['check'] == 'future_edit' else 'cutting data at'} "
                  f"t={cex['t'].date()}" + (" (value only exists with future data)" if cex["kind"] == "availability" else ""))
    else:
        detail = f"no dependence on data after t ({len(dates)} dates)"
    if write_evidence:
        path = EVIDENCE if evidence_path is None else evidence_path
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"id": eid, "hypothesis_id": hypothesis_id, "leak": leak, "n_dates": len(dates),
               "dates": [str(d.date()) for d in dates], "max_rel_diff": None if not np.isfinite(max_diff) else max_diff,
               "ts": datetime.now(timezone.utc).isoformat()}
        if cex:
            rec["counterexample"] = {"check": cex["check"], "kind": cex["kind"], "t": str(cex["t"].date()),
                                     "feature_date": str(cex["feature_date"].date()),
                                     "before": cex["before"], "after": cex["after"],
                                     "edited_range": f"> {cex['t'].date()}"}
        with open(path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    value = float(max_diff) if np.isfinite(max_diff) else 1.0
    return TestResult(hypothesis_id=hypothesis_id, test="leak_audit", passed=not leak, kill=True,
                      value=value, detail=detail, evidence_ref=f"runs/audit_evidence.jsonl#{eid}")


def leak_audit_spec(spec, panel: pd.DataFrame, hypothesis_id: str = "unknown", **kw) -> TestResult:
    """Audit a Spec using engine.features, or the reference copy if that import fails."""
    from data.features_ref import get_feature_fns
    fe, _ = get_feature_fns()
    return leak_audit(lambda p: fe(spec, p), panel, hypothesis_id=hypothesis_id, **kw)


def edit_future_demo(spec, panel: pd.DataFrame, t, multiplier: float = 1.3) -> dict:
    """Multiply every value after t by `multiplier`; recompute the feature and position AT t."""
    from data.features_ref import get_feature_fns
    fe, po = get_feature_fns()
    t = as_ts(t)
    t = panel.index[panel.index <= t][-1]          # snap to the month-end on or before t
    edited = edit_future(panel, t, multiplier=multiplier)

    def at(s):
        v = s.get(t, np.nan)
        return None if pd.isna(v) else float(v)

    before, after = at(fe(spec, panel)), at(fe(spec, edited))
    pb, pa = at(po(spec, panel)), at(po(spec, edited))

    def same(x, y):
        if x is None or y is None:
            return x is None and y is None
        return abs(x - y) <= AUDIT_TOL * max(1.0, abs(x))

    passed = same(before, after) and same(pb, pa)
    text = (f"PASS: the feature on {t.date()} is unchanged after multiplying all data after that month by {multiplier}."
            if passed else
            f"LEAK: on {t.date()} the feature moved from {before} to {after} (position {pb} -> {pa}) "
            f"when only data AFTER {t.date()} was edited x{multiplier}.")
    return {"t": str(t.date()), "multiplier": multiplier, "before": before, "after": after,
            "position_before": pb, "position_after": pa, "passed": passed,
            "edited_range": f"> {t.date()}", "detail": text}
