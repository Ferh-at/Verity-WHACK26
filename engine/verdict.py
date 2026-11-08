"""Kill tests requested by the skeptic, and the rules that turn those tests into a verdict.

The outlier and regime tests ask whether a discovery result is driven by a few months or one
sub-period. They only apply when there is a significant positive discovery result
(p_block < ALPHA, mean > 0); otherwise they pass as not applicable. Without that limit, noise
would be killed by them and a "Can't tell yet" verdict could never occur."""
from datetime import date

import numpy as np
import pandas as pd

from config import ALPHA, CANT_TELL_MDE, COST_STRESS, FDR_Q, N_BOOT, OUTLIER_DROP_FRAC, SEED, SPLITS
from engine.backtest import asset_availability, eligible_net, split_bounds, strategy_returns
from engine.features import feature
from engine.nulls import p_block
from engine.stats import sharpe
from schemas import Hypothesis, Objection, Power, TestResult, Verdict

TRUSTED = ("Promising", "Replicated")
GATES = ("luck_hurdle", "by_fdr", "validation")


def _tr(h_id, test, passed, value, detail, kill=True, ref=None):
    return TestResult(hypothesis_id=h_id, test=test, passed=bool(passed), kill=kill,
                      value=None if value is None or pd.isna(value) else float(value),
                      detail=detail, evidence_ref=ref or f"runs/tests.jsonl#{test}")


def _applicable(net: pd.Series, base_p: float | None, n_boot: int, seed: int) -> tuple[bool, float]:
    p = p_block(net, n=n_boot, seed=seed) if base_p is None else base_p
    return (len(net) >= 8 and net.mean() > 0 and p < ALPHA), p


def outlier_test(net: pd.Series, hypothesis_id: str = "", base_p: float | None = None,
                 n_boot: int = N_BOOT, seed: int = SEED) -> TestResult:
    """Drop the top OUTLIER_DROP_FRAC months by |net| (at least 1), and separately leave each calendar
    year out. Kill if any version has p_block >= ALPHA or a non-positive mean."""
    ok, p0 = _applicable(net, base_p, n_boot, seed)
    if not ok:
        return _tr(hypothesis_id, "outlier", True, p0, "not applicable (no significant discovery result)")
    k = max(1, int(np.ceil(OUTLIER_DROP_FRAC * len(net))))
    trimmed = net.drop(net.abs().nlargest(k).index)
    versions = [(f"top {k} month(s) dropped", trimmed)]
    versions += [(f"{y} left out", net[net.index.year != y]) for y in sorted(set(net.index.year))]
    worst_p, worst_name = 0.0, ""
    for name, x in versions:
        p = p_block(x, n=n_boot, seed=seed)
        if x.mean() <= 0 or p >= ALPHA:
            return _tr(hypothesis_id, "outlier", False, p,
                       f"fails with {name}: mean {x.mean() * 100:.2f}%/m, p={p:.3f}")
        if p > worst_p:
            worst_p, worst_name = p, name
    return _tr(hypothesis_id, "outlier", True, worst_p,
               f"robust: worst case {worst_name}, p={worst_p:.3f} ({len(versions)} versions)")


def regime_test(net: pd.Series, hypothesis_id: str = "", base_p: float | None = None,
                n_boot: int = N_BOOT, seed: int = SEED) -> TestResult:
    """Split discovery into three chronological thirds; kill if net Sharpe <= 0 in >= 2 of them."""
    ok, p0 = _applicable(net, base_p, n_boot, seed)
    if not ok:
        return _tr(hypothesis_id, "regime", True, p0, "not applicable (no significant discovery result)")
    thirds = np.array_split(net.to_numpy(), 3)
    srs = [sharpe(t) for t in thirds]
    bad = sum(1 for s in srs if not (s > 0))
    txt = ", ".join("n/a" if pd.isna(s) else f"{s:.2f}" for s in srs)
    return _tr(hypothesis_id, "regime", bad < 2, float(bad),
               f"net Sharpe by third of discovery: {txt} ({bad} of 3 <= 0)")


def cost_test(net: pd.Series, gross: pd.Series, hypothesis_id: str = "", base_p: float | None = None,
              n_boot: int = N_BOOT, seed: int = SEED, multiple: float = COST_STRESS) -> TestResult:
    """Re-test with trading costs scaled by `multiple`; kill if the mean turns non-positive or
    p_block >= ALPHA. net_k = gross - k * (gross - net)."""
    ok, p0 = _applicable(net, base_p, n_boot, seed)
    if not ok:
        return _tr(hypothesis_id, "costs", True, p0, "not applicable (no significant discovery result)")
    stressed = gross - multiple * (gross - net)
    p = p_block(stressed, n=n_boot, seed=seed)
    passed = stressed.mean() > 0 and p < ALPHA
    return _tr(hypothesis_id, "costs", passed, p,
               f"at {multiple:g}x costs: mean {stressed.mean() * 100:.3f}%/month, p={p:.3f}")


def audit_dates(panel: pd.DataFrame, splits: dict | None = None, n: int | None = None) -> list:
    """Evenly spaced month-ends across discovery and validation (after a 36-month warm-up)."""
    from config import AUDIT_N_DATES
    s = splits or SPLITS
    lo = pd.Timestamp(s["discovery"][0]) + pd.offsets.MonthEnd(36)
    hi = pd.Timestamp(s["validation"][1])
    idx = panel.index[(panel.index >= lo) & (panel.index <= hi)]
    if len(idx) == 0:
        idx = panel.index
    k = min(n or AUDIT_N_DATES, len(idx))
    return list(idx[np.linspace(0, len(idx) - 1, k).round().astype(int)])


def spec_columns(h: Hypothesis) -> list[str]:
    cols = [h.spec.series, h.spec.asset] + ([h.spec.series2] if h.spec.series2 else [])
    return list(dict.fromkeys(cols))


def leakage_test(h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None,
                 n_dates: int | None = None, seed: int = SEED, *, evidence_path=None,
                 write_evidence=True) -> TestResult:
    from data.audit import leak_audit
    availability = asset_availability(panel, h.spec.asset)
    windows = {name: split_bounds(name, splits, availability) for name in ("discovery", "validation")}
    sub = panel.loc[:windows["validation"][1], spec_columns(h)]
    res = leak_audit(lambda pn: feature(h.spec, pn), sub, dates=audit_dates(sub, windows, n_dates),
                     seed=seed, hypothesis_id=h.id, evidence_path=evidence_path, write_evidence=write_evidence)
    return res.model_copy(update={"hypothesis_id": h.id, "kill": True})


def run_objection(obj: Objection, h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None,
                  base_p: float | None = None, n_boot: int = N_BOOT, seed: int = SEED,
                  net: pd.Series | None = None) -> TestResult:
    """Turn a skeptic objection into a kill-only test."""
    if obj.type == "leakage":
        return leakage_test(h, panel, splits, seed=seed)
    if obj.type == "vendor":
        from data.vendor import vendor_test
        d0, d1 = split_bounds("discovery", splits, asset_availability(panel, h.spec.asset))
        return vendor_test(h, panel.loc[:d1], dates=panel.loc[d0:d1].index)
    if obj.type == "hindsight":
        return _tr(h.id, "hindsight", True, None, "recorded; checked inside the holdout (post-cutoff months)",
                   kill=False, ref="runs/vault.jsonl")
    if obj.type == "costs":
        df = strategy_returns(h.spec, panel[spec_columns(h)], "discovery", splits)
        e = df[df["pos"].notna() & df["net"].notna()]
        return cost_test(e["net"], e["gross"], h.id, base_p=base_p, n_boot=n_boot, seed=seed)
    if net is None:
        net = eligible_net(strategy_returns(h.spec, panel, "discovery", splits))
    if obj.type == "factor":
        from engine.factor import factor_proxies, factor_test
        ok, _ = _applicable(net, base_p, n_boot, seed)
        if not ok:
            return _tr(h.id, "factor", True, None, "not applicable (no significant discovery result)")
        return factor_test(net, factor_proxies(panel), h.id, kill=True)
    fn = outlier_test if obj.type == "outlier" else regime_test
    return fn(net, h.id, base_p=base_p, n_boot=n_boot, seed=seed)


def _get(results: list[TestResult], name: str) -> TestResult | None:
    return next((r for r in results if r.test == name), None)


def classify(results: list[TestResult], power: Power, holdout: TestResult | None) -> tuple[str, str | None]:
    """Verdict rules in order. Returns (label, killed_by = first failing test, None if trusted)."""
    for r in results:
        if r.kill and not r.passed:
            return "Rejected", r.test
    sign = _get(results, "sign")
    if sign is not None and not sign.passed:
        return "Rejected", "sign"
    bn = _get(results, "block_null")
    p = bn.value if bn is not None and bn.value is not None else 1.0
    if p >= ALPHA:
        return ("Rejected" if power.mde_sharpe_80 <= CANT_TELL_MDE else "Can't tell yet"), "block_null"
    for g in GATES:
        r = _get(results, g)
        if r is None or not r.passed:
            return "Fragile", g
    if holdout is not None and holdout.passed:
        return "Replicated", None
    return "Promising", None


def decide(h: Hypothesis, results: list[TestResult], power: Power, holdout: TestResult | None,
           as_of: date | None = None, dsr: float | None = None,
           evidence_refs: list[str] | None = None) -> Verdict:
    label, _ = classify(results, power, holdout)
    reasons = list(results) + ([holdout] if holdout is not None else [])
    return Verdict(hypothesis_id=h.id, label=label, reasons=reasons, power=power,
                   holdout_opened=holdout is not None,
                   holdout_passed=None if holdout is None else holdout.passed,
                   dsr=dsr, evidence_refs=evidence_refs or [], explanation=None,
                   as_of=as_of or pd.Timestamp(SPLITS["discovery"][1]).date())


def killed_by(v: Verdict) -> str | None:
    holdout = _get(v.reasons, "holdout")
    rest = [r for r in v.reasons if r.test != "holdout"]
    return classify(rest, v.power, holdout)[1]


def by_gate(q: float | None, hypothesis_id: str = "") -> TestResult:
    passed = q is not None and q < FDR_Q
    return _tr(hypothesis_id, "by_fdr", passed, q,
               f"BY q={q:.3f} ({'<' if passed else '>='} {FDR_Q})" if q is not None else "BY q n/a", kill=False)
