"""One thesis run:
dedupe + pre-register -> leak audit -> discovery stats -> luck hurdle + BY -> objections (always
leakage + outlier, plus the skeptic's) -> validation -> [factor] -> holdout vault -> verdict.

CLI: python -m engine.pipeline --thesis "..." --n 20"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config import (ALPHA, FACTOR_ENABLED, N_BOOT, N_HURDLE, N_HYPOTHESES_DEFAULT, SEED, SPLITS, VALID_T_MIN)
from engine.backtest import default_splits, eligible_net, split_bounds, strategy_returns, asset_availability
from engine.dedupe import cluster_returns
from engine.factor import factor_proxies, factor_test
from engine.hurdle import luck_hurdle
from engine.locks import file_lock
from engine.multitest import by_qvalues, holm_qvalues
from engine.nulls import p_block
from engine.power import dsr as deflated_sharpe, moments, power_line
from engine.stats import nw_t, sharpe
from engine.vault import RUNS_DIR, Vault, VaultError
from engine.verdict import (_tr, by_gate, classify, decide, leakage_test, outlier_test, run_objection,
                            spec_columns)
from schemas import Hypothesis, Objection, TestResult, Thesis, TrialLogEntry, Verdict

ObjectionsFn = Callable[[Hypothesis, dict], list[Objection]]


class Ledger:
    """Everything tried in this session. The hurdle, BY and DSR are computed over all cluster
    representatives in the ledger, so they get stricter as more ideas are tried."""

    def __init__(self):
        self.reps: list[Hypothesis] = []
        self.nets: dict[str, pd.Series] = {}
        self.p_block: dict[str, float] = {}
        self.sr_m: dict[str, float] = {}
        self.verdicts: dict[str, Verdict] = {}
        self.hurdle_cache: dict = {}
        self.n_tried = 0

    @property
    def effective_n(self) -> int:
        return len(self.reps)


SESSION = Ledger()


def _now():
    return datetime.now(timezone.utc)


def disc_summary(net: pd.Series, p: float | None = None) -> dict:
    """Discovery-only summary handed to the skeptic."""
    if len(net) == 0:
        return {"n_obs": 0, "sharpe_net_ann": None, "nw_t": None, "p_block": p, "best_month": None,
                "worst_month": None, "share_of_return_from_top_month": None}
    total = net.sum()
    return {"n_obs": int(len(net)), "sharpe_net_ann": round(sharpe(net), 3), "nw_t": round(nw_t(net), 3),
            "p_block": None if p is None else round(p, 4),
            "best_month": f"{net.idxmax():%Y-%m}", "worst_month": f"{net.idxmin():%Y-%m}",
            "share_of_return_from_top_month": round(float(net.max() / total), 3) if total > 0 else None}


def _trial(h_id, batch_id, split, df, p, dup_of=None, p_search=None, hurdle=None, q=None) -> TrialLogEntry:
    net = eligible_net(df)
    gross = df["gross"][df["pos"].notna()].dropna()
    n = len(net)
    f = lambda v: 0.0 if v is None or not np.isfinite(v) else float(v)
    return TrialLogEntry(trial_id="tr_" + uuid.uuid4().hex[:10], hypothesis_id=h_id, dup_of=dup_of,
                         batch_id=batch_id, split=split, n_obs=n, coverage=n / len(df) if len(df) else 0.0,
                         sharpe_ann=f(sharpe(gross)), sharpe_net_ann=f(sharpe(net)),
                         nw_t=f(nw_t(net)) if n >= 3 else 0.0, p_block=float(p),
                         p_search=p_search, hurdle=hurdle, q_by=q, ts=_now())


def _append(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(r + "\n")


def run_batch(thesis: Thesis | None, hyps: list[Hypothesis], panel: pd.DataFrame,
              objections_fn: ObjectionsFn | None = None, *, splits: dict | None = None,
              ledger: Ledger | None = None, vault: Vault | None = None, runs_dir: Path | None = RUNS_DIR,
              factor_enabled: bool | None = None, proxies: pd.DataFrame | None = None,
              n_boot: int = N_BOOT, n_hurdle: int = N_HURDLE, seed: int = SEED,
              audit: bool = True, audit_n_dates: int | None = None,
              explain_fn: Callable | None = None, info: dict | None = None,
              objections_batch_fn: Callable | None = None) -> list[Verdict]:
    """Test a batch of hypotheses; returns one Verdict per hypothesis (duplicates inherit their
    representative's verdict). Writes trials/tests/verdicts/hurdle to runs_dir unless it is None.
    `info`, if given, is filled with per-batch details (hurdle dict, cluster map, trial entries)."""
    t0 = time.time()
    splits = splits or default_splits()
    ledger = SESSION if ledger is None else ledger
    vault = Vault(runs_dir) if vault is None else vault
    factor_enabled = FACTOR_ENABLED if factor_enabled is None else factor_enabled
    batch_id = "ba_" + uuid.uuid4().hex[:10]
    hyps = sorted(hyps, key=lambda h: h.created_at)
    by_id = {h.id: h for h in hyps}
    ledger.n_tried += len(hyps)

    for h in hyps:
        vault.register(h)

    frames = {h.id: strategy_returns(h.spec, panel[spec_columns(h)], "discovery", splits) for h in hyps}
    nets = {hid: eligible_net(df) for hid, df in frames.items()}
    mapping = cluster_returns([h.id for h in hyps], {**ledger.nets, **nets}, reps=[r.id for r in ledger.reps])
    new_reps = [h for h in hyps if mapping[h.id] == h.id and h.id not in ledger.nets]

    results: dict[str, list[TestResult]] = {}
    pvals: dict[str, float] = {}
    for h in hyps:
        pvals[h.id] = p_block(nets[h.id], n=n_boot, seed=seed)
    for h in new_reps:
        net = nets[h.id]
        res = []
        if audit:
            evidence = Path(runs_dir) / "audit_evidence.jsonl" if runs_dir is not None else None
            res.append(leakage_test(h, panel, splits, n_dates=audit_n_dates, seed=seed,
                                    evidence_path=evidence, write_evidence=runs_dir is not None))
        m = net.mean() if len(net) else float("nan")
        res.append(_tr(h.id, "sign", len(net) > 0 and m > 0, m,
                       f"discovery mean net return {m * 100:.3f}%/month" if len(net) else "no eligible months"))
        p = pvals[h.id]
        res.append(_tr(h.id, "block_null", p < ALPHA, p, f"one-sided p={p:.3f}", kill=False))
        results[h.id] = res
        ledger.reps.append(h)
        ledger.nets[h.id] = net
        ledger.p_block[h.id] = p
        ledger.sr_m[h.id] = moments(net)[0]

    hd = luck_hurdle(ledger.reps, panel, n=n_hurdle, seed=seed, splits=splits, cache=ledger.hurdle_cache)
    q_by = by_qvalues(ledger.p_block)
    q_holm = holm_qvalues(ledger.p_block)
    if factor_enabled and proxies is None:
        proxies = factor_proxies(panel)

    verdicts: dict[str, Verdict] = {}
    trials: list[TrialLogEntry] = []
    batched_objections: dict = {}
    if objections_batch_fn is not None:
        holdout_start = pd.Timestamp(SPLITS["holdout"][0])
        eligible = [(h, disc_summary(nets[h.id], pvals[h.id])) for h in new_reps
                    if len(nets[h.id]) and nets[h.id].index[-1] < holdout_start]
        if eligible:
            try:
                batched_objections = objections_batch_fn(eligible) or {}
            except Exception:
                batched_objections = {}
    for h in new_reps:
        res = results[h.id]
        net, p = nets[h.id], pvals[h.id]
        s_obs, hurdle = hd["observed"][h.id], hd["hurdle"]
        res.append(_tr(h.id, "luck_hurdle", np.isfinite(s_obs) and s_obs > hurdle, s_obs,
                       f"NW t={s_obs:.2f} vs hurdle {hurdle:.2f} (p_search={hd['p_search'][h.id]:.3f}, "
                       f"{ledger.effective_n} effective trials)", kill=False, ref="runs/hurdle.json"))
        res.append(by_gate(q_by[h.id], h.id))
        res.append(outlier_test(net, h.id, base_p=p, n_boot=n_boot, seed=seed))
        if objections_batch_fn is not None:
            incoming = list(batched_objections.get(h.id) or [])
        elif objections_fn is not None and len(net) and net.index[-1] < pd.Timestamp(SPLITS["holdout"][0]):
            incoming = list(objections_fn(h, disc_summary(net, p)) or [])
        else:
            incoming = []
        if incoming:
            done = {"leakage", "outlier"}
            for obj in incoming[:3]:
                if obj.type in done:
                    continue
                done.add(obj.type)
                res.append(run_objection(obj, h, panel, splits, base_p=p, n_boot=n_boot, seed=seed, net=net))
        vdf = strategy_returns(h.spec, panel[spec_columns(h)], "validation", splits)
        vnet = eligible_net(vdf)
        vt = nw_t(vnet) if len(vnet) >= 3 else float("nan")
        vok = len(vnet) >= 3 and vnet.mean() > 0 and vt > VALID_T_MIN
        res.append(_tr(h.id, "validation", vok, vt,
                       f"validation NW t={vt:.2f}, mean {vnet.mean() * 100 if len(vnet) else 0:.3f}%/month, "
                       f"n={len(vnet)} (needs same sign and t > {VALID_T_MIN})", kill=False))
        if factor_enabled:
            if p < ALPHA and len(net) and net.mean() > 0:
                res.append(factor_test(net, proxies, h.id, kill=True))
            else:
                res.append(_tr(h.id, "factor", True, None, "not applicable (no significant discovery result)"))
        power = power_line(net)
        holdout = None
        trials.append(_trial(h.id, batch_id, "discovery", frames[h.id], p, None, hd["p_search"][h.id],
                             hurdle, q_by[h.id]))
        trials.append(_trial(h.id, batch_id, "validation", vdf, 1.0))
        consumed = False
        if classify(res, power, None)[0] == "Promising":
            try:
                holdout, hind = vault.open_holdout_full(h, panel[spec_columns(h)], splits, n_boot=n_boot, seed=seed)
                res.append(hind)
            except VaultError:
                if not vault.is_opened(h):
                    raise
                consumed = True
                res.append(_tr(h.id, "holdout", False, None, "Holdout already consumed in this run directory.",
                               kill=False, ref="runs/vault.jsonl"))
        sr_m, g3, g4 = moments(net)
        d = deflated_sharpe(sr_m, len(net), g3, g4, list(ledger.sr_m.values()), ledger.effective_n) \
            if len(net) >= 4 else None
        as_of = split_bounds("discovery", splits, asset_availability(panel, h.spec.asset))[1].date()
        v = decide(h, res, power, holdout, as_of=as_of, dsr=d,
                   evidence_refs=[f"runs/trials.jsonl#{trials[-2].trial_id}", "runs/hurdle.json"])
        if consumed:
            v.holdout_opened = True
        verdicts[h.id] = v
        ledger.verdicts[h.id] = v

    for h in hyps:
        rep = mapping[h.id]
        if h.id in verdicts:
            continue
        if rep != h.id or h.id not in results:
            trials.append(_trial(h.id, batch_id, "discovery", frames[h.id], pvals[h.id], dup_of=rep))
        base = ledger.verdicts.get(rep)
        if base is None:
            continue
        verdicts[h.id] = base.model_copy(update={"hypothesis_id": h.id,
                                                 "evidence_refs": [f"dup_of:{rep}"] + base.evidence_refs})

    out = [verdicts[h.id] for h in hyps if h.id in verdicts]
    if explain_fn is not None:
        for i, v in enumerate(out):
            try:
                out[i] = v.model_copy(update={"explanation": explain_fn(v, verdict_facts(v))})
            except Exception:
                pass

    if runs_dir is not None:
        runs_dir = Path(runs_dir)
        _append(runs_dir / "hypotheses.jsonl", [h.model_dump_json() for h in hyps])
        _append(runs_dir / "trials.jsonl", [t.model_dump_json() for t in trials])
        _append(runs_dir / "tests.jsonl",
                [json.dumps({"id": f"t_{uuid.uuid4().hex[:8]}", "batch_id": batch_id, **r.model_dump()})
                 for rs in results.values() for r in rs])
        _append(runs_dir / "verdicts.jsonl", [v.model_dump_json() for v in out])
    def number(value):
        return float(value) if value is not None and np.isfinite(value) else None

    hurdle_record = {
        "batch_id": batch_id, "hurdle": number(hd["hurdle"]),
        "null_max": [float(x) for x in hd["null_max"] if np.isfinite(x)],
        "observed": {k: number(v) for k, v in hd["observed"].items()},
        "p_search": hd["p_search"], "q_by": q_by, "q_holm": q_holm,
        "ideas_tried": ledger.n_tried, "effective_trials": ledger.effective_n,
        "duplicates_merged": sum(1 for t in trials if t.dup_of is not None),
        "thesis_id": thesis.id if thesis else None,
    }
    if runs_dir is not None:
        target = runs_dir / "hurdle.json"
        temp = target.with_name(f".hurdle-{batch_id}.tmp")
        temp.write_text(json.dumps(hurdle_record, indent=1, allow_nan=False), encoding="utf-8")
        temp.replace(target)
    if info is not None:
        info.update(hurdle=hd, hurdle_record=hurdle_record, mapping=mapping, trials=trials,
                    q_by=q_by, batch_id=batch_id, seconds=time.time() - t0)
    return out


def verdict_facts(v: Verdict) -> dict:
    """Flat, display-rounded numbers for the explainer."""
    facts = {"label": v.label, "years": f"{v.power.years:.1f}", "mde": f"{v.power.mde_sharpe_80:.2f}"}
    if v.power.mintrl_years is not None:
        facts["mintrl_years"] = f"{v.power.mintrl_years:.1f}"
    if v.dsr is not None:
        facts["dsr"] = f"{v.dsr:.2f}"
    for r in v.reasons:
        if r.value is None:
            continue
        fmt = "{:.3f}" if r.test in ("block_null", "by_fdr", "outlier") else "{:.2f}"
        facts[r.test] = fmt.format(r.value)
    return facts


def fixture_batch(n=20, *, runs_dir=RUNS_DIR / "fixture_engine"):
    """Run bundled hypotheses through the real engine without contacting a provider."""
    if n < 1:
        raise ValueError("At least one fixture hypothesis is required.")
    root = RUNS_DIR.parent / "mocks"
    with (root / "sample_hypotheses.jsonl").open(encoding="utf-8") as source:
        hyps = [Hypothesis.model_validate_json(line) for line in source if line.strip()][:n]
    panel = pd.read_parquet(root / "panel.parquet")
    info = {}
    runs_dir = Path(runs_dir)
    if runs_dir.resolve() == RUNS_DIR.resolve():
        raise ValueError("Use a separate run directory for synthetic holdout reservations.")
    with file_lock(runs_dir / ".fixture_batch.lock"):
        verdicts = run_batch(None, hyps, panel, ledger=Ledger(), runs_dir=runs_dir, info=info)
    return hyps, verdicts, info


def main():
    ap = argparse.ArgumentParser(description="Thesis -> hypotheses -> verdicts")
    ap.add_argument("--thesis", default="An inverted yield curve predicts equity drawdowns")
    ap.add_argument("--n", type=int, default=N_HYPOTHESES_DEFAULT)
    ap.add_argument("--mode", default="thesis", choices=["thesis", "explore"])
    ap.add_argument("--fixtures", action="store_true", help="Run the bundled synthetic hypotheses offline.")
    a = ap.parse_args()
    if a.fixtures:
        _, vs, info = fixture_batch(a.n)
        print(f"{len(vs)} synthetic fixture verdicts in {info['seconds']:.1f}s")
        for v in vs:
            print(f"{v.hypothesis_id}  {v.label}")
        return
    from data.store import using_mock
    if using_mock():
        ap.error("Fetch market data first, or pass --fixtures for an offline run.")
    from agents.generator import generate
    from data.pit import decision_panel as load_panel
    try:
        from agents.skeptic import object_to_many
    except ImportError:
        object_to_many = None
    th = Thesis(id="th_" + uuid.uuid4().hex[:8], text=a.thesis, created_at=_now())
    hyps = generate(th, a.n, a.mode)
    t0 = time.time()
    vs = run_batch(th, hyps, load_panel(), objections_batch_fn=object_to_many)
    print(f"{len(vs)} verdicts in {time.time() - t0:.1f}s (engine time)")
    for v in vs:
        print(f"{v.hypothesis_id}  {v.label:15s}  mde80={v.power.mde_sharpe_80:.2f}  "
              + "; ".join(f"{r.test}:{'ok' if r.passed else 'FAIL'}" for r in v.reasons))


if __name__ == "__main__":
    main()
