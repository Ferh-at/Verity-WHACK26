"""Research session: explore, challenge, then confirm.

explore    discovery split only: leak audit, sign, block null and outlier for each new idea, then the
           luck hurdle and BY recomputed over every idea tried in the session (all rounds). This is the
           only evidence the reviewer ever sees.
challenge  kill-only tests requested by the reviewer (regime, costs, factor, vendor).
confirm    runs once, after the reviewer has finished: validation, the optional factor test and the
           one-shot holdout for nominated ideas, then the verdict for every idea."""
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALPHA, FACTOR_ENABLED, N_BOOT, N_HURDLE, NOMINATE_MAX, SEED, SPLITS, VALID_T_MIN
from engine.backtest import asset_availability, default_splits, eligible_net, split_bounds, strategy_returns
from engine.dedupe import cluster_returns
from engine.factor import factor_proxies, factor_test
from engine.hurdle import luck_hurdle
from engine.multitest import by_qvalues, holm_qvalues
from engine.nulls import p_block
from engine.pipeline import Ledger, _append, _trial
from engine.power import dsr as deflated_sharpe, moments, power_line
from engine.stats import nw_t, sharpe
from engine.vault import RUNS_DIR, Vault, VaultError
from engine.verdict import _tr, by_gate, classify, decide, leakage_test, outlier_test, run_objection, spec_columns
from schemas import Hypothesis, Objection, TestResult, Thesis, TrialLogEntry, Verdict

CHALLENGES = ("regime", "costs", "factor", "vendor")
STATUS_ORDER = ("survives", "fragile", "weak", "killed")


def _num(x):
    return float(x) if x is not None and np.isfinite(x) else None


class ResearchSession:
    """One research run. The ledger covers every idea tried in the run, across all rounds."""

    def __init__(self, panel: pd.DataFrame, thesis: Thesis | None = None, *, splits: dict | None = None,
                 vault: Vault | None = None, runs_dir: Path | None = RUNS_DIR, n_boot: int = N_BOOT,
                 n_hurdle: int = N_HURDLE, seed: int = SEED, factor_enabled: bool | None = None,
                 audit: bool = True, audit_n_dates: int | None = None):
        self.panel, self.thesis = panel, thesis
        self.splits = splits or default_splits()
        self.runs_dir = None if runs_dir is None else Path(runs_dir)
        self.vault = Vault(self.runs_dir) if vault is None else vault
        self.n_boot, self.n_hurdle, self.seed = n_boot, n_hurdle, seed
        self.factor_enabled = FACTOR_ENABLED if factor_enabled is None else factor_enabled
        self.audit, self.audit_n_dates = audit, audit_n_dates
        self.ledger = Ledger()
        self.hyps: dict[str, Hypothesis] = {}
        self.round_of: dict[str, int] = {}
        self.frames: dict[str, pd.DataFrame] = {}
        self.nets: dict[str, pd.Series] = {}
        self.pvals: dict[str, float] = {}
        self.mapping: dict[str, str] = {}
        self.results: dict[str, list[TestResult]] = {}
        self.hurdle = {"hurdle": float("nan"), "null_max": np.array([]), "p_search": {}, "observed": {}}
        self.q_by: dict[str, float] = {}
        self.q_holm: dict[str, float] = {}
        self.round = 0
        self.history: list[dict] = []
        self.batch_id = "ba_" + uuid.uuid4().hex[:10]
        self.confirmed = False

    @property
    def reps(self) -> list[str]:
        return [h.id for h in self.ledger.reps]

    def explore(self, hyps: list[Hypothesis]) -> list[str]:
        """Test new ideas on the discovery split and re-price the search-level gates for all of them."""
        if self.confirmed:
            raise RuntimeError("The session is confirmed; start a new research run.")
        self.round += 1
        new = [h for h in sorted(hyps, key=lambda h: h.created_at) if h.id not in self.hyps]
        for h in new:
            self.vault.register(h)
            self.hyps[h.id] = h
            self.round_of[h.id] = self.round
            df = strategy_returns(h.spec, self.panel[spec_columns(h)], "discovery", self.splits)
            self.frames[h.id] = df
            self.nets[h.id] = eligible_net(df)
            self.pvals[h.id] = p_block(self.nets[h.id], n=self.n_boot, seed=self.seed)
        self.ledger.n_tried += len(new)
        mapping = cluster_returns([h.id for h in new], {**self.ledger.nets, **{h.id: self.nets[h.id] for h in new}},
                                  reps=self.reps)
        self.mapping.update(mapping)
        for h in new:
            if mapping[h.id] != h.id:
                continue
            net, p = self.nets[h.id], self.pvals[h.id]
            res = []
            if self.audit:
                evidence = self.runs_dir / "audit_evidence.jsonl" if self.runs_dir is not None else None
                res.append(leakage_test(h, self.panel, self.splits, n_dates=self.audit_n_dates, seed=self.seed,
                                        evidence_path=evidence, write_evidence=self.runs_dir is not None))
            m = net.mean() if len(net) else float("nan")
            res.append(_tr(h.id, "sign", len(net) > 0 and m > 0, m,
                           f"discovery mean net return {m * 100:.3f}%/month" if len(net) else "no eligible months"))
            res.append(_tr(h.id, "block_null", p < ALPHA, p, f"one-sided p={p:.3f}", kill=False))
            res.append(outlier_test(net, h.id, base_p=p, n_boot=self.n_boot, seed=self.seed))
            self.results[h.id] = res
            self.ledger.reps.append(h)
            self.ledger.nets[h.id] = net
            self.ledger.p_block[h.id] = p
            self.ledger.sr_m[h.id] = moments(net)[0]
        self.hurdle = luck_hurdle(self.ledger.reps, self.panel, n=self.n_hurdle, seed=self.seed,
                                  splits=self.splits, cache=self.ledger.hurdle_cache)
        self.q_by = by_qvalues(self.ledger.p_block)
        self.q_holm = holm_qvalues(self.ledger.p_block)
        self.history.append({"round": self.round, "new": [h.id for h in new], "ideas_tried": self.ledger.n_tried,
                             "effective_trials": self.ledger.effective_n, "hurdle": _num(self.hurdle["hurdle"])})
        return [h.id for h in new]

    def gates(self, hid: str) -> list[TestResult]:
        """Search-level gates at the current ledger size; they tighten as ideas are added."""
        s = self.hurdle["observed"].get(hid, float("nan"))
        bar = self.hurdle["hurdle"]
        ps = self.hurdle["p_search"].get(hid, 1.0)
        return [_tr(hid, "luck_hurdle", np.isfinite(s) and s > bar, s,
                    f"NW t={s:.2f} vs hurdle {bar:.2f} (p_search={ps:.3f}, "
                    f"{self.ledger.effective_n} effective trials)", kill=False, ref="runs/hurdle.json"),
                by_gate(self.q_by.get(hid), hid)]

    def status(self, hid: str) -> tuple[str, list[str]]:
        """(survives | fragile | weak | killed, failed test names) on discovery evidence only."""
        res = self.results[hid] + self.gates(hid)
        failed = [r.test for r in res if not r.passed]
        if any(r.kill and not r.passed for r in res) or "sign" in failed:
            return "killed", failed
        if "block_null" in failed:
            return "weak", failed
        if failed:
            return "fragile", failed
        return "survives", failed

    def evidence(self, hid: str) -> dict:
        """Discovery-only facts for the reviewer prompt (no validation or holdout numbers)."""
        status, failed = self.status(hid)
        net = self.nets[hid]
        out = {"id": hid, "status": status, "failed": ",".join(failed) or "none",
               "observed_t": _num(self.hurdle["observed"].get(hid)), "p_search": _num(self.hurdle["p_search"].get(hid)),
               "q_by": _num(self.q_by.get(hid)), "p_block": _num(self.pvals[hid]), "n_obs": int(len(net)),
               "sharpe_net_ann": _num(sharpe(net)) if len(net) > 2 else None}
        holdout_start = pd.Timestamp(SPLITS["holdout"][0])
        if len(net) and net.index[-1] < holdout_start:
            out["best_month"] = f"{net.idxmax():%Y-%m}"
            out["worst_month"] = f"{net.idxmin():%Y-%m}"
            total = net.sum()
            if total > 0:
                out["share_of_return_from_top_month"] = round(float(net.max() / total), 3)
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if v is not None}

    def challenge(self, objections: list[Objection]) -> list[TestResult]:
        """Run kill-only tests the reviewer asked for. Each test runs at most once per idea."""
        out = []
        for obj in objections:
            hid = self.mapping.get(obj.hypothesis_id, obj.hypothesis_id)
            if hid not in self.results or obj.type not in CHALLENGES:
                continue
            if any(r.test == obj.type for r in self.results[hid]):
                continue
            r = run_objection(obj, self.hyps[hid], self.panel, self.splits, base_p=self.pvals[hid],
                              n_boot=self.n_boot, seed=self.seed, net=self.nets[hid])
            self.results[hid].append(r)
            out.append(r)
        return out

    def confirm(self, nominated: list[str]) -> tuple[list[Verdict], list[TrialLogEntry], dict]:
        """Validation for every idea and the one-shot holdout for nominated ideas that pass every
        earlier gate. Returns (verdicts, trials, hurdle record); writes them under runs_dir."""
        if self.confirmed:
            raise RuntimeError("confirm() runs once per research session.")
        self.confirmed = True
        chosen = list(dict.fromkeys(self.mapping.get(h, h) for h in nominated))[:NOMINATE_MAX]
        proxies = factor_proxies(self.panel) if self.factor_enabled else None
        verdicts: dict[str, Verdict] = {}
        trials: list[TrialLogEntry] = []
        tests: list[TestResult] = []
        for h in self.ledger.reps:
            hid, net, p = h.id, self.nets[h.id], self.pvals[h.id]
            res = list(self.results[hid]) + self.gates(hid)
            vdf = strategy_returns(h.spec, self.panel[spec_columns(h)], "validation", self.splits)
            vnet = eligible_net(vdf)
            vt = nw_t(vnet) if len(vnet) >= 3 else float("nan")
            vok = len(vnet) >= 3 and vnet.mean() > 0 and vt > VALID_T_MIN
            res.append(_tr(hid, "validation", vok, vt,
                           f"validation NW t={vt:.2f}, mean {vnet.mean() * 100 if len(vnet) else 0:.3f}%/month, "
                           f"n={len(vnet)} (needs same sign and t > {VALID_T_MIN})", kill=False))
            if self.factor_enabled and not any(r.test == "factor" for r in res):
                if p < ALPHA and len(net) and net.mean() > 0:
                    res.append(factor_test(net, proxies, hid, kill=True))
                else:
                    res.append(_tr(hid, "factor", True, None, "not applicable (no significant discovery result)"))
            power = power_line(net)
            trials.append(_trial(hid, self.batch_id, "discovery", self.frames[hid], p, None,
                                 self.hurdle["p_search"].get(hid), _num(self.hurdle["hurdle"]), self.q_by.get(hid)))
            trials.append(_trial(hid, self.batch_id, "validation", vdf, 1.0))
            holdout, consumed = None, False
            if hid in chosen and classify(res, power, None)[0] == "Promising":
                try:
                    holdout, hind = self.vault.open_holdout_full(h, self.panel[spec_columns(h)], self.splits,
                                                                 n_boot=self.n_boot, seed=self.seed)
                    res.append(hind)
                except VaultError:
                    if not self.vault.is_opened(h):
                        raise
                    consumed = True
                    res.append(_tr(hid, "holdout", False, None, "Holdout already consumed in this run directory.",
                                   kill=False, ref="runs/vault.jsonl"))
            sr_m, g3, g4 = moments(net)
            d = deflated_sharpe(sr_m, len(net), g3, g4, list(self.ledger.sr_m.values()), self.ledger.effective_n) \
                if len(net) >= 4 else None
            as_of = split_bounds("discovery", self.splits, asset_availability(self.panel, h.spec.asset))[1].date()
            v = decide(h, res, power, holdout, as_of=as_of, dsr=d,
                       evidence_refs=[f"runs/trials.jsonl#{trials[-2].trial_id}", "runs/hurdle.json"])
            if consumed:
                v.holdout_opened = True
            verdicts[hid] = v
            tests.extend(res)
        for hid, h in self.hyps.items():
            if hid in verdicts:
                continue
            rep = self.mapping[hid]
            trials.append(_trial(hid, self.batch_id, "discovery", self.frames[hid], self.pvals[hid], dup_of=rep))
            verdicts[hid] = verdicts[rep].model_copy(update={
                "hypothesis_id": hid, "evidence_refs": [f"dup_of:{rep}"] + verdicts[rep].evidence_refs})
        out = [verdicts[hid] for hid in self.hyps]
        record = self.hurdle_record(chosen)
        if self.runs_dir is not None:
            self._write(out, trials, tests, record)
        return out, trials, record

    def hurdle_record(self, nominated: list[str] | None = None) -> dict:
        return {
            "batch_id": self.batch_id, "hurdle": _num(self.hurdle["hurdle"]),
            "null_max": [float(x) for x in self.hurdle["null_max"] if np.isfinite(x)],
            "observed": {k: _num(v) for k, v in self.hurdle["observed"].items()},
            "p_search": self.hurdle["p_search"], "q_by": self.q_by, "q_holm": self.q_holm,
            "ideas_tried": self.ledger.n_tried, "effective_trials": self.ledger.effective_n,
            "duplicates_merged": sum(1 for hid, rep in self.mapping.items() if rep != hid),
            "rounds": self.history, "nominated": nominated or [],
            "thesis_id": self.thesis.id if self.thesis else None,
        }

    def _write(self, verdicts, trials, tests, record):
        d = self.runs_dir
        _append(d / "hypotheses.jsonl", [h.model_dump_json() for h in self.hyps.values()])
        _append(d / "trials.jsonl", [t.model_dump_json() for t in trials])
        _append(d / "tests.jsonl", [json.dumps({"id": f"t_{uuid.uuid4().hex[:8]}", "batch_id": self.batch_id,
                                                **r.model_dump()}) for r in tests])
        _append(d / "verdicts.jsonl", [v.model_dump_json() for v in verdicts])
        target = d / "hurdle.json"
        temp = target.with_name(f".hurdle-{self.batch_id}.tmp")
        temp.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
        temp.replace(target)
