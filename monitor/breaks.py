"""M2.9 Planted breaks: how fast does the calibrated CUSUM notice a rule flipping?

    python -m monitor.breaks --sims 200 --rules bonds_protect vix_control

Stable simulation as in calibration; at sim month BREAK_MONTH the second input is reflected
(r' = 2*mean(r) - r), which keeps its mean and volatility and negates the relationship.
delay_m = first alarm month at or after BREAK_MONTH minus BREAK_MONTH (None = never).
Alarms before BREAK_MONTH are false alarms. Detected = delay_m <= DETECT_WITHIN.

Writes BenchmarkRun rows (kind="break", and kind="stable" from the same seeds without a break)
to runs/benchmark_monitor.jsonl and the summary to runs/breaks.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import BREAK_MONTH, DETECT_WITHIN, N_BREAK_SIMS, CUSUM_K, SEED
from schemas import BenchmarkRun
from data.store import ROOT
from monitor.calibrate import calib_inputs, simulate_s, get_params
from monitor.cusum import cusum_paths
from monitor.rules import ALARM_RULES

ROWS_PATH = ROOT / "runs" / "benchmark_monitor.jsonl"
SUMMARY_PATH = ROOT / "runs" / "breaks.json"


def break_sims(rule_id: str = "bonds_protect", n: int = N_BREAK_SIMS, seed: int = SEED,
               params: dict | None = None) -> list[BenchmarkRun]:
    rows, _, _, _ = calib_inputs(rule_id)
    c = params or get_params(rule_id)
    mu0, sd0, h = c["mu0"], c["sd0"], c["h"]
    rng = np.random.default_rng([seed, 999])            # independent of calibration seeds
    S = simulate_s(rows, rule_id, n, rng, break_at=BREAK_MONTH)
    alarm = cusum_paths((S - mu0) / sd0, CUSUM_K) > h
    out = []
    for i in range(n):
        pre = bool(alarm[i, :BREAK_MONTH].any())
        post = np.flatnonzero(alarm[i, BREAK_MONTH:])
        delay = int(post[0]) if len(post) else None
        out.append(BenchmarkRun(seed=seed + i, kind="break", type=rule_id, truth=True, label=None,
                                killed_by=None, delay_m=delay, false_alarm=pre))
    return out


def stable_runs(rule_id: str, n: int = N_BREAK_SIMS, seed: int = SEED, params: dict | None = None) -> list[BenchmarkRun]:
    rows, _, _, _ = calib_inputs(rule_id)
    c = params or get_params(rule_id)
    rng = np.random.default_rng([seed, 998])
    S = simulate_s(rows, rule_id, n, rng)
    alarm = cusum_paths((S - c["mu0"]) / c["sd0"], CUSUM_K) > c["h"]
    return [BenchmarkRun(seed=seed + i, kind="stable", type=rule_id, truth=False, label=None, killed_by=None,
                         delay_m=None, false_alarm=bool(alarm[i].any())) for i in range(n)]


def metrics(runs: list[BenchmarkRun]) -> dict:
    d = [r.delay_m for r in runs]
    det = [x for x in d if x is not None]
    within = [x for x in det if x <= DETECT_WITHIN]
    clean = [r for r in runs if not r.false_alarm]                 # no alarm before the break
    clean_within = [r for r in clean if r.delay_m is not None and r.delay_m <= DETECT_WITHIN]
    return {"n": len(runs),
            "median_delay_m": float(np.median(det)) if det else None,
            "median_delay_all_m": float(np.median([x if x is not None else np.inf for x in d])),
            "share_detected_within_24m": len(within) / len(runs),
            "share_detected_ever": len(det) / len(runs),
            "share_false_alarm_before_break": float(np.mean([r.false_alarm for r in runs])),
            "share_within_24m_excluding_prebreak_alarms": (len(clean_within) / len(clean)) if clean else None,
            "delays": d}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_BREAK_SIMS)
    ap.add_argument("--rules", nargs="*", default=["bonds_protect", "vix_control"])
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    summary = json.loads(SUMMARY_PATH.read_text()) if SUMMARY_PATH.exists() else {}
    all_rows = []
    for r in a.rules:
        params = get_params(r)
        br = break_sims(r, a.sims, a.seed, params)
        st = stable_runs(r, a.sims, a.seed, params)
        provenance = {"data_source": params["data_source"], "input_id": params["input_id"],
                      "calibration_h": params["h"]}
        m = {**metrics(br), **provenance}
        m["stable_false_alarm_share"] = float(np.mean([x.false_alarm for x in st]))
        summary[r] = m
        all_rows.extend({**row.model_dump(mode="json"), **provenance} for row in br + st)
        print(f"{r:16s} median delay {m['median_delay_m']} m | detected <=24m {m['share_detected_within_24m']:.0%} "
              f"| ever {m['share_detected_ever']:.0%} | pre-break false alarm {m['share_false_alarm_before_break']:.0%}")
    ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # rewrite rows for the rules we ran, keep the others
    keep = []
    if ROWS_PATH.exists():
        keep = [l for l in ROWS_PATH.read_text().splitlines()
                if l.strip() and json.loads(l)["type"] not in a.rules]
    ROWS_PATH.write_text("\n".join(keep + [json.dumps(row) for row in all_rows]) + "\n")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
