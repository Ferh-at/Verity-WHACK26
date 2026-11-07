"""Regenerate the bundled, explicitly synthetic demo. Run with python -m mocks.build."""
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from agents import probe
from agents.llm import key_for
from app.fixtures import sample_hypotheses, sample_panel, sample_replay
from config import SEED
from schemas import BenchmarkRun, Power, TestResult, TrialLogEntry, Verdict

OUT = Path(__file__).resolve().parent
FIXTURE_TS = datetime(2026, 11, 7, 13, 45, tzinfo=timezone.utc)


def jsonl(name, rows):
    (OUT / f"sample_{name}.jsonl").write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in rows), encoding="utf-8"
    )


def main():
    rng = np.random.default_rng(SEED)
    hyps = [h.model_copy(update={"created_at": FIXTURE_TS}) for h in sample_hypotheses()]
    trials, verdicts = [], []
    labels = ["Fragile", "Can't tell yet", "Rejected", "Replicated", "Promising"]
    null_max = rng.normal(2.05, .53, 200)
    hurdle = float(np.quantile(null_max, .95))
    observed = {}
    for i, h in enumerate(hyps):
        label = labels[i] if i < 5 else labels[i % 3]
        dup_of = hyps[i - 1].id if i in (8, 15) else None
        t = {"Rejected": -.34, "Can't tell yet": .61, "Fragile": 1.92,
             "Promising": 3.30, "Replicated": 3.62}[label] + rng.normal(0, .10)
        p = {"Rejected": .61, "Can't tell yet": .27, "Fragile": .03,
             "Promising": .001, "Replicated": .001}[label]
        q = {"Rejected": .99, "Can't tell yet": .91, "Fragile": .25,
             "Promising": .04, "Replicated": .04}[label]
        sharpe = t / 3.1
        mde = 1.09 if label == "Can't tell yet" else .73
        opened = label == "Replicated"
        trial_id = f"tr_fixture{i:03d}"
        observed[h.id] = t
        trial = TrialLogEntry(
            trial_id=trial_id, hypothesis_id=h.id, dup_of=dup_of, batch_id="m3_fixture",
            split="discovery", n_obs=139, coverage=.972, sharpe_ann=sharpe+.04,
            sharpe_net_ann=sharpe, nw_t=t, p_block=p,
            p_search=float((1 + (null_max >= t).sum()) / (len(null_max)+1)),
            hurdle=hurdle, q_by=q, ts=FIXTURE_TS,
        )
        trials.append(trial.model_dump(mode="json"))
        tests = [
            ("leak_audit", True, True, 0, "Future edits left the feature unchanged."),
            ("block_null", p < .05, False, p, f"One-sided block-bootstrap p = {p:.3f}."),
            ("luck_hurdle", t > hurdle, False, hurdle, f"Observed t = {t:.2f}; hurdle = {hurdle:.2f}."),
            ("by_fdr", q < .10, False, q, f"BY-adjusted q = {q:.3f}."),
        ]
        if label == "Rejected":
            tests.append(("outlier", False, True, .18, "Result did not survive the outlier test."))
        else:
            tests.append(("outlier", True, True, p, "Removing extreme months did not change this gate."))
        if label in {"Promising", "Replicated"}:
            tests.append(("validation", True, False, 1.42, "Same sign and validation t cleared the gate."))
        if opened:
            tests.append(("holdout", True, False, .68, "Holdout passed with positive net Sharpe."))
            trials.append(trial.model_copy(update={
                "split": "post_cutoff", "trial_id": f"tr_post{i:03d}", "sharpe_net_ann": .61,
            }).model_dump(mode="json"))
        refs = [f"runs/trials.jsonl#{trial_id}"]
        reasons = [
            TestResult(hypothesis_id=h.id, test=name, passed=passed, kill=kill, value=value,
                       detail=detail, evidence_ref=f"fixture:tests/{h.id}/{name}")
            for name, passed, kill, value, detail in tests
        ]
        verdict = Verdict(
            hypothesis_id=h.id, label=label, reasons=reasons,
            power=Power(years=139/12, mde_sharpe_80=mde, mintrl_years=round(13/max(.4, sharpe), 2)
                        if sharpe > 0 else None, power_at_05=.43),
            holdout_opened=opened, holdout_passed=True if opened else None,
            dsr=.93 if label in {"Promising", "Replicated"} else None,
            evidence_refs=refs, explanation=None, as_of=date(2026, 10, 31) if opened else date(2019, 12, 31),
        )
        verdicts.append(verdict.model_dump(mode="json"))
    jsonl("hypotheses", [h.model_dump(mode="json") for h in hyps])
    jsonl("trials", trials)
    jsonl("verdicts", verdicts)
    jsonl("frames", [frame.model_dump(mode="json") for frame in sample_replay()])
    panel = sample_panel()
    panel.to_parquet(OUT / "panel.parquet")
    benchmark = []
    traps = ["real"] * 10 + ["leaky"] * 10 + ["outlier"] * 10 + ["regime"] * 10 + ["beta"] * 10 + ["dup"] * 20 + ["noise"] * 130
    for i, trap in enumerate(traps):
        real = trap == "real"
        sr = [0.5, 1.0, 1.5, 2.0][i % 4] if real else None
        trusted = bool((real and rng.random() < sr / 2.4) or i == 163)
        naive = bool(real or rng.random() < (.16 if trap == "noise" else .8))
        killed = {"leaky": "leak_audit", "outlier": "outlier", "regime": "regime",
                  "beta": "factor", "dup": "duplicate", "noise": "by_fdr"}.get(trap, "block_null")
        row = BenchmarkRun(
            seed=SEED, kind="trap", type=trap, truth=real,
            label="Promising" if trusted else ("Can't tell yet" if i % 7 == 0 else "Rejected"),
            killed_by=None if trusted else killed, delay_m=None, false_alarm=None,
        )
        benchmark.append({**row.model_dump(), "naive_trusted": naive, "planted_sr": sr})
    for rule in ("bonds_protect", "vix_control"):
        for i in range(200):
            delay = int(rng.integers(5, 33)) if rng.random() < .93 else None
            benchmark.append(BenchmarkRun(
                seed=i, kind="break", type=rule, truth=True, label=None,
                killed_by=None, delay_m=delay, false_alarm=bool(rng.random() < .11),
            ).model_dump())
    jsonl("benchmark", benchmark)
    cal = {
        name: {"mu0": .38, "sd0": .10, "h": 9.2, "false_alarm_rate_per_20y": .97,
               "max_c": rng.gamma(3.3, 2.8, 1000).tolist()}
        for name in ("bonds_protect", "vix_control", "dollar_haven", "gold_realyield", "btc_diversifier")
    }
    (OUT / "sample_calib.json").write_text(json.dumps(cal), encoding="utf-8")
    (OUT / "sample_hurdle.json").write_text(json.dumps({
        "hurdle": hurdle, "null_max": null_max.tolist(), "observed": observed,
    }), encoding="utf-8")

    cache = OUT / "probe_cache"
    truth = OUT / "probe_truth"
    cache.mkdir(exist_ok=True)
    truth.mkdir(exist_ok=True)
    for name, (_, source, _) in probe.SERIES.items():
        series = panel[name].rename("value")
        pd.DataFrame({"date": series.index, "published": series.index,
                      "value": series.values}).to_csv(truth / f"{source}_{name}.csv", index=False)
        actual = probe.truth_series(name, truth)
        pre = pd.PeriodIndex(probe.PRE, freq="M").to_timestamp(how="end").normalize()
        post = actual.index[actual.index > pd.Timestamp(probe.LLM_CUTOFF)][-10:]
        for stamp in pre.append(post):
            user = probe.prompt(name, stamp)
            model = "gpt-4o-mini"
            key = key_for(model, probe.SYSTEM, user, 0.0)
            error = .012 if stamp in pre else .095
            response = {"value": float(actual[stamp]) * (1 + rng.uniform(-error, error))}
            (cache / f"{key}.json").write_text(json.dumps({
                "request": {"system": probe.SYSTEM, "user": user, "temperature": 0.0},
                "response": response, "model": model, "ts": FIXTURE_TS.isoformat(),
            }), encoding="utf-8")


if __name__ == "__main__":
    main()
