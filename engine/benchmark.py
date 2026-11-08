"""Planted-trap benchmark: synthetic data with known real and fake signals, then a measured FDR.

    python -m engine.benchmark --seeds 50 --out runs/benchmark.jsonl
    python -m engine.benchmark --noise-search 200          # pure-noise searches

Rows are BenchmarkRun (one per hypothesis per seed). Real signals are typed "real:<SR>" so recall can be
reported per SR level. The naive pipeline (discovery iid t-test p < ALPHA, nothing else) is written to
<out>_naive.jsonl with label "Promising" (trusted) or "Rejected"; summary numbers go to
runs/benchmark_metrics.json for the Proof page."""
import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import ALPHA, BENCH, N_BOOT, N_HURDLE
from engine.backtest import eligible_net, strategy_returns
from engine.dsl import make_hypothesis
from engine.features import position
from engine.pipeline import Ledger, run_batch
from engine.stats import naive_p
from engine.vault import RUNS_DIR, Vault
from engine.verdict import TRUSTED, killed_by
from schemas import BenchmarkRun, Hypothesis, Spec

B = BENCH
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bench_splits(dates: pd.DatetimeIndex) -> dict:
    d, v = B["disc"], B["val"]
    return {"discovery": (dates[0], dates[d - 1]), "validation": (dates[d], dates[d + v - 1]),
            "holdout": (dates[d + v], dates[B["T"] - 1])}


def make_bench(seed: int, noise_only: bool = False, n_noise: int = 100):
    """Returns (panel, hyps, splits). Feature X_t = ar*X_{t-1} + sqrt(1-ar^2)*eta_t;
    return r_{t+1} = mu + sigma*(ic_t*X_t + sqrt(1-ic_t^2)*eps_{t+1})."""
    rng = np.random.default_rng(seed)
    T, mu, sigma, ar = B["T"], B["mu"], B["sigma"], B["ar"]
    dates = pd.date_range("1990-01-31", periods=T, freq="ME")
    splits = bench_splits(dates)
    cols: dict[str, np.ndarray] = {}
    hyps: list[Hypothesis] = []

    def ar_x():
        eta = rng.standard_normal(T)
        x = np.empty(T)
        x[0] = eta[0]
        for t in range(1, T):
            x[t] = ar * x[t - 1] + math.sqrt(1 - ar ** 2) * eta[t]
        return x

    def rets(x, ic):
        ic = np.broadcast_to(np.asarray(ic, dtype=float), (T,))
        eps = rng.standard_normal(T)
        r = np.full(T, np.nan)
        r[1:] = mu + sigma * (ic[:-1] * x[:-1] + np.sqrt(1 - ic[:-1] ** 2) * eps[1:])
        return r

    def price(r):
        rr = np.nan_to_num(r, nan=0.0)
        return 100 * np.cumprod(1 + rr)

    def add(i, trap, truth, x=None, r=None, spec=None, asset=None):
        if x is not None:
            cols[f"BENCH_X{i}"] = x
        if r is not None:
            cols[f"BENCH_A{i}"] = price(r)
        spec = spec or Spec(feature="level", series=f"BENCH_X{i}", lookback=1, transform="clip_z",
                            asset=asset or f"BENCH_A{i}", direction=1)
        hyps.append(make_hypothesis(spec, "planted", trap, created_at=T0 + timedelta(seconds=i),
                                    truth=truth, trap_type=trap))

    i = 0
    if noise_only:
        for _ in range(n_noise):
            x = ar_x()
            add(i, "noise", False, x, rets(x, 0.0))
            i += 1
        return pd.DataFrame({**cols, "RF": 0.0}, index=dates), hyps, splits

    for j in range(B["real"]):
        sr = B["real_sr"][j % len(B["real_sr"])]
        x = ar_x()
        add(i, f"real:{sr}", True, x, rets(x, sr / math.sqrt(12)))
        i += 1
    for _ in range(B["leaky"]):
        x = ar_x()
        spec = Spec(feature="centered_ma", series=f"BENCH_A{i}", lookback=3, transform="sign",
                    asset=f"BENCH_A{i}", direction=1)
        add(i, "leaky", False, None, rets(x, 0.0), spec=spec)
        i += 1
    for _ in range(B["outlier"]):
        x = ar_x()
        r = rets(x, 0.0)
        spec = Spec(feature="level", series=f"BENCH_X{i}", lookback=1, transform="clip_z",
                    asset=f"BENCH_A{i}", direction=1)
        p = position(spec, pd.DataFrame({f"BENCH_X{i}": x}, index=dates)).to_numpy()
        ok = [m for m in range(B["disc"] - 1) if np.isfinite(p[m]) and p[m] != 0]
        m = int(rng.choice(ok))
        r[m + 1] = mu + B["outlier_sigma"] * sigma * np.sign(p[m])
        add(i, "outlier", False, x, r, spec=spec)
        i += 1
    for _ in range(B["regime"]):
        x = ar_x()
        ic = np.where(np.arange(T) < B["regime_months"], B["regime_ic"], 0.0)
        add(i, "regime", False, x, rets(x, ic))
        i += 1
    mkt = rng.normal(B["beta_mu"], sigma, T)
    mkt[0] = np.nan
    cols["BENCH_MKT"] = price(mkt)
    for _ in range(B["beta"]):
        x = 1 + rng.normal(0, 0.1, T)
        r = mkt + rng.normal(0, 0.01, T)
        spec = Spec(feature="level", series=f"BENCH_X{i}", lookback=1, transform="sign",
                    asset=f"BENCH_A{i}", direction=1)
        add(i, "beta", False, x, r, spec=spec)
        i += 1
    x0 = ar_x()
    dup_asset = f"BENCH_A{i}"
    cols[dup_asset] = price(rets(x0, 0.0))
    for _ in range(B["dup"]):
        add(i, "dup", False, x0 + rng.normal(0, B["dup_noise"], T), asset=dup_asset)
        i += 1
    for _ in range(B["noise"]):
        x = ar_x()
        add(i, "noise", False, x, rets(x, 0.0))
        i += 1
    return pd.DataFrame({**cols, "RF": 0.0}, index=dates), hyps, splits


def run_seed(seed: int, factor: bool = True, noise_only: bool = False, n_noise: int = 100,
             n_boot: int = N_BOOT, n_hurdle: int = N_HURDLE, audit: bool = True) -> dict:
    panel, hyps, splits = make_bench(seed, noise_only, n_noise)
    proxies = None
    if "BENCH_MKT" in panel:
        proxies = pd.DataFrame({"MKT": panel["BENCH_MKT"].pct_change(fill_method=None)})
    t0 = time.time()
    vs = run_batch(None, hyps, panel, None, splits=splits, ledger=Ledger(), vault=Vault(None), runs_dir=None,
                   factor_enabled=factor and proxies is not None, proxies=proxies, n_boot=n_boot,
                   n_hurdle=n_hurdle, seed=seed, audit=audit, audit_n_dates=8)
    vmap = {v.hypothesis_id: v for v in vs}
    rows, naive = [], []
    for h in hyps:
        v = vmap[h.id]
        rows.append(BenchmarkRun(seed=seed, kind="trap", type=h.trap_type, truth=bool(h.truth), label=v.label,
                                 killed_by=killed_by(v), delay_m=None, false_alarm=None))
        net = eligible_net(strategy_returns(h.spec, panel, "discovery", splits))
        ok = naive_p(net) < ALPHA
        naive.append(BenchmarkRun(seed=seed, kind="trap", type=h.trap_type, truth=bool(h.truth),
                                  label="Promising" if ok else "Rejected", killed_by=None if ok else "naive_t",
                                  delay_m=None, false_alarm=None))
    return {"seed": seed, "rows": [r.model_dump() for r in rows], "naive": [r.model_dump() for r in naive],
            "seconds": time.time() - t0}


def _fdr_by_seed(rows: list[dict]) -> dict[int, tuple[int, int]]:
    out = defaultdict(lambda: [0, 0])
    for r in rows:
        s = out[r["seed"]]
        if r["label"] in TRUSTED:
            s[0] += 1
            s[1] += 0 if r["truth"] else 1
    return {k: tuple(v) for k, v in out.items()}


def _fdr_stats(rows: list[dict], n_boot: int = 1000, seed: int = 0) -> dict:
    per = _fdr_by_seed(rows)
    seeds = sorted(per)
    fdr = np.array([f / t if t else 0.0 for t, f in (per[s] for s in seeds)])
    tr = sum(per[s][0] for s in seeds)
    fa = sum(per[s][1] for s in seeds)
    rng = np.random.default_rng(seed)
    boots = [fdr[rng.integers(0, len(fdr), len(fdr))].mean() for _ in range(n_boot)] if len(fdr) else [0]
    return {"fdr_mean": float(fdr.mean()) if len(fdr) else 0.0, "fdr_pooled": fa / tr if tr else 0.0,
            "fdr_lo": float(np.percentile(boots, 2.5)), "fdr_hi": float(np.percentile(boots, 97.5)),
            "trusted": tr, "false_trusted": fa, "n_seeds": len(seeds)}


def bench_metrics(rows: list[dict], naive_rows: list[dict] | None = None) -> dict:
    rows = [r if isinstance(r, dict) else r.model_dump() for r in rows]
    out = {"ours": _fdr_stats(rows)}
    if naive_rows:
        out["naive"] = _fdr_stats([r if isinstance(r, dict) else r.model_dump() for r in naive_rows])
    recall = {}
    for sr in sorted({r["type"].split(":")[1] for r in rows if r["type"].startswith("real:")}, key=float):
        rr = [r for r in rows if r["type"] == f"real:{sr}"]
        recall[sr] = sum(r["label"] in TRUSTED for r in rr) / len(rr)
    out["recall_by_sr"] = recall
    table = defaultdict(Counter)
    for r in rows:
        table[r["type"].split(":")[0]][r["killed_by"] or "trusted"] += 1
    out["killed_by"] = {k: dict(v) for k, v in table.items()}
    out["labels"] = dict(Counter(r["label"] for r in rows))
    out["cant_tell"] = out["labels"].get("Can't tell yet", 0)
    real = [r for r in rows if r["truth"]]
    out["real_wrongly_killed"] = (sum(r["label"] == "Rejected" and r["killed_by"] not in ("block_null", "sign")
                                      for r in real) / len(real)) if real else None
    for trap, tests in [("leaky", ("leak_audit",)), ("outlier", ("outlier", "block_null", "sign")),
                        ("regime", ("regime", "validation")), ("beta", ("factor",))]:
        tr = [r for r in rows if r["type"] == trap]
        if tr:
            out[f"{trap}_killed_by_{'_or_'.join(tests)}"] = sum(r["killed_by"] in tests for r in tr) / len(tr)
            out[f"{trap}_not_trusted"] = sum(r["label"] not in TRUSTED for r in tr) / len(tr)
    dup = [r for r in rows if r["type"] == "dup"]
    if dup:
        per_seed = defaultdict(set)
        for r in dup:
            per_seed[r["seed"]].add((r["label"], r["killed_by"]))
        out["dup_single_verdict_share"] = sum(len(v) == 1 for v in per_seed.values()) / len(per_seed)
    return out


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_bench(seeds=range(50), out: str | Path = RUNS_DIR / "benchmark.jsonl", workers: int | None = None,
              factor: bool = True, resume: bool = False, **kw) -> dict:
    out = Path(out)
    naive_out = out.with_name(out.stem + "_naive.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        for f in (out, naive_out):
            f.unlink(missing_ok=True)
    done = {r["seed"] for r in _read(out)}
    todo = [s for s in seeds if s not in done]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run_seed, s, factor, **kw) for s in todo]
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            with open(out, "a", encoding="utf-8") as fo:
                fo.writelines(json.dumps(r) + "\n" for r in res["rows"])
            with open(naive_out, "a", encoding="utf-8") as fo:
                fo.writelines(json.dumps(r) + "\n" for r in res["naive"])
            print(f"seed {res['seed']} done ({k}/{len(todo)}, {res['seconds']:.1f}s, {time.time() - t0:.0f}s total)",
                  flush=True)
    m = bench_metrics(_read(out), _read(naive_out))
    m["factor_enabled"] = factor
    (out.parent / "benchmark_metrics.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    return m


def noise_search(seeds=range(200), n: int = 100, workers: int | None = None, **kw) -> dict:
    """Must-have 4: share of pure-noise searches (n candidates each) with >= 1 trusted result.
    The leak audit is off by default: it can only reject, so this is a conservative upper bound."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    kw.setdefault("audit", False)
    hits = []
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(run_seed, s, False, True, n, **kw) for s in seeds]
        for k, f in enumerate(as_completed(futs), 1):
            res = f.result()
            hits.append(any(r["label"] in TRUSTED for r in res["rows"]))
            if k % 20 == 0:
                print(f"{k}/{len(futs)} searches, {sum(hits)} with a trusted result, {time.time() - t0:.0f}s",
                      flush=True)
    return {"n_searches": len(hits), "candidates": n, "share_with_trusted": float(np.mean(hits)),
            "leak_audit": kw["audit"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default=str(RUNS_DIR / "benchmark.jsonl"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--no-factor", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--noise-search", type=int, default=0, help="run the pure-noise search test with N seeds")
    a = ap.parse_args()
    if a.noise_search:
        res = noise_search(range(a.noise_search), workers=a.workers)
        (RUNS_DIR / "noise_search.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        print(json.dumps(res, indent=1))
        return
    m = run_bench(range(a.start, a.start + a.seeds), a.out, a.workers, factor=not a.no_factor, resume=a.resume)
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
