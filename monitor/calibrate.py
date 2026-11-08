"""Calibrate each alarm rule's CUSUM threshold h on stable simulations.

    python -m monitor.calibrate --sims 1000 --out runs/calib.json
    python -m monitor.calibrate --sims 10000 --rules bonds_protect vix_control   # overnight

1. Real joint monthly input rows of the rule over CALIB_WINDOW (e.g. (r_SPX, r_TLT) pairs).
2. mu0, sd0 = mean / std of the REAL strength series over CALIB_WINDOW (locked parameters).
3. Each simulation: stationary-bootstrap the rows (mean block CALIB_BLOCK) to SIM_MONTHS + W,
   compute s_t for 240 months, run the CUSUM with the real mu0/sd0, record max_t C_t.
4. h = quantile(max C, CALIB_QUANTILE).
Reported fa_rate_per_20y = mean number of separate alarm episodes per 240-month simulation.

Work is done in chunks of 500 sims; each chunk has its own seed ([seed, chunk]) and is
checkpointed to runs/calib_ckpt/, so a crashed overnight run resumes where it stopped and gives
identical numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (CALIB_WINDOW, N_CALIB_SIMS, SIM_MONTHS, CALIB_BLOCK, CALIB_QUANTILE, CUSUM_K, SEED,
                    FALSE_ALARM_YEARS)
from data import pit
from data.store import ROOT, as_ts, cache_source
from monitor.boot import stationary_indices
from monitor.cusum import cusum_paths, n_episodes
from monitor.rules import RULES, ALARM_RULES, pairs, rolling_stat, strength_series

CALIB_PATH = ROOT / "runs" / "calib.json"
CKPT_DIR = ROOT / "runs" / "calib_ckpt"
CHUNK = 500


def calib_inputs(rule_id: str, panel: pd.DataFrame | None = None):
    """(rows (T,2) array of real paired inputs in the window, mu0, sd0, window start used)."""
    p = panel if panel is not None else pit.panel(CALIB_WINDOW[1])
    lo, hi = as_ts(CALIB_WINDOW[0]), as_ts(CALIB_WINDOW[1])
    s = strength_series(rule_id, p).loc[lo:hi].dropna()
    if len(s) < 12:
        raise ValueError(f"{rule_id}: too little strength history in calibration window ({len(s)})")
    P = pairs(rule_id, p).loc[lo:hi].dropna()
    return P[["a", "b"]].to_numpy(), float(s.mean()), float(s.std(ddof=1)), str(s.index[0].date())


def simulate_s(rows: np.ndarray, rule_id: str, n: int, rng: np.random.Generator,
               break_at: int | None = None, b_mean: float | None = None) -> np.ndarray:
    """Strength paths (n, SIM_MONTHS). Optional planted break: from sim month `break_at` the second
    input is reflected, r' = 2*mean(r) - r (same mean and volatility, correlation negated)."""
    W, stat = RULES[rule_id]["window"], RULES[rule_id]["stat"]
    L = SIM_MONTHS + W - 1                                  # exactly SIM_MONTHS complete windows
    idx = stationary_indices(len(rows), CALIB_BLOCK, n, rng, length=L)
    A, B = rows[idx, 0], rows[idx, 1].copy()
    if break_at is not None:
        m = rows[:, 1].mean() if b_mean is None else b_mean
        start_row = W - 1 + break_at                        # row of the first post-break window end
        B[:, start_row:] = 2 * m - B[:, start_row:]
    return rolling_stat(A, B, W, stat)                      # (n, SIM_MONTHS)


def _chunk(rule_id, rows, mu0, sd0, seed, ci, size):
    rng = np.random.default_rng([seed, ci])
    S = simulate_s(rows, rule_id, size, rng)
    C = cusum_paths((S - mu0) / sd0, CUSUM_K)
    return C.max(axis=1), C


def calibrate(rule_id: str, n: int = N_CALIB_SIMS, seed: int = SEED, panel: pd.DataFrame | None = None,
              checkpoint: bool = True, quantile: float = CALIB_QUANTILE) -> dict:
    rows, mu0, sd0, wstart = calib_inputs(rule_id, panel)
    source = cache_source() if panel is None else "supplied_panel"
    # Maxima can be reused at a different quantile, but only for identical inputs.
    fingerprint = hashlib.sha256(np.asarray(rows, dtype="<f8").tobytes())
    fingerprint.update(repr((source, mu0, sd0, RULES[rule_id]["window"], RULES[rule_id]["stat"],
                             SIM_MONTHS, CALIB_BLOCK, CUSUM_K, CHUNK)).encode())
    input_id = fingerprint.hexdigest()[:20]
    ck = CKPT_DIR / f"{rule_id}_s{seed}_{input_id}.npz"
    done: dict[int, np.ndarray] = {}
    if checkpoint and ck.exists():
        with np.load(ck) as z:
            done = {int(k[1:]): z[k] for k in z.files}
    n_chunks = int(np.ceil(n / CHUNK))
    maxc_parts = []
    for ci in range(n_chunks):
        size = min(CHUNK, n - ci * CHUNK)
        if ci in done and len(done[ci]) == size:
            mc = done[ci]
        else:
            mc, _ = _chunk(rule_id, rows, mu0, sd0, seed, ci, size)
            done[ci] = mc
            if checkpoint:
                CKPT_DIR.mkdir(parents=True, exist_ok=True)
                tmp = ck.with_suffix(".tmp.npz")
                np.savez(tmp, **{f"c{k}": v for k, v in done.items()})
                tmp.replace(ck)
        maxc_parts.append(mc)
    max_c = np.concatenate(maxc_parts)[:n]
    h = float(np.quantile(max_c, quantile))
    # alarm episodes per simulation need the full paths: regenerate chunk by chunk (same seeds)
    eps = []
    for ci in range(n_chunks):
        size = min(CHUNK, n - ci * CHUNK)
        _, C = _chunk(rule_id, rows, mu0, sd0, seed, ci, size)
        eps.append(n_episodes(C > h))
    eps = np.concatenate(eps)[:n]
    fa = float(eps.mean()) * (FALSE_ALARM_YEARS * 12 / SIM_MONTHS)
    # 95% CI for the false-alarm rate by bootstrap over simulations
    rng = np.random.default_rng(seed)
    bs = eps[rng.integers(0, n, (1000, n))].mean(axis=1)
    return {"rule_id": rule_id, "data_source": source, "input_id": input_id,
            "mu0": mu0, "sd0": sd0, "h": h, "k": CUSUM_K,
            "window": RULES[rule_id]["window"], "calib_window": [wstart, CALIB_WINDOW[1]],
            "n_rows": int(len(rows)), "n_sims": int(n), "seed": int(seed), "quantile": float(quantile),
            "fa_rate_per_20y": fa, "false_alarm_rate_per_20y": fa,
            "fa_rate_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            "p_any_alarm": float((max_c > h).mean()),
            "max_c": [round(float(x), 4) for x in max_c]}


def load_calib(path: Path = CALIB_PATH) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def save_calib(results: dict, path: Path = CALIB_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cur = load_calib(path)
    cur.update(results)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cur))
    tmp.replace(path)


def get_params(rule_id: str, path: Path = CALIB_PATH, n_if_missing: int = N_CALIB_SIMS) -> dict:
    c = load_calib(path).get(rule_id)
    if c is None or c.get("data_source") != cache_source():
        # Keep the selected simulation size and quantile when switching caches.
        n = c.get("n_sims", n_if_missing) if c else n_if_missing
        q = c.get("quantile", CALIB_QUANTILE) if c else CALIB_QUANTILE
        seed = c.get("seed", SEED) if c else SEED
        c = calibrate(rule_id, n=n, seed=seed, quantile=q)
        save_calib({rule_id: c}, path)
    return c


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=N_CALIB_SIMS)
    ap.add_argument("--out", default=str(CALIB_PATH))
    ap.add_argument("--rules", nargs="*", default=ALARM_RULES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--quantile", type=float, default=CALIB_QUANTILE,
                    help="quantile of max C used for h (config default 0.632; 0.368 targets ~1 alarm/20y)")
    a = ap.parse_args(argv)
    for r in a.rules:
        t0 = time.time()
        res = calibrate(r, n=a.sims, seed=a.seed, quantile=a.quantile)
        save_calib({r: res}, Path(a.out))
        print(f"{r:16s} mu0={res['mu0']:+.3f} sd0={res['sd0']:.3f} h={res['h']:.3f} "
              f"false alarms/20y={res['fa_rate_per_20y']:.2f} {res['fa_rate_ci95']} "
              f"P(any alarm in 20y)={res['p_any_alarm']:.3f}  [{time.time() - t0:.1f}s]")


if __name__ == "__main__":
    main()
