"""Benchmark aggregates preserve the per-seed denominator and bootstrap unit."""
from collections import Counter, defaultdict

import numpy as np

from config import SEED

TRUSTED = {"Promising", "Replicated"}


def benchmark_metrics(rows, sidecar=None):
    traps = [(record, raw) for record, raw in rows if record.kind == "trap"]
    by_seed = defaultdict(list)
    grid = defaultdict(Counter)
    strengths = defaultdict(list)
    for row, raw in traps:
        by_seed[row.seed].append((row, raw))
        grid[row.type][row.killed_by or ("Trusted" if row.label in TRUSTED else "No kill recorded")] += 1
        sr = raw.get("planted_sr", raw.get("sr", raw.get("target_sr")))
        if row.truth and sr is not None:
            strengths[float(sr)].append(row.label in TRUSTED)

    def naive(raw):
        for key in ("naive_trusted", "naive_passed"):
            if isinstance(raw.get(key), bool):
                return raw[key]
        p = raw.get("naive_p")
        return p < .05 if p is not None else None

    def measure(which):
        samples = []
        false_total = selected_total = 0
        for group in by_seed.values():
            flags = [which(row, raw) for row, raw in group]
            if any(flag is None for flag in flags):
                return None
            selected = [row for (row, _), flag in zip(group, flags) if flag]
            false = sum(not row.truth for row in selected)
            false_total += false
            selected_total += len(selected)
            samples.append(false / len(selected) if selected else 0.0)
        if not samples:
            return None
        interval = None
        if len(samples) > 1:
            rng = np.random.default_rng(SEED)
            boots = rng.choice(samples, size=(2000, len(samples))).mean(axis=1)
            interval = [float(x) for x in np.quantile(boots, [.025, .975])]
        return {"mean": float(np.mean(samples)), "ci": interval,
                "pooled": false_total / selected_total if selected_total else 0.0}

    result = {
        "ours": measure(lambda row, raw: row.label in TRUSTED),
        "naive": measure(lambda row, raw: naive(raw)),
        "grid": grid,
        "recall": {sr: sum(flags) / len(flags) for sr, flags in strengths.items()},
        "uncertain": sum(row.label == "Can't tell yet" for row, _ in traps),
        "leak_catch": (
            sum(row.killed_by in {"leak_audit", "leakage"} for row, _ in traps if row.type == "leaky")
            / grid["leaky"].total() if grid.get("leaky") else None
        ),
        "seeds": len(by_seed),
        "trials": len(traps),
    }
    if sidecar:
        for key in ("ours", "naive", "recall"):
            if key in sidecar and (not result[key] or key == "recall"):
                result[key] = sidecar[key]
    return result


def alarm_rate(calibration):
    return calibration.get("false_alarm_rate_per_20y", calibration.get("fa_rate_per_20y"))
