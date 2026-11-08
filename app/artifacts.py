"""Read the team run contracts and replay immutable demo fixtures when needed."""
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from agents.llm import ROOT
from schemas import BenchmarkRun, Hypothesis, ReplayFrame, TrialLogEntry, Verdict

RUNS = ROOT / "runs"
MOCKS = ROOT / "mocks"


@st.cache_data(show_spinner=False)
def _load(path, mtime, lines):
    with path.open(encoding="utf-8") as file:
        if not lines:
            return json.load(file)
        rows = []
        for line in file:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A writer can be appending the final line as a page renders.
                    if line.endswith("\n"):
                        raise
        return rows


def read(path, default=None):
    try:
        return _load(path, path.stat().st_mtime_ns, path.suffix == ".jsonl")
    except FileNotFoundError:
        return default


def run_file(name, demo=False, default=None):
    path = MOCKS / f"sample_{name}" if demo else RUNS / name
    return read(path, default)


def rows(name, schema, demo=False):
    return [schema.model_validate(row) for row in run_file(name, demo, [])]


@dataclass
class Discovery:
    hypotheses: list[Hypothesis]
    trials: list[TrialLogEntry]
    verdicts: list[Verdict]
    hurdle: dict
    demo: bool
    data_source: str | None = None
    rounds: list = field(default_factory=list)
    lineage: dict = field(default_factory=dict)
    round_of: dict = field(default_factory=dict)
    thesis: str | None = None


def cached_discovery():
    latest = read(RUNS / "latest.json")
    if latest:
        return Discovery(
            [Hypothesis.model_validate(row) for row in latest["hypotheses"]],
            [TrialLogEntry.model_validate(row) for row in latest["trials"]],
            [Verdict.model_validate(row) for row in latest["verdicts"]],
            latest.get("hurdle", {}), False, latest.get("data_source"),
            latest.get("rounds", []), latest.get("lineage", {}), latest.get("round_of", {}),
            latest.get("thesis"),
        )
    hs = rows("hypotheses.jsonl", Hypothesis)
    ts = rows("trials.jsonl", TrialLogEntry)
    vs = rows("verdicts.jsonl", Verdict)
    if hs and ts and vs:
        batch = ts[-1].batch_id
        trials = [row for row in ts if row.batch_id == batch]
        ids = {row.hypothesis_id for row in trials}
        by_id = {row.hypothesis_id: row for row in vs if row.hypothesis_id in ids}
        matched = [h for h in {h.id: h for h in hs}.values() if h.id in ids]
        if matched and all(h.id in by_id for h in matched):
            return Discovery(matched, trials, list(by_id.values()), run_file("hurdle.json", default={}), False)
    return Discovery(
        rows("hypotheses.jsonl", Hypothesis, True),
        rows("trials.jsonl", TrialLogEntry, True),
        rows("verdicts.jsonl", Verdict, True),
        run_file("hurdle.json", True, {}),
        True,
    )


def replay():
    live = rows("frames.jsonl", ReplayFrame)
    demo = not live
    frames = live or rows("frames.jsonl", ReplayFrame, True)
    unique = {(frame.rule_id, frame.as_of): frame for frame in frames}
    return sorted(unique.values(), key=lambda frame: frame.as_of), demo


def benchmark():
    """Read the paired benchmark ledgers, or the packaged measured run."""
    live = [row for row in run_file("benchmark.jsonl", default=[]) if row.get("kind") == "trap"]
    recorded = MOCKS / "recorded_benchmark"
    if live:
        records = live
        naive = run_file("benchmark_naive.jsonl", default=[])
    else:
        records = read(recorded / "benchmark.jsonl", [])
        naive = read(recorded / "benchmark_naive.jsonl", [])
    demo = not records
    if demo:
        records = [row for row in run_file("benchmark.jsonl", True, []) if row.get("kind") == "trap"]

    def identity(row):
        return row["seed"], row["type"], row["truth"]

    paired = defaultdict(deque)
    for row in naive:
        if row.get("kind") == "trap":
            paired[identity(row)].append(row["label"] in {"Promising", "Replicated"})
    output = []
    for row in records:
        raw = dict(row)
        key = identity(raw)
        if paired[key]:
            raw.setdefault("naive_trusted", paired[key].popleft())
        name, separator, sr = raw["type"].partition(":")
        if name == "real" and separator:
            raw.update(type=name, planted_sr=float(sr))
        output.append((BenchmarkRun.model_validate(raw), raw))
    return output, demo


def monitor_benchmark():
    """Read the monitor simulations separately from the planted-trap benchmark."""
    for name in ("benchmark_monitor.jsonl", "breaks.jsonl", "benchmark.jsonl"):
        records = [row for row in run_file(name, default=[]) if row.get("kind") in {"break", "stable"}]
        if records:
            return [(BenchmarkRun.model_validate(row), row) for row in records], False
    records = [row for row in run_file("benchmark.jsonl", True, []) if row.get("kind") in {"break", "stable"}]
    return [(BenchmarkRun.model_validate(row), row) for row in records], True


def calibration():
    live = run_file("calib.json", default={})
    return (live, False) if live else (run_file("calib.json", True, {}), True)


@st.cache_data(show_spinner=False)
def demo_panel():
    return pd.read_parquet(MOCKS / "panel.parquet")
