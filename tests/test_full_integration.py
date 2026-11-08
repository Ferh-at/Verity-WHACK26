"""Integration checks across the engine, the market cache and the app."""
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import artifacts
from app.metrics import benchmark_metrics
from data import pit
from engine import pipeline, vault
from engine.dsl import make_hypothesis
from schemas import Spec

ROOT = Path(__file__).resolve().parents[1]


def test_decision_panel_matches_publication_dates():
    t = pd.Timestamp("2012-02-29")
    raw, aligned = pit.panel(), pit.decision_panel()
    assert aligned.loc[t, "UNRATE"] == raw["UNRATE"].shift().loc[t]
    assert aligned.loc[t, "UNRATE"] == pit.as_of("UNRATE", t).iloc[-1]
    assert aligned.loc[t, "SPX"] == raw.loc[t, "SPX"]


def test_supplied_benchmark_pairs_naive_and_recovers_recall(monkeypatch, tmp_path):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    rows, demo = artifacts.benchmark()
    assert not demo
    metrics = benchmark_metrics(rows)
    assert metrics["seeds"] == 50 and metrics["trials"] == 10_000
    assert metrics["ours"]["mean"] == pytest.approx(0.003333333333)
    assert metrics["naive"]["mean"] == pytest.approx(0.781340774025)
    assert metrics["recall"][2.0] == pytest.approx(.97)


def test_offline_fixture_batch_runs_engine(monkeypatch, tmp_path):
    monkeypatch.setenv("OFFLINE", "1")
    _, verdicts, info = pipeline.fixture_batch(20, runs_dir=tmp_path)
    assert len(verdicts) == 20 and info["trials"]
    assert (tmp_path / "prereg.jsonl").exists()
    assert (tmp_path / "hurdle.json").exists()


def test_vault_reservation_prevents_retry_after_interruption(monkeypatch, tmp_path):
    h = make_hypothesis(Spec(feature="level", series="T10Y3M", lookback=1,
                             transform="sign", asset="SPX", direction=1), "rule", "Vault")
    v = vault.Vault(tmp_path)
    v.register(h)

    def interrupt(*args, **kwargs):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(vault, "strategy_returns", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        v.open_holdout(h, pd.DataFrame())
    with pytest.raises(vault.VaultError, match="already opened"):
        vault.Vault(tmp_path).open_holdout(h, pd.DataFrame())
