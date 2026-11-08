from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app(path):
    return AppTest.from_file(str(ROOT / path), default_timeout=10).run()


def button(at, label):
    return next(widget for widget in at.button if widget.label == label)


@pytest.mark.parametrize("rehearsal", [1, 2])
def test_offline_full_click_path(monkeypatch, rehearsal):
    from app.artifacts import cached_discovery

    monkeypatch.setenv("OFFLINE", "1")
    home = app("app/Home.py")
    assert not home.exception and len(home.metric) == 3
    research = home.switch_page("pages/1_Research.py").run()
    assert not research.exception
    button(research, "Replay last run").click().run()
    assert not research.exception
    expected = len(cached_discovery().verdicts)
    assert sum(exp.label == "Raw JSON evidence" for exp in research.expander) == expected > 0
    assert len(research.get("plotly_chart")) >= 1
    button(research, "Rerun with edited future").click().run()
    assert any("LEAK" in item.value for item in research.error)
    audit_options = research.selectbox[0].options
    research.selectbox[0].select(audit_options[1].split(" · ")[0]).run()
    button(research, "Rerun with edited future").click().run()
    assert any("PASS" in item.value for item in research.success)
    proof = research.switch_page("pages/2_Proof.py").run()
    assert not proof.exception and len(proof.get("plotly_chart")) == 2
    sources = proof.switch_page("pages/3_Data_and_methods.py").run()
    assert not sources.exception and len(sources.dataframe) >= 3


def test_live_research_loop_and_snapshot(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from agents import researcher
    from agents.llm import assert_no_holdout
    from agents.reviewer import DraftReview, accept, prompt
    from app import artifacts, runner
    from app.fixtures import sample_hypotheses, sample_panel

    monkeypatch.setattr(artifacts, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    hyps = sample_hypotheses()
    stages, prompts = [], []

    def review(thesis, session, round_no, max_rounds, budget):
        system, user = prompt(thesis, session, round_no, max_rounds, budget)
        assert_no_holdout(system)
        assert_no_holdout(user)
        prompts.append(user)
        refine = [{"based_on": session.reps[0], "because": "luck_hurdle", "rationale": "Trend persistence.",
                   "spec": {"feature": "mom", "series": "SPX", "series2": None, "lookback": 12,
                            "transform": "sign", "asset": "SPX", "direction": 1}}]
        draft = DraftReview(assessment="Most ideas do not beat luck.", decision="refine", refinements=refine,
                            challenges=[{"hypothesis_id": session.reps[0], "type": "regime", "rationale": "Split."}])
        return accept(draft, session, round_no, max_rounds, budget, thesis.id)

    monkeypatch.setattr(researcher, "generate", lambda thesis, n, mode: hyps[:n])
    monkeypatch.setattr(researcher, "review", review)
    monkeypatch.setattr(runner, "require", {"data.pit": SimpleNamespace(panel=sample_panel)}.__getitem__)
    monkeypatch.setattr(runner, "active_data_source", lambda: "market_cache")
    result = runner.live_research("Test the yield curve", 20, stages.append)
    assert [r["round"] for r in result.rounds] == [1, 2]
    assert result.rounds[0]["refinements"][0]["based_on"] == hyps[0].id
    assert result.rounds[1]["ideas_tried"] == 21 and result.rounds[1]["hurdle"] >= result.rounds[0]["hurdle"]
    assert len(result.verdicts) == 21 and all(v.explanation for v in result.verdicts)
    assert all("validation" not in p and "holdout" not in p.lower() for p in prompts)
    replay = artifacts.cached_discovery()
    assert not replay.demo and replay.rounds == result.rounds and replay.thesis == "Test the yield curve"


def test_fdr_weights_seeds_and_keeps_missing_naive_visible():
    from app.metrics import benchmark_metrics
    from schemas import BenchmarkRun

    def trap(seed, truth, label, naive=None):
        record = BenchmarkRun(seed=seed, kind="trap", type="real" if truth else "noise",
                              truth=truth, label=label, killed_by=None, delay_m=None, false_alarm=None)
        return record, {"naive_trusted": naive}

    data = [trap(1, False, "Promising", True), trap(2, True, "Replicated", True),
            trap(2, False, "Fragile", True)]
    metrics = benchmark_metrics(data)
    assert metrics["ours"] == {"mean": .5, "ci": [0, 1], "pooled": .5}
    assert metrics["naive"]["mean"] == .75
    assert metrics["naive"]["pooled"] == pytest.approx(2 / 3)
    assert benchmark_metrics(data + [trap(2, False, "Rejected")])["naive"] is None
