import json
from datetime import date
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
    monkeypatch.setenv("OFFLINE", "1")
    home = app("app/Home.py")
    assert not home.exception and len(home.metric) == 3
    discover = home.switch_page("pages/1_Discover.py").run()
    assert not discover.exception
    button(discover, "Replay cached run").click().run()
    assert not discover.exception
    assert sum(exp.label == "Raw JSON evidence" for exp in discover.expander) == 20
    assert len(discover.get("plotly_chart")) == 1
    button(discover, "Rerun with edited future").click().run()
    assert any("LEAK" in item.value for item in discover.error)
    audit_options = discover.selectbox[0].options
    discover.selectbox[0].select(audit_options[1].split(" · ")[0]).run()
    button(discover, "Rerun with edited future").click().run()
    assert any("PASS" in item.value for item in discover.success)
    proof = discover.switch_page("pages/2_Proof.py").run()
    assert not proof.exception and len(proof.get("plotly_chart")) == 5
    monitor = proof.switch_page("pages/3_Monitor.py").run()
    assert not monitor.exception and len(monitor.get("plotly_chart")) == 2
    monitor.select_slider[0].set_value(date(2022, 5, 31)).run()
    button(monitor, "Play").click().run()
    assert not monitor.exception
    assert any("FIRST ALARM" in item.value for item in monitor.markdown)
    assert any(item.label == "VIX control" and item.value == "Quiet" for item in monitor.metric)
    for element in monitor.get("plotly_chart"):
        spec = json.loads(element.proto.spec)
        assert all(date.fromisoformat(x[:10]) <= date(2022, 6, 30)
                   for trace in spec["data"] for x in trace["x"])
    button(monitor, "Reset replay").click().run()
    assert not monitor.exception
    assert monitor.select_slider[0].value == date(2015, 1, 31)


def test_live_boundary_and_snapshot_against_team_interfaces(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from agents import explainer, llm
    from app import artifacts, runner
    from app.fixtures import sample_hypotheses, sample_panel
    from schemas import Objection, TrialLogEntry, Verdict

    runs = tmp_path / "runs"
    monkeypatch.setattr(artifacts, "RUNS", runs)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    hyps = sample_hypotheses()
    stages = []
    seen = []

    monkeypatch.setattr(runner, "generate", lambda thesis, n, mode: hyps[:n])
    monkeypatch.setattr(runner, "object_to_many", lambda pairs: {
        h.id: [Objection(hypothesis_id=h.id, type="regime", rationale="Check the subsamples.")]
        for h, _summary in pairs
    })
    monkeypatch.setattr(explainer, "chat_json", lambda *_: (_ for _ in ()).throw(llm.OfflineMiss("not cached")))
    trial_rows = artifacts.rows("trials.jsonl", TrialLogEntry, True)
    expected = artifacts.rows("verdicts.jsonl", Verdict, True)

    def run_batch(thesis, hypotheses, panel, objections_fn=None, *, info, objections_batch_fn=None):
        assert thesis.text == "Test the yield curve"
        assert len(hypotheses) == 20 and "SPX" in panel
        assert objections_fn is None and objections_batch_fn is not None
        seen.extend(objections_batch_fn([(h, {"p_block": .03}) for h in hypotheses]))
        for row in trial_rows:
            llm.append_jsonl(runs / "trials.jsonl", row.model_dump(mode="json"))
        llm.write_json(runs / "hurdle.json", {"null_max": [2.0, 2.5], "hurdle": 2.5})
        info.update(trials=trial_rows, hurdle_record={"null_max": [2.0, 2.5], "hurdle": 2.5})
        return expected

    interfaces = {
        "engine.pipeline": SimpleNamespace(run_batch=run_batch),
        "data.pit": SimpleNamespace(panel=sample_panel),
    }
    monkeypatch.setattr(runner, "require", interfaces.__getitem__)
    monkeypatch.setattr(runner, "active_data_source", lambda: "market_cache")
    result = runner.live_discovery("Test the yield curve", 20, "thesis", stages.append)
    assert len(seen) == len(result.verdicts) == 20
    assert all(v.explanation for v in result.verdicts)
    assert result.hurdle["hurdle"] == 2.5 and len(stages) == 3
    assert not artifacts.cached_discovery().demo


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
