"""Checks for the artefact boundary and synthetic-data separation."""
import json
from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app import artifacts, runner
from data import pit
from monitor import calibrate as cal

ROOT = Path(__file__).resolve().parents[1]


def write_lines(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_proof_reads_m2_breaks_without_m1_results(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    monkeypatch.setenv("OFFLINE", "1")
    raw = {
        "type": "bonds_protect", "label": None, "killed_by": None,
        "data_source": "synthetic_cache", "input_id": "integration", "calibration_h": 7.5,
    }
    write_lines(tmp_path / "benchmark_monitor.jsonl", [
        {**raw, "seed": 1, "kind": "break", "truth": True, "delay_m": 11, "false_alarm": False},
        {**raw, "seed": 2, "kind": "break", "truth": True, "delay_m": None, "false_alarm": False},
        {**raw, "seed": 1, "kind": "stable", "truth": False, "delay_m": None, "false_alarm": True},
    ])
    (tmp_path / "calib.json").write_text(json.dumps({
        "bonds_protect": {"data_source": "synthetic_cache", "input_id": "integration",
                          "h": 7.5, "fa_rate_per_20y": .47, "max_c": [1, 8], "quantile": .632}
    }))
    at = AppTest.from_file(str(ROOT / "app/pages/2_Proof.py")).run()
    assert not at.exception
    assert any(m.label == "Detected within 24 months" and m.value == "50%" for m in at.metric)
    assert any("Synthetic cache" in w.value for w in at.warning)
    assert any("1 separate stable simulations" in c.value for c in at.caption)
    benchmark_rows, _ = artifacts.benchmark()
    assert all(row.kind == "trap" for row, _ in benchmark_rows)
    assert artifacts.monitor_benchmark()[1] is False


def test_monitor_uses_first_onset_and_replay_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "RUNS", tmp_path)
    monkeypatch.setenv("OFFLINE", "1")
    basic = {"rule_id": "bonds_protect", "strength": {"est": .1, "lo": 0, "hi": .2},
             "integrity": {"status": "ok"}, "first_alarm": None}
    write_lines(tmp_path / "frames.jsonl", [
        {**basic, "as_of": "2019-12-31", "change": {"cusum": 8, "threshold": 7, "alarm": True}},
        {**basic, "as_of": "2020-01-31", "change": {"cusum": 0, "threshold": 7, "alarm": False}},
        {**basic, "as_of": "2020-02-29", "first_alarm": "2020-02-29",
         "change": {"cusum": 9, "threshold": 7, "alarm": True}},
        {**basic, "rule_id": "curve_recession", "as_of": "2019-12-31",
         "change": {"display_only": True, "cusum": None, "threshold": None, "alarm": False}},
    ])
    (tmp_path / "frames_meta.json").write_text(json.dumps({
        "data_source": "synthetic_cache",
        "rules": {"bonds_protect": {
            "first_alarm": "2020-02-29", "locked": {"h": 7, "input_id": "integration"},
            "delay_vs_2022_01_m": -23, "expost_marker": "2020-01-31", "delay_vs_marker_m": 1,
        }},
    }))
    (tmp_path / "calib.json").write_text(json.dumps({"bonds_protect": {"h": 42}}))
    at = AppTest.from_file(str(ROOT / "app/pages/3_Monitor.py")).run()
    assert not at.exception
    assert any("Calibration-period context" in c.value for c in at.caption)
    assert any("h 7.00" in c.value for c in at.caption)
    assert any("Synthetic cache" in w.value for w in at.warning)
    assert not any("FIRST ALARM" in m.value for m in at.markdown)
    at.select_slider[0].set_value(date(2020, 1, 31)).run()
    next(b for b in at.button if b.label == "Play").click().run()
    assert any("FIRST ALARM" in m.value for m in at.markdown)
    assert any("23 months before January 2022" in c.value for c in at.caption)
    assert any("identified retrospectively" in c.value for c in at.caption)
    at.selectbox[0].select("curve_recession").run()
    at.select_slider[0].set_value(date(2019, 12, 31)).run()
    assert len(at.get("plotly_chart")) == 1


def test_checkpoints_require_identical_calibration_input(tmp_path, monkeypatch):
    monkeypatch.setattr(cal, "CKPT_DIR", tmp_path)
    panel = pit.panel()
    first = cal.calibrate("bonds_protect", n=30, panel=panel)
    panel.loc["2014-01-31":"2016-12-31", "TLT"] *= 1.03
    second = cal.calibrate("bonds_protect", n=30, panel=panel)
    assert first["input_id"] != second["input_id"]
    assert second["max_c"] == cal.calibrate("bonds_protect", n=30, panel=panel, checkpoint=False)["max_c"]
    assert len(list(tmp_path.glob("*.npz"))) == 2


def test_live_discovery_stops_before_a_synthetic_batch(monkeypatch):
    import pytest

    monkeypatch.setattr(runner, "require", lambda _: object())
    monkeypatch.setattr(runner, "active_data_source", lambda: "synthetic_cache")
    monkeypatch.setattr(runner, "generate", lambda *_: pytest.fail("Unexpected generation"))
    with pytest.raises(runner.IntegrationMissing, match="Fetch it first"):
        runner.live_discovery("Test the yield curve", 20, "thesis", lambda _: None)
