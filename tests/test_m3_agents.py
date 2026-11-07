import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

from agents import explainer, generator, llm, probe, skeptic
from agents.guard import check_numbers
from app.fixtures import sample_hypotheses
from schemas import Thesis, Verdict

MOCKS = Path(__file__).resolve().parents[1] / "mocks"


class Output(BaseModel):
    value: int


def audit(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def isolated_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LOG_PATH", tmp_path / "calls.jsonl")
    monkeypatch.delenv("LLM_CACHE_DIR", raising=False)
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("LLM_MODEL_FALLBACK", "gpt-4o")
    monkeypatch.setenv("USE_CACHE", "1")
    return llm.LOG_PATH


def test_provider_retry_fallback_cache_and_every_attempt_logged(monkeypatch, isolated_llm):
    attempts = []

    def complete(**kwargs):
        attempts.append(kwargs)
        data = '{"value": 7}' if kwargs["model"] == "gpt-4o" else '{"missing": 7}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=data))])

    monkeypatch.setenv("OFFLINE", "0")
    monkeypatch.setenv("LLM_API_KEY", "fake")
    monkeypatch.setattr(llm, "OpenAI", lambda **_: SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=complete))))
    assert llm.chat_json("generator", "JSON only", "Return JSON", Output, .7).value == 7
    assert len(attempts) == 4
    assert "failed validation" in attempts[1]["messages"][1]["content"]
    assert attempts[0]["response_format"] == {"type": "json_object"}
    monkeypatch.setenv("OFFLINE", "1")
    assert llm.chat_json("generator", "JSON only", "Return JSON", Output, .7).value == 7
    with pytest.raises(llm.OfflineMiss):
        llm.chat_json("generator", "JSON only", "Different request", Output, .7)
    with pytest.raises(llm.HoldoutViolation):
        llm.chat_json("generator", "JSON only", "Use 2021-03", Output, .7)
    assert [row["source"] for row in audit(isolated_llm)] == ["provider"] * 4 + ["cache", "offline_miss", "blocked"]
    assert [row["ok"] for row in audit(isolated_llm)] == [False, False, False, True, True, False, False]


@pytest.mark.parametrize("text", [
    "2021-03", "March 2022", "SPX returned 4.5%", "prices: [11, 12]",
    'Discovery summary: {"holdout_sharpe": 0.1}',
    'FACTS: {"sharpe": [0.3, 0.5]}',
    'FACTS: {"as_of": "2020-01-31"}',
])
def test_prompt_boundary(text):
    with pytest.raises(llm.HoldoutViolation):
        llm.assert_no_holdout(text)


def test_generator_refills_invalid_and_duplicate_rows(monkeypatch):
    initial = sample_hypotheses()
    drafted = [{"spec": h.spec.model_dump(), "rationale": h.rationale} for h in initial]
    calls = []

    def chat(role, system, user, schema, temperature):
        llm.assert_no_holdout(system)
        llm.assert_no_holdout(user)
        calls.append(user)
        leaky = {"spec": {**drafted[0]["spec"], "feature": "centered_ma"}, "rationale": "Leak"}
        if len(calls) % 2:
            return schema(hypotheses=[drafted[0], drafted[0], leaky, {"spec": {}, "rationale": "Invalid"}])
        assert "Duplicate spec" in user
        return schema(hypotheses=drafted[1:])

    monkeypatch.setattr(generator, "chat_json", chat)
    for name in ["curve", "gold", "dollar", "rates", "stocks"] * 10:
        thesis = Thesis(id="th_test", text=f"Study {name} as a macro signal", created_at=llm.now())
        hyps = generator.generate(thesis, 20)
        assert len({h.spec_hash for h in hyps}) == 20
        assert all(h.spec.feature not in generator.LEAKY for h in hyps)
        assert all(h.id == "hy_" + h.spec_hash[:10] for h in hyps)
    assert len(calls) == 100


def test_skeptic_fixed_menu_and_summary_boundary(monkeypatch):
    def chat(role, system, user, schema, temperature):
        llm.assert_no_holdout(system)
        llm.assert_no_holdout(user)
        assert "holdout" not in user.lower()
        assert "2022" not in user
        return schema(objections=[{"type": "regime", "rationale": "Only one subperiod may work."},
                                  {"type": "regime", "rationale": "Duplicate."},
                                  {"type": "vendor", "rationale": "Check the vendor."}])

    monkeypatch.setattr(skeptic, "chat_json", chat)
    h = sample_hypotheses()[0]
    for _ in range(30):
        objections = skeptic.object_to(h, {"p_block": .02, "best_month": "2011-04-30",
                                           "worst_month": "2022-01-31", "holdout": .7})
        assert [o.type for o in objections] == ["regime", "vendor"]
        assert all(o.hypothesis_id == h.id for o in objections)
    assert not skeptic.object_to(h, {"split": "holdout", "p_block": .01})
    with pytest.raises(ValidationError):
        skeptic.Objections(objections=[{"type": "promote", "rationale": "Strong"}])
    with pytest.raises(ValidationError):
        skeptic.Objections(objections=[{"type": "outlier", "rationale": "Check", "params": {"alpha": .5}}])


def test_explainer_guard_regeneration_and_sealed_fallback(monkeypatch):
    calls = []

    def chat(role, system, user, schema, temperature):
        llm.assert_no_holdout(system)
        llm.assert_no_holdout(user)
        calls.append(user)
        return schema(explanation="Sharpe 1.27.")

    monkeypatch.setattr(explainer, "chat_json", chat)
    verdicts = [Verdict.model_validate_json(line) for line in (MOCKS / "sample_verdicts.jsonl").read_text().splitlines()]
    v = next(v for v in verdicts if v.label == "Can't tell yet")
    facts = {"sharpe": "1.25", "mde_sharpe_80": "0.92", "power_pct": "80%"}
    assert not check_numbers("Sharpe 1.27", facts)
    assert not check_numbers("Sharpe 1.250", facts)
    assert not check_numbers("Sharpe 1.25e0", {"sharpe": "1.25", "e": "0"})
    assert check_numbers("Sharpe 1.25. Date 2019-12-31.", {**facts, "as_of": "2019-12-31"})
    for _ in range(30):
        text = explainer.explain_verdict(v, facts)
        assert check_numbers(text, facts)
        assert "0.92" in text
    assert len(calls) == 60
    assert "1.25" in explainer.explain_verdict(v.model_copy(update={"as_of": date(2020, 1, 31)}), facts)
    assert len(calls) == 60


def test_probe_replays_80_cache_only(monkeypatch, isolated_llm, tmp_path):
    monkeypatch.setenv("OFFLINE", "1")
    monkeypatch.setenv("LLM_CACHE_DIR", str(MOCKS / "probe_cache"))
    monkeypatch.setattr(llm, "OpenAI", lambda **_: pytest.fail("Offline mode constructed a client"))
    out = tmp_path / "probe.json"
    result = probe.run_probe(MOCKS / "probe_truth", out)
    assert out.exists()
    assert len(result["rows"]) == 80
    assert all(row["status"] == "answered" for row in result["rows"])
    assert len(audit(isolated_llm)) == 80
