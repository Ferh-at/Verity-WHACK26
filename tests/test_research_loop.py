import pandas as pd
import pytest

from agents.llm import REVIEW_FIELDS, assert_no_holdout, now
from agents.reviewer import DraftReview, accept, fallback, prompt
from app.fixtures import sample_hypotheses
from engine.research import ResearchSession
from engine.vault import Vault
from schemas import Thesis

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
MOM = {"feature": "mom", "series": "SPX", "series2": None, "lookback": 12, "transform": "sign",
       "asset": "SPX", "direction": 1}


@pytest.fixture(scope="module")
def session():
    panel = pd.read_parquet(ROOT / "mocks/panel.parquet")
    s = ResearchSession(panel, runs_dir=None, vault=Vault(None), n_boot=200, n_hurdle=60)
    s.explore(sample_hypotheses()[:12])
    return s


def thesis(text="An inverted 10y-3m curve predicted the 2022 drawdown"):
    return Thesis(id="th_test0001", text=text, created_at=now())


def test_prompt_passes_boundary_and_shows_discovery_only(session):
    system, user = prompt(thesis(), session, 1, 3, 5)
    assert_no_holdout(system)
    assert_no_holdout(user)
    assert "2022" not in user
    for hid in session.reps:
        assert set(session.evidence(hid)) <= REVIEW_FIELDS
    assert "validation" not in user


def test_accept_enforces_limits_in_code(session):
    ids = session.reps
    refine = [{"based_on": ids[0], "because": "luck_hurdle", "rationale": "Trend.", "spec": MOM},
              {"based_on": ids[0], "because": "thesis", "rationale": "Leak.", "spec": {**MOM, "feature": "centered_ma"}},
              {"based_on": ids[1], "because": "thesis", "rationale": "Bad lookback.", "spec": {**MOM, "lookback": 5}},
              {"based_on": ids[1], "because": "thesis", "rationale": "Slower.", "spec": {**MOM, "lookback": 6}},
              {"based_on": ids[1], "because": "thesis", "rationale": "Over budget.", "spec": {**MOM, "lookback": 24}}]
    draft = DraftReview(assessment="Signals look weak in 2008.", decision="refine", refinements=refine,
                        challenges=[{"hypothesis_id": "hy_unknown", "type": "regime", "rationale": "x"},
                                    {"hypothesis_id": ids[0], "type": "costs", "rationale": "Turnover."}],
                        nominate=ids[:3])
    out = accept(draft, session, 1, 3, budget=2, thesis_id="th_test0001")
    assert [h.spec.lookback for h, _ in out.refinements] == [12, 6]
    assert any("budget" in d for d in out.dropped)
    assert all(h.source == "review" for h, _ in out.refinements)
    assert [c.type for c in out.challenges] == ["costs"]
    assert all(session.status(h)[0] == "survives" for h in out.nominate)
    assert "withheld" in out.assessment
    assert any("leaky" in d.lower() or "audit" in d.lower() for d in out.dropped)
    final = accept(draft, session, 3, 3, budget=5)
    assert final.decision == "confirm" and not final.refinements


def test_hurdle_never_falls_and_holdout_only_for_nominated():
    panel = pd.read_parquet(ROOT / "mocks/panel.parquet")
    s = ResearchSession(panel, runs_dir=None, vault=Vault(None), n_boot=200, n_hurdle=60)
    hyps = sample_hypotheses()
    s.explore(hyps[:10])
    s.explore(hyps[10:])
    hurdles = [r["hurdle"] for r in s.history]
    assert hurdles[1] >= hurdles[0]
    assert s.history[-1]["ideas_tried"] == 20
    verdicts, trials, record = s.confirm(nominated=[])
    assert len(verdicts) == 20 and not any(v.holdout_opened for v in verdicts)
    assert record["rounds"] == s.history
    with pytest.raises(RuntimeError):
        s.explore(hyps[:1])


def test_fallback_nominates_survivors_only(session):
    out = fallback(session, "offline")
    assert out.fallback and out.decision == "confirm"
    assert all(session.status(h)[0] == "survives" for h in out.nominate)
