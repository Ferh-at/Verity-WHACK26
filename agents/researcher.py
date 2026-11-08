"""The AI research loop: generate -> engine explores -> AI reviews -> (refine -> explore -> review)* -> confirm.

Every round is charged to one ledger, so the luck hurdle rises as the AI tries more ideas. The reviewer
sees discovery evidence only; validation and the one-shot holdout run once, after it has finished.

CLI: python -m agents.researcher --thesis "..." --n 20"""
import argparse
import uuid
from collections import Counter
from typing import Callable

from agents.generator import generate
from agents.llm import now
from agents.reviewer import review
from config import N_HYPOTHESES_DEFAULT, REFINE_BUDGET, RESEARCH_ROUNDS
from engine.research import ResearchSession
from schemas import Thesis


def run_research(thesis: Thesis, n: int, mode: str, panel, stage: Callable[[str], None] = lambda _: None, *,
                 max_rounds: int = RESEARCH_ROUNDS, budget: int = REFINE_BUDGET,
                 generate_fn=None, review_fn=None, session: ResearchSession | None = None) -> dict:
    generate_fn, review_fn = generate_fn or generate, review_fn or review
    session = session or ResearchSession(panel, thesis)
    stage("Round 1 · AI generates hypotheses from the thesis")
    pending = generate_fn(thesis, n, mode)
    lineage: dict[str, dict] = {}
    rounds, outcome = [], None
    for round_no in range(1, max_rounds + 1):
        stage(f"Round {round_no} · engine tests {len(pending)} new ideas on the discovery period")
        new_ids = session.explore(pending)
        stage(f"Round {round_no} · AI reviewer reads the evidence")
        outcome = review_fn(thesis, session, round_no, max_rounds, budget)
        tested = session.challenge(outcome.challenges)
        by_key = {(r.hypothesis_id, r.test): r for r in tested}
        rounds.append({
            "round": round_no, "new": new_ids, "ideas_tried": session.ledger.n_tried,
            "effective_trials": session.ledger.effective_n, "hurdle": session.history[-1]["hurdle"],
            "status": dict(Counter(session.status(h)[0] for h in session.reps)),
            "survivors": [h for h in session.reps if session.status(h)[0] == "survives"],
            "assessment": outcome.assessment, "decision": outcome.decision, "fallback": outcome.fallback,
            "challenges": [{"hypothesis_id": c.hypothesis_id, "type": c.type, "rationale": c.rationale,
                            **({"passed": r.passed, "detail": r.detail}
                               if (r := by_key.get((c.hypothesis_id, c.type))) else {"detail": "already run"})}
                           for c in outcome.challenges],
            "refinements": [{"id": h.id, **meta} for h, meta in outcome.refinements],
            "nominate": outcome.nominate, "dropped": outcome.dropped,
        })
        for h, meta in outcome.refinements:
            lineage[h.id] = meta
        if outcome.decision == "confirm" or not outcome.refinements:
            break
        pending = [h for h, _ in outcome.refinements]
    stage("Confirm · validation for every idea, one-shot holdout for nominated ideas")
    verdicts, trials, record = session.confirm(outcome.nominate if outcome else [])
    record["loop"] = rounds
    return {"session": session, "hypotheses": list(session.hyps.values()), "verdicts": verdicts,
            "trials": trials, "hurdle": record, "rounds": rounds, "lineage": lineage,
            "round_of": dict(session.round_of)}


def main():
    ap = argparse.ArgumentParser(description="Thesis -> AI research loop -> verdicts")
    ap.add_argument("--thesis", default="An inverted yield curve predicts equity drawdowns")
    ap.add_argument("--n", type=int, default=N_HYPOTHESES_DEFAULT)
    ap.add_argument("--mode", default="thesis", choices=["thesis", "explore"])
    ap.add_argument("--rounds", type=int, default=RESEARCH_ROUNDS)
    a = ap.parse_args()
    from data.pit import decision_panel
    from data.store import using_mock
    if using_mock():
        ap.error("Fetch market data first (python -m data.fetch).")
    thesis = Thesis(id="th_" + uuid.uuid4().hex[:8], text=a.thesis, created_at=now())
    out = run_research(thesis, a.n, a.mode, decision_panel(), print, max_rounds=a.rounds)
    for r in out["rounds"]:
        print(f"round {r['round']}: ideas {r['ideas_tried']}, effective {r['effective_trials']}, "
              f"hurdle {r['hurdle']}, decision {r['decision']}, nominate {r['nominate']}")
        print(f"  reviewer: {r['assessment']}")
    for v in out["verdicts"]:
        print(f"{v.hypothesis_id}  {v.label}")


if __name__ == "__main__":
    main()
