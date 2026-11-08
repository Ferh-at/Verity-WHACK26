"""AI reviewer: reads the engine's discovery-only evidence and decides the next research step.

Its powers are deliberately asymmetric. It can
  challenge  request extra kill-only tests (they can reject an idea, never promote it),
  refine     propose new DSL hypotheses, each charged to the session ledger like any other idea,
  nominate   choose at most NOMINATE_MAX surviving ideas for the one-shot holdout,
and it chooses whether to run another round. Labels always come from the deterministic rules."""
import json
import re
from dataclasses import dataclass, field as dc_field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.catalog_view import catalog
from agents.generator import GRAMMAR, make_hypothesis, spec_from, universe
from agents.llm import LLMError, chat_json, log, prose
from config import NOMINATE_MAX, TEMPERATURE
from engine.research import CHALLENGES, STATUS_ORDER, ResearchSession
from schemas import Hypothesis, Objection, Thesis

TESTS = ("leak_audit", "sign", "block_null", "outlier", "luck_hurdle", "by_fdr",
         "regime", "costs", "factor", "vendor", "thesis")

SYSTEM = """You are the lead researcher reviewing the results of your own search. A deterministic statistics
engine has tested every hypothesis on the discovery period only. You never see validation or holdout data.
Each hypothesis comes with an Engine evidence line:
  status: survives (passed every discovery gate), fragile (significant on its own but failed the luck hurdle
          or the false-discovery control), weak (not significant), killed (failed a kill test or has the wrong sign)
  failed: the tests it failed. observed_t: its Newey-West t-statistic. p_search and q_by: p-values adjusted
          for the size of the search.
The session line gives hurdle_t: the best t-statistic that pure luck produced when the whole search was
re-run on scrambled data. A hypothesis must beat it.
Every hypothesis you add is charged to the search ledger, which raises the luck hurdle for ALL hypotheses.
Refine like a careful researcher: each refinement must answer a specific failure in the evidence (for example
a fragile idea on a better-suited asset, or a channel the thesis implies but the first round missed). Never
fish for significance by trying small variations of the same idea. While rounds remain and no hypothesis
survives, propose your best-motivated refinements; stop early only if you judge that new ideas are unlikely
to beat a higher hurdle.
Your actions:
  challenges: extra tests that can only reject a hypothesis. Menu:
    regime - may only work in one sub-period (net Sharpe in thirds of the discovery period)
    costs  - may not survive realistic trading costs (re-tested at five times the cost)
    factor - may just be market, trend or carry exposure (alpha after those factors)
    vendor - may be an artefact of one price vendor
  refinements: new hypotheses in the grammar below. Each has based_on (the id it responds to), because (the
    failed test it responds to, or "thesis" for a new angle) and a one-sentence economic rationale.
  nominate: ids of hypotheses with status survives that deserve the single look at the sealed holdout.
  decision: "refine" to test your refinements in another round, or "confirm" to stop and send nominations
    to the holdout. On the final round the decision must be "confirm".
  assessment: at most two plain-English sentences on what the evidence says. Do not write any numbers.
Respect refine_budget and nominate_max from the session line.
"""

RETURN = """Return JSON: {"assessment": "...", "challenges": [{"hypothesis_id": "<id>", "type": "<menu item>",
"rationale": "..."}], "refinements": [{"based_on": "<id>", "because": "<test name>", "spec": {...},
"rationale": "..."}], "nominate": ["<id>"], "decision": "refine"}"""


class DraftChallenge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis_id: str
    type: Literal["regime", "costs", "factor", "vendor"]
    rationale: str = Field(min_length=1, max_length=300)


class DraftRefinement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    based_on: str
    because: Literal[TESTS]
    spec: dict
    rationale: str = Field(min_length=1, max_length=300)


class DraftReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    assessment: str = Field(min_length=1, max_length=600)
    challenges: list[DraftChallenge] = Field(default_factory=list, max_length=12)
    refinements: list[DraftRefinement] = Field(default_factory=list, max_length=12)
    nominate: list[str] = Field(default_factory=list, max_length=8)
    decision: Literal["refine", "confirm"]


@dataclass
class Review:
    assessment: str
    decision: str
    challenges: list[Objection] = dc_field(default_factory=list)
    refinements: list[tuple[Hypothesis, dict]] = dc_field(default_factory=list)
    nominate: list[str] = dc_field(default_factory=list)
    dropped: list[str] = dc_field(default_factory=list)
    fallback: bool = False


def prompt(thesis: Thesis, session: ResearchSession, round_no: int, max_rounds: int, budget: int):
    descriptions, assets = universe(catalog())
    system = SYSTEM + GRAMMAR.format(SERIES_LIST=descriptions, ASSET_LIST=assets) + RETURN
    final = round_no >= max_rounds
    head = {"round": round_no, "max_rounds": max_rounds, "ideas_tried": session.ledger.n_tried,
            "effective_trials": session.ledger.effective_n, "hurdle_t": session.history[-1]["hurdle"],
            "refine_budget": 0 if final else budget, "nominate_max": NOMINATE_MAX}
    blocks = []
    for hid in sorted(session.reps, key=lambda h: STATUS_ORDER.index(session.status(h)[0])):
        h = session.hyps[hid]
        blocks.append(f"Id: {hid}\nHypothesis spec: {h.spec.model_dump_json()}\n"
                      f"Rationale: {json.dumps(prose(h.rationale), ensure_ascii=False)}\n"
                      f"Engine evidence: {json.dumps(session.evidence(hid), sort_keys=True)}")
    user = (f"Thesis: {json.dumps(prose(thesis.text), ensure_ascii=False)}\n"
            + ("This is the final round. Your decision must be confirm.\n" if final else "")
            + f"Engine evidence: {json.dumps({k: v for k, v in head.items() if v is not None}, sort_keys=True)}\n\n"
            + "\n\n".join(blocks))
    return system, user


def fallback(session: ResearchSession, reason: str) -> Review:
    """Deterministic stand-in when the model is unavailable: no new ideas, nominate the strongest survivors."""
    survivors = sorted((h for h in session.reps if session.status(h)[0] == "survives"),
                       key=lambda h: -(session.hurdle["observed"].get(h) or 0))
    return Review(assessment="The reviewer model was unavailable, so the engine nominated its strongest "
                             "surviving ideas and stopped.", decision="confirm",
                  nominate=survivors[:NOMINATE_MAX], dropped=[reason], fallback=True)


def review(thesis: Thesis, session: ResearchSession, round_no: int, max_rounds: int, budget: int) -> Review:
    system, user = prompt(thesis, session, round_no, max_rounds, budget)
    try:
        draft = chat_json("reviewer", system, user, DraftReview, TEMPERATURE["reviewer"])
    except LLMError as exc:
        log.warning("Reviewer unavailable: %s", exc)
        return fallback(session, f"Reviewer unavailable ({type(exc).__name__}).")
    return accept(draft, session, round_no, max_rounds, budget, thesis.id)


def accept(draft: DraftReview, session: ResearchSession, round_no: int, max_rounds: int, budget: int,
           thesis_id: str | None = None) -> Review:
    """Enforce the reviewer's limits in code. Anything outside them is dropped and reported."""
    entries = catalog()
    dropped = []
    final = round_no >= max_rounds
    known = set(session.reps)

    challenges, seen = [], set()
    for c in draft.challenges:
        hid = session.mapping.get(c.hypothesis_id, c.hypothesis_id)
        if hid not in known or c.type not in CHALLENGES:
            dropped.append(f"Challenge on unknown idea {c.hypothesis_id} dropped.")
        elif (hid, c.type) not in seen:
            seen.add((hid, c.type))
            challenges.append(Objection(hypothesis_id=hid, type=c.type, rationale=c.rationale))

    refinements, hashes = [], set()
    for r in draft.refinements:
        if final:
            dropped.append("Refinement ignored: final round.")
            continue
        if len(refinements) >= budget:
            dropped.append("Refinement ignored: round budget reached.")
            continue
        try:
            spec = spec_from(r.spec, entries)
        except (ValidationError, ValueError, KeyError) as exc:
            reason = "malformed spec" if isinstance(exc, ValidationError) else str(exc)
            dropped.append(f"Refinement of {r.based_on} rejected by the grammar: {reason}")
            continue
        h = make_hypothesis(spec, "review", r.rationale, thesis_id)
        if h.id in session.hyps or h.spec_hash in hashes:
            dropped.append(f"Refinement of {r.based_on} already tested ({h.id}).")
            continue
        hashes.add(h.spec_hash)
        refinements.append((h, {"based_on": r.based_on, "because": r.because, "rationale": r.rationale}))

    nominate = []
    for hid in dict.fromkeys(session.mapping.get(x, x) for x in draft.nominate):
        if hid not in known or session.status(hid)[0] != "survives":
            dropped.append(f"Nomination of {hid} refused: only surviving ideas can reach the holdout.")
        elif len(nominate) < NOMINATE_MAX:
            nominate.append(hid)

    decision = "confirm" if final or not refinements else draft.decision
    assessment = draft.assessment.strip()
    if re.search(r"\d", assessment):
        dropped.append("Assessment withheld: it contained numbers, which only the engine may report.")
        assessment = "The reviewer's assessment was withheld by the number guard."
    return Review(assessment, decision, challenges, refinements, nominate, dropped)
