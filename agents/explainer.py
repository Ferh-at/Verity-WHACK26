"""Guarded explanations with a deterministic path for sealed-period results."""
import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field

from agents.guard import check_numbers, display_date
from agents.llm import EXPLAIN_FIELDS, LLMError, chat_json
from config import SPLITS, TEMPERATURE
from schemas import ReplayFrame, Verdict

log = logging.getLogger(__name__)
SYSTEM = """You explain statistical results to a portfolio manager in plain English, in at most 3 sentences.
Use ONLY the numbers given in FACTS, written exactly as given. Do not compute new numbers, do not
round differently, do not add dates or statistics that are not in FACTS. If FACTS say the label is
"Can't tell yet", explain that the data is too short to detect a plausible effect, citing the
detectable Sharpe. Never claim the strategy will make money."""


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanation: str = Field(min_length=1, max_length=700)


def explain_label(label, facts):
    sharpe = facts.get("sharpe_net_ann", facts.get("net_sharpe", facts.get("sharpe")))
    p = facts.get("p_block", facts.get("p"))
    mde = facts.get("mde_sharpe_80")
    evidence = ", ".join(
        phrase for value, phrase in (
            (sharpe, f"net Sharpe {sharpe}"), (p, f"one-sided p {p}")
        ) if value is not None
    )
    lead = f"{label}: {evidence}." if evidence else f"{label}."
    if label == "Can't tell yet":
        lead += " The sample is too short to resolve a plausible effect."
    if mde is not None:
        lead += f" The sample can detect an annual Sharpe of at least {mde} at 80% power."
    return lead


def _ask(label, facts, subject, fallback):
    clean = {key: value for key, value in facts.items() if key in EXPLAIN_FIELDS}
    if not clean or facts.keys() - EXPLAIN_FIELDS - {"hypothesis", "plain_english_spec"}:
        return fallback
    user = (
        f"LABEL: {label}\nFACTS: {json.dumps(clean, sort_keys=True, allow_nan=False)}"
        f"\nHypothesis: {json.dumps(subject, ensure_ascii=False)}"
        '\nReturn JSON as {"explanation":"..."}'
    )
    for attempt in range(2):
        suffix = "\nThe last text added an unsupported number. Copy every number exactly from FACTS." if attempt else ""
        try:
            answer = chat_json("explainer", SYSTEM, user + suffix, Explanation, TEMPERATURE["explainer"])
        except LLMError:
            break
        text = answer.explanation.strip()
        if (text and check_numbers(text, facts)
                and len(re.split(r"(?<=[.!?])\s+", text)) <= 3
                and not re.search(r"\b(?:guaranteed|will make money|risk.free)\b", text, re.I)):
            return text
        log.info("Explainer response failed output checks.")
    return fallback


def explain_verdict(v: Verdict, facts: dict) -> str:
    evidence = {"power_pct": "80%", **facts, "label": v.label}
    fallback = explain_label(v.label, evidence)
    if v.holdout_opened or display_date(v.as_of) >= SPLITS["holdout"][0]:
        return fallback
    subject = evidence.get("plain_english_spec", evidence.get("hypothesis", "Monthly strategy"))
    return _ask(v.label, evidence, subject, fallback)


def explain_frame(frame: ReplayFrame, facts: dict) -> str:
    alarm = frame.change.get("alarm") is True
    fallback = (
        "The rule's strength fell far enough to trigger the locked change detector."
        if alarm else "The locked change detector shows no alarm on this date."
    )
    if not frame.change:
        fallback = "This rule is monitored for strength only."
    return fallback
