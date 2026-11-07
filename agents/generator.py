"""Translate a thesis into validated, deduplicated DSL hypotheses."""
import hashlib
import json
import logging
from importlib import import_module
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agents.catalog_view import FALLBACK, catalog, field
from agents.llm import LLMError, chat_json, now
from config import LOOKBACKS, N_HYPOTHESES_DEFAULT, TEMPERATURE
from schemas import Hypothesis, Spec, Thesis

log = logging.getLogger(__name__)
DEFAULT_THESIS = "An inverted yield curve predicts equity drawdowns"
EXPLORE_THESIS = "Explore macro relationships useful to a global macro trader."
LEAKY = {"centered_ma", "fullsample_z"}

SYSTEM = """You are a quantitative research assistant. You turn a trader's market thesis into testable hypotheses
in a fixed grammar. You never see or produce data values, dates after 2019, or performance numbers.
Each hypothesis is a JSON object with fields:
  feature: one of "mom" (past L-month return of a price series), "level" (current value),
           "change" (L-month change), "zscore" (value vs its trailing L-month mean/sd, L>=12),
           "corr" (trailing L-month correlation of monthly changes of series and series2, L>=12)
  series: one of {SERIES_LIST}
  series2: same list, only for "corr", else null
  lookback: one of [1,3,6,12,24,36] (use 1 for "level")
  transform: "sign" or "clip_z"
  asset: the traded asset, one of {ASSET_LIST}
  direction: 1 if a HIGH feature value means go LONG the asset next month, -1 if it means go SHORT
Rules: the position is set at month-end using only data up to that month and held for the next month.
Make hypotheses diverse (different series, lookbacks and assets); do not submit near-copies that differ
only by lookback unless the thesis requires it. Each rationale is one sentence of economic reasoning.
Return JSON: {{"hypotheses": [{{"spec": {{...}}, "rationale": "..."}}]}}"""


class Candidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypotheses: list[dict]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: dict
    rationale: str = Field(min_length=1, max_length=500)


def prompt(thesis, n, mode, entries=None):
    if entries is None:
        entries = catalog()
    descriptions = ", ".join(
        f"{key}: {field(entry, 'description') or FALLBACK.get(key, ('', key))[1]}"
        for key, entry in entries.items() if field(entry, "type") != "events"
    )
    assets = ", ".join(key for key, entry in entries.items() if field(entry, "tradable", False))
    system = SYSTEM.format(SERIES_LIST=descriptions, ASSET_LIST=assets)
    text = EXPLORE_THESIS if mode == "explore" else thesis.text.strip()
    user = f"Thesis: {json.dumps(text, ensure_ascii=False)}\nMode: {mode}\nReturn exactly {n} hypotheses."
    return system, user


def _dsl():
    try:
        return import_module("engine.dsl")
    except ModuleNotFoundError as exc:
        if exc.name not in {"engine", "engine.dsl"}:
            raise
        return None


def validate_candidate(spec, entries):
    if spec.feature in LEAKY:
        raise ValueError("Leaky control is reserved for the audit.")
    if spec.lookback not in LOOKBACKS:
        raise ValueError("Lookback is outside the DSL.")
    if spec.feature == "level" and spec.lookback != 1:
        raise ValueError("Level requires lookback one.")
    if spec.feature in {"zscore", "corr"} and spec.lookback < 12:
        raise ValueError("Trailing z-score and correlation require a year of history.")
    if spec.series not in entries or field(entries[spec.series], "type") == "events":
        raise ValueError("Unknown or unsupported numeric series.")
    if spec.feature == "mom" and field(entries[spec.series], "type") != "price":
        raise ValueError("Momentum requires a price series.")
    if spec.feature == "corr":
        if (spec.series2 == spec.series or spec.series2 not in entries
                or field(entries[spec.series2], "type") == "events"):
            raise ValueError("Correlation requires a different numeric second series.")
    elif spec.series2 is not None:
        raise ValueError("Only correlation accepts a second series.")
    if spec.asset not in entries or not field(entries[spec.asset], "tradable"):
        raise ValueError("Asset is not tradable.")
    dsl = _dsl()
    if dsl is not None:
        dsl.validate_spec(spec, entries)


def make_hypothesis(spec, source, rationale, thesis_id=None):
    dsl = _dsl()
    if dsl is not None and source != "planted":
        return dsl.make_hypothesis(spec, source, rationale, thesis_id=thesis_id)
    digest = hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()
    return Hypothesis(id=f"hy_{digest[:10]}", thesis_id=thesis_id, source=source,
                      spec=spec, rationale=rationale, spec_hash=digest, created_at=now())


def generate(thesis: Thesis, n=N_HYPOTHESES_DEFAULT,
             mode: Literal["thesis", "explore"] = "thesis") -> list[Hypothesis]:
    if mode not in {"thesis", "explore"} or type(n) is not int or not 1 <= n <= 99:
        raise ValueError("Provide thesis or explore mode and a hypothesis count from 1 to 99.")
    if mode == "thesis" and not 1 <= len(thesis.text.strip()) <= 500:
        raise ValueError("Enter a thesis of 1 to 500 characters.")
    entries = catalog()
    system, user = prompt(thesis, n, mode, entries)
    found = {}
    repair = ""
    for attempt in range(2):
        try:
            result = chat_json("generator", system, user + repair, Candidates, TEMPERATURE["generator"])
        except LLMError:
            if attempt == 0 or not found:
                raise
            log.warning("Generator refill unavailable. Returning %s validated ideas.", len(found))
            break
        rejected = []
        for index, raw in enumerate(result.hypotheses, 1):
            spec = None
            try:
                row = Candidate.model_validate(raw)
                if row.spec.keys() - Spec.model_fields.keys():
                    raise ValueError("Unknown spec fields.")
                spec = Spec.model_validate(row.spec, strict=True)
                if type(row.spec["direction"]) is not int:
                    raise ValueError("Direction must be an integer.")
                validate_candidate(spec, entries)
                h = make_hypothesis(spec, mode, row.rationale, thesis.id)
                if h.spec_hash in found:
                    raise ValueError("Duplicate spec.")
                found[h.spec_hash] = h
                if len(found) == n:
                    return list(found.values())
            except (ValueError, KeyError) as exc:
                reason = "Malformed candidate or spec." if isinstance(exc, ValidationError) else str(exc)
                # Repair prompts may contain only approved specs. Never echo malformed model output.
                safe_ids = spec and all(
                    item is None or item in entries
                    for item in (spec.series, spec.series2, spec.asset)
                )
                detail = f"Rejected candidate. {reason}"
                if safe_ids and spec.feature not in LEAKY and spec.lookback in LOOKBACKS:
                    detail += f"\nHypothesis spec: {spec.model_dump_json()}"
                rejected.append(detail)
                log.info("Generator candidate %s dropped: %s", index, reason)
        if attempt == 0:
            # Keep the repair prompt numeric-data free and reference invalid ideas by row.
            repairs = "\n".join(rejected[:n]) or "The batch was short."
            kept = "\n".join(f"Hypothesis spec: {h.spec.model_dump_json()}" for h in found.values())
            repair = f"\nReturn only {n - len(found)} new hypotheses. Exclude the retained specs below.\n{kept}\n{repairs}"
    if not found:
        raise LLMError("Generator returned no usable hypotheses after the refill.")
    return list(found.values())
