"""Request objections. The statistics pipeline owns every test and verdict."""
import json
import math
from datetime import date, datetime
from numbers import Integral, Real
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agents.llm import DISC_FIELDS, LLMError, chat_json, log, prose
from config import SPLITS, TEMPERATURE
from schemas import Hypothesis, Objection

SYSTEM = """You are a skeptical risk manager reviewing a candidate trading hypothesis. Your only job is to find
reasons it may be a false discovery. You may ONLY raise objections from this menu:
  leakage   - the feature might use information not available at the decision time
  outlier   - the result may be driven by one or a few extreme months
  regime    - it may only work in one sub-period of the sample
  vendor    - the result may be a data artefact of one price vendor
  hindsight - the idea may rely on knowing later events
Each objection will be turned into an automatic statistical test that can only reject the hypothesis.
Raise at most 3 objections, the most plausible first. Do not invent numbers. Return JSON:
{"objections":[{"type":"<menu item>","rationale":"<one sentence>","params":{}}]}"""


class DraftObjection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["leakage", "outlier", "regime", "vendor", "hindsight"]
    rationale: str = Field(min_length=1, max_length=300)
    params: dict = Field(default_factory=dict)

    @field_validator("params")
    @classmethod
    def fixed_tests(cls, value):
        # Test thresholds, dates and sample windows belong to the engine.
        if value:
            raise ValueError("params must be empty for the fixed tests")
        return value


class Objections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    objections: list[DraftObjection] = Field(max_length=3)


class BatchItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hypothesis_id: str
    objections: list[DraftObjection] = Field(default_factory=list, max_length=3)


class BatchObjections(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[BatchItem]


BATCH_SYSTEM = SYSTEM.replace(
    'Return JSON:\n{"objections":[{"type":"<menu item>","rationale":"<one sentence>","params":{}}]}',
    "The user lists several hypotheses. Raise at most 3 objections for each id.\n"
    'Return JSON: {"items":[{"hypothesis_id":"<id>","objections":'
    '[{"type":"<menu item>","rationale":"<one sentence>","params":{}}]}]}',
)


def _clean_summary(disc_summary: dict) -> dict | None:
    if disc_summary.get("split", "discovery") != "discovery":
        return None
    clean = {}
    for key in DISC_FIELDS:
        value = disc_summary.get(key)
        if key in {"best_month", "worst_month"}:
            if isinstance(value, datetime):
                value = value.date()
            if isinstance(value, date):
                value = value.isoformat()
            if isinstance(value, str) and value < SPLITS["holdout"][0]:
                clean[key] = value
        elif isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
            clean[key] = int(value) if isinstance(value, Integral) else float(value)
    return clean


def _objections_from(hypothesis_id: str, drafts: list[DraftObjection]) -> list[Objection]:
    objections = {}
    for item in drafts:
        objections.setdefault(item.type, Objection(hypothesis_id=hypothesis_id, **item.model_dump()))
    return list(objections.values())


def _block(h: Hypothesis, clean: dict) -> str:
    return (
        f"Id: {h.id}\n"
        f"Hypothesis spec: {h.spec.model_dump_json()}\n"
        f"Rationale: {json.dumps(prose(h.rationale), ensure_ascii=False)}\n"
        f"Discovery summary: {json.dumps(clean, sort_keys=True)}"
    )


def object_to(h: Hypothesis, disc_summary: dict) -> list[Objection]:
    clean = _clean_summary(disc_summary)
    if clean is None:
        return []
    user = _block(h, clean)
    try:
        drafts = chat_json("skeptic", SYSTEM, user, Objections, TEMPERATURE["skeptic"])
    except LLMError as exc:
        log.warning("Optional skeptic unavailable for %s: %s", h.id, exc)
        return []
    return _objections_from(h.id, drafts.objections)


def object_to_many(pairs: list[tuple[Hypothesis, dict]]) -> dict[str, list[Objection]]:
    """One skeptic call for a whole discovery batch."""
    blocks = []
    for hypothesis, summary in pairs:
        clean = _clean_summary(summary)
        if clean is not None:
            blocks.append(_block(hypothesis, clean))
    if not blocks:
        return {}
    user = "Review every hypothesis below.\n\n" + "\n\n".join(blocks)
    try:
        drafts = chat_json("skeptic", BATCH_SYSTEM, user, BatchObjections, TEMPERATURE["skeptic"])
    except LLMError as exc:
        log.warning("Optional batched skeptic unavailable: %s", exc)
        return {}
    found = {}
    for item in drafts.items:
        found.setdefault(item.hypothesis_id, _objections_from(item.hypothesis_id, item.objections))
    return found
