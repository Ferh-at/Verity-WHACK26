"""Hypothesis DSL: spec validation, hashing, Hypothesis construction (M1.1)."""
import hashlib
import json
from datetime import datetime, timezone

from catalog import CATALOG
from config import LOOKBACKS
from schemas import Hypothesis, Spec

LEAKY_FEATURES = {"centered_ma", "fullsample_z"}


def validate_spec(spec: Spec, catalog: dict | None = None, source: str | None = None,
                  allow_leaky: bool = False) -> None:
    """Raise ValueError with a reason if the spec is not allowed."""
    cat = CATALOG if catalog is None else catalog
    if spec.lookback not in LOOKBACKS:
        raise ValueError(f"lookback {spec.lookback} not in {LOOKBACKS}")
    if spec.feature == "level" and spec.lookback != 1:
        raise ValueError("level must store lookback 1")
    if spec.feature in ("zscore", "corr") and spec.lookback < 12:
        raise ValueError(f"{spec.feature} needs lookback >= 12")
    if spec.series not in cat:
        raise ValueError(f"unknown series {spec.series}")
    if spec.feature == "corr":
        if spec.series2 is None:
            raise ValueError("corr needs series2")
        if spec.series2 not in cat:
            raise ValueError(f"unknown series2 {spec.series2}")
        if spec.series2 == spec.series:
            raise ValueError("series2 must differ from series")
    elif spec.series2 is not None:
        raise ValueError("series2 is only allowed for corr")
    if spec.asset not in cat or not cat[spec.asset].get("tradable", False):
        raise ValueError(f"asset {spec.asset} is not tradable")
    if spec.feature in LEAKY_FEATURES and not (allow_leaky or source == "planted"):
        raise ValueError(f"{spec.feature} is a leaky control, allowed only for planted/audit use")


def spec_hash(spec: Spec) -> str:
    return hashlib.sha256(json.dumps(spec.model_dump(), sort_keys=True).encode()).hexdigest()


def make_hypothesis(spec: Spec, source: str, rationale: str, thesis_id: str | None = None,
                    created_at: datetime | None = None, truth: bool | None = None,
                    trap_type: str | None = None) -> Hypothesis:
    h = spec_hash(spec)
    return Hypothesis(id="hy_" + h[:10], thesis_id=thesis_id, source=source, spec=spec,
                      rationale=rationale, spec_hash=h,
                      created_at=created_at or datetime.now(timezone.utc),
                      truth=truth, trap_type=trap_type)
