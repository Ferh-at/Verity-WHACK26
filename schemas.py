"""Shared data contracts for hypotheses, trials, tests, verdicts and monitor frames."""
from typing import Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime, date


class Thesis(BaseModel):
    id: str                      # "th_" + 8 hex chars
    text: str                    # trader's thesis, <= 500 chars
    created_at: datetime


FeatureType = Literal["mom", "level", "change", "zscore", "corr",
                      "centered_ma", "fullsample_z"]     # last two = leaky CONTROLS, never offered to the LLM


class Spec(BaseModel):
    feature: FeatureType
    series: str                  # catalog id, e.g. "T10Y3M"
    series2: Optional[str] = None  # only for "corr"
    lookback: int                # one of LOOKBACKS; ignored for "level" (store 1)
    transform: Literal["sign", "clip_z"]
    asset: str                   # tradable catalog id: SPX, NASDAQ, TLT, GOLD, DXY, BTC (or BENCH_A{i})
    direction: Literal[1, -1]


class Hypothesis(BaseModel):
    id: str                      # "hy_" + first 10 hex of spec_hash
    thesis_id: Optional[str]
    source: Literal["thesis", "explore", "rule", "planted", "review"]  # review = added by the reviewer
    spec: Spec
    rationale: str               # LLM text; shown, never tested
    spec_hash: str               # sha256 of json.dumps(spec.model_dump(), sort_keys=True)
    created_at: datetime
    truth: Optional[bool] = None # benchmark only: True = planted real signal
    trap_type: Optional[str] = None  # benchmark only: real|leaky|outlier|regime|beta|dup|noise


class TrialLogEntry(BaseModel):
    trial_id: str                # "tr_" + uuid4 hex[:10]
    hypothesis_id: str
    dup_of: Optional[str]        # hypothesis_id of cluster representative if duplicate, else None
    batch_id: str                # one per thesis run
    split: Literal["discovery", "validation", "holdout", "post_cutoff"]
    n_obs: int                   # months with a non-NaN position
    coverage: float              # n_obs / months in split
    sharpe_ann: float            # gross
    sharpe_net_ann: float        # after costs
    nw_t: float                  # Newey-West t of mean net return
    p_block: float               # one-sided stationary-bootstrap p
    p_search: Optional[float]    # luck-hurdle p (discovery only)
    hurdle: Optional[float]      # hurdle value (NW-t units) at time of test
    q_by: Optional[float]        # BY-adjusted q
    ts: datetime


class Objection(BaseModel):
    hypothesis_id: str
    type: Literal["leakage", "outlier", "regime", "vendor", "hindsight", "costs", "factor"]
    rationale: str               # <= 300 chars
    params: dict = {}            # e.g. {"date": "2008-10-31"} for outlier


class TestResult(BaseModel):
    __test__ = False             # not a pytest test class
    hypothesis_id: str
    test: Literal["leak_audit", "block_null", "luck_hurdle", "by_fdr", "validation", "holdout",
                  "outlier", "regime", "vendor", "hindsight", "factor", "sign", "costs"]
    passed: bool
    kill: bool                   # True only if failing this test forces Rejected
    value: Optional[float]       # main number (p, t, Sharpe ...)
    detail: str                  # one-line human-readable
    evidence_ref: str            # "runs/tests.jsonl#<id>" or chart key


class Power(BaseModel):
    years: float                 # n_obs / 12 of the split used for the verdict (discovery)
    mde_sharpe_80: float         # annual Sharpe detectable with 80% power
    mintrl_years: Optional[float]  # None if observed Sharpe <= 0
    power_at_05: float           # power to detect annual Sharpe 0.5


Label = Literal["Rejected", "Can't tell yet", "Fragile", "Promising", "Replicated"]


class Verdict(BaseModel):
    hypothesis_id: str
    label: Label
    reasons: list[TestResult]
    power: Power
    holdout_opened: bool
    holdout_passed: Optional[bool]
    dsr: Optional[float]         # deflated Sharpe probability (0-1)
    evidence_refs: list[str]
    explanation: Optional[str]   # filled by the explainer
    as_of: date


class ReplayFrame(BaseModel):
    rule_id: str                 # bonds_protect|dollar_haven|gold_realyield|btc_diversifier|curve_recession|vix_control
    as_of: date
    integrity: dict              # {"vendors_agree": bool|None, "max_gap_pct": float|None, "status": "ok"|"flag"|"single_vendor"}
    strength: dict               # {"est": float, "lo": float, "hi": float}
    change: Optional[dict]       # {"cusum": float, "threshold": float, "alarm": bool} (None for curve_recession)
    first_alarm: Optional[date]


class BenchmarkRun(BaseModel):
    seed: int
    kind: Literal["trap", "break", "stable"]
    type: str                    # trap_type, or rule_id for break/stable
    truth: bool                  # traps: real signal? breaks: True
    label: Optional[Label]       # traps
    killed_by: Optional[str]     # first failing test name, None if trusted
    delay_m: Optional[int]       # breaks: months from break to first alarm (None = missed)
    false_alarm: Optional[bool]  # stable sims / pre-break alarms
