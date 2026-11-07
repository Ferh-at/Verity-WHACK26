"""Member 3 orchestration at the frozen Member 1/2 interfaces."""
from importlib import import_module

from agents.explainer import explain_label
from agents.generator import generate
from agents.llm import ROOT, now, write_json
from agents.skeptic import object_to_many
from app.artifacts import Discovery
from app.present import plain_spec, verdict_facts
from engine.locks import file_lock
from schemas import Thesis, TrialLogEntry, Verdict


class IntegrationMissing(RuntimeError):
    pass


def require(module):
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module or module.startswith(f"{exc.name}."):
            raise IntegrationMissing(f"Waiting for the team module {module}. Use the cached replay.") from exc
        raise


def active_data_source():
    try:
        store = require("data.store")
    except IntegrationMissing:
        return "unknown"
    return store.cache_source() if hasattr(store, "cache_source") else "unknown"


def live_discovery(text, n, mode, stage):
    from uuid import uuid4

    engine = require("engine.pipeline")
    pit = require("data.pit")
    source = active_data_source()
    if source == "synthetic_cache":
        raise IntegrationMissing("Discovery uses the market-data cache. Fetch it first, or replay the synthetic demo.")
    thesis = Thesis(id=f"th_{uuid4().hex[:8]}", text=text, created_at=now())
    stage("Generate validated hypotheses")
    hyps = generate(thesis, n, mode)
    ids = {h.id for h in hyps}
    if len(ids) < n:
        stage(f"Generator returned {len(ids)} of {n} requested ideas after validation.")

    # The shared engine writes an unkeyed hurdle.json. Serialise M3 live batches
    # and associate trials by new trial ID plus batch ID and hypothesis ID.
    lock_path = ROOT / "runs/.m3_batch.lock"
    with file_lock(lock_path, wait=False):
        panel = (getattr(pit, "decision_panel", None) or pit.panel)()
        stage("Engine running audit → test → hurdle → skeptic → validation")
        info = {}
        verdicts = [
            Verdict.model_validate(row) for row in engine.run_batch(
                thesis, hyps, panel, None, objections_batch_fn=object_to_many, info=info)
        ]
        if not verdicts:
            raise RuntimeError("The engine returned no verdicts.")
        trials = [TrialLogEntry.model_validate(row) for row in info.get("trials", [])]
        hurdle = info.get("hurdle_record", {})

        stage("Build verdict cards and guarded explanations")
        hypotheses = {h.id: h for h in hyps}
        discovery = {t.hypothesis_id: t for t in trials if t.split == "discovery"}

        def add_explanation(v):
            facts = verdict_facts(v, discovery.get(v.hypothesis_id))
            if h := hypotheses.get(v.hypothesis_id):
                facts["plain_english_spec"] = plain_spec(h.spec, spell_out=True)
            return v.model_copy(update={"explanation": explain_label(v.label, facts)})

        verdicts = [add_explanation(v) for v in verdicts]
        write_json(ROOT / "runs/m3_latest.json", {
            "hypotheses": [h.model_dump(mode="json") for h in hyps],
            "trials": [t.model_dump(mode="json") for t in trials],
            "verdicts": [v.model_dump(mode="json") for v in verdicts],
            "hurdle": hurdle,
            "data_source": source,
        })
    return Discovery(hyps, trials, verdicts, hurdle, False, source)


def fixture_discovery(n):
    engine = require("engine.pipeline")
    hyps, verdicts, info = engine.fixture_batch(n)
    return Discovery(hyps, info["trials"], verdicts, info["hurdle_record"], True, "engine_fixture")
