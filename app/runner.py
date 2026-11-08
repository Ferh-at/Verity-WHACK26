"""Runs the AI research loop for the UI and stores the latest run for replay."""
from importlib import import_module

from agents.explainer import explain_label
from agents.llm import ROOT, now, write_json
from agents.researcher import run_research
from app.artifacts import Discovery
from app.present import plain_spec, verdict_facts
from engine.locks import file_lock
from engine.research import ResearchSession
from schemas import Thesis


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


def live_research(text, n, stage):
    from uuid import uuid4

    pit = require("data.pit")
    source = active_data_source()
    if source == "synthetic_cache":
        raise IntegrationMissing("The research loop needs the market-data cache. Run `python -m data.fetch` first.")
    thesis = Thesis(id=f"th_{uuid4().hex[:8]}", text=text, created_at=now())
    with file_lock(ROOT / "runs/.batch.lock", wait=False):
        panel = (getattr(pit, "decision_panel", None) or pit.panel)()
        session = ResearchSession(panel, thesis, runs_dir=ROOT / "runs")
        out = run_research(thesis, n, "thesis", panel, stage, session=session)
        if not out["verdicts"]:
            raise RuntimeError("The engine returned no verdicts.")
        hypotheses = {h.id: h for h in out["hypotheses"]}
        discovery = {t.hypothesis_id: t for t in out["trials"] if t.split == "discovery"}

        def add_explanation(v):
            facts = verdict_facts(v, discovery.get(v.hypothesis_id))
            if h := hypotheses.get(v.hypothesis_id):
                facts["plain_english_spec"] = plain_spec(h.spec, spell_out=True)
            return v.model_copy(update={"explanation": explain_label(v.label, facts)})

        verdicts = [add_explanation(v) for v in out["verdicts"]]
        write_json(ROOT / "runs/latest.json", {
            "thesis": text,
            "hypotheses": [h.model_dump(mode="json") for h in out["hypotheses"]],
            "trials": [t.model_dump(mode="json") for t in out["trials"]],
            "verdicts": [v.model_dump(mode="json") for v in verdicts],
            "hurdle": out["hurdle"], "data_source": source,
            "rounds": out["rounds"], "lineage": out["lineage"], "round_of": out["round_of"],
        })
    return Discovery(out["hypotheses"], out["trials"], verdicts, out["hurdle"], False, source,
                     out["rounds"], out["lineage"], out["round_of"], text)
