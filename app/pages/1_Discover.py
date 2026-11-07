from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.explainer import explain_label
from agents.generator import DEFAULT_THESIS
from agents.guard import check_numbers
from agents.llm import LLMError, offline
from app.artifacts import cached_discovery, demo_panel
from app.fixtures import CONTROL, sample_edit
from app.present import (LABEL_COLORS, LABEL_ORDER, MINT, heading, histogram, number, pct,
                         plain_spec, shell, source_note, verdict_facts)
from app.runner import IntegrationMissing, active_data_source, fixture_discovery, live_discovery, require
from config import N_HYPOTHESES_DEFAULT, SPLITS


def show_ledger(result):
    st.subheader("Trial ledger")
    discovery = [row for row in result.trials if row.split == "discovery"]
    by_id = {h.id: h for h in result.hypotheses}
    a, b, c = st.columns(3)
    a.metric("Ideas tried", len(result.hypotheses))
    b.metric("Effective trials", sum(row.dup_of is None for row in discovery) if discovery else "—")
    c.metric("Duplicates merged", sum(row.dup_of is not None for row in discovery) if discovery else "—")
    if not discovery:
        st.caption("The engine has not written discovery trials for this batch.")
        return
    table = []
    for row in discovery:
        h = by_id.get(row.hypothesis_id)
        spec = h.spec if h else None
        table.append({
            "id": row.hypothesis_id, "feature": spec.feature if spec else None,
            "series": spec.series if spec else None, "asset": spec.asset if spec else None,
            "lookback": spec.lookback if spec else None, "dup_of": row.dup_of,
            "n_obs": row.n_obs, "sharpe_net_ann": row.sharpe_net_ann, "nw_t": row.nw_t,
            "p_block": row.p_block, "p_search": row.p_search, "q_by": row.q_by,
        })
    decimals = {"sharpe_net_ann": 2, "nw_t": 2, "p_block": 3, "p_search": 3, "q_by": 3}
    st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True,
                 column_config={key: st.column_config.NumberColumn(format=f"%.{n}f")
                                for key, n in decimals.items()})


def show_hurdle(result):
    st.subheader("The luck hurdle")
    null_max = result.hurdle.get("null_max", [])
    if not null_max:
        st.caption("No search-level null was written for this batch.")
        return
    fig = histogram(null_max, result.hurdle.get("hurdle"), "Best null Newey–West t", "95th percentile")
    observed = result.hurdle.get("observed", {})
    if not isinstance(observed, dict):
        observed = {}
    if not observed:
        observed = {row.hypothesis_id: row.nw_t for row in result.trials if row.split == "discovery"}
    ranked = sorted(((key, val) for key, val in observed.items() if isinstance(val, (int, float))),
                    key=lambda pair: pair[1], reverse=True)[:5]
    if ranked:
        fig.add_trace(go.Scatter(x=[x for _, x in ranked], y=[0] * len(ranked), text=[h for h, _ in ranked],
                                 mode="markers", marker=dict(color=MINT, size=10, symbol="diamond"),
                                 name="Top observed hypotheses", hovertemplate="%{text}<br>t=%{x:.2f}<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Best result chance found in each of {len(null_max)} re-runs of the same search. Green diamonds show the strongest observed results.")


def show_verdicts(result):
    st.subheader("Verdicts")
    by_id = {h.id: h for h in result.hypotheses}
    trials = {(row.hypothesis_id, row.split): row for row in result.trials}
    for v in sorted(result.verdicts, key=lambda v: LABEL_ORDER.index(v.label)):
        h = by_id.get(v.hypothesis_id)
        discovery = trials.get((v.hypothesis_id, "discovery"))
        facts = verdict_facts(v, discovery)
        with st.container(border=True):
            st.markdown(f":{LABEL_COLORS[v.label]}-background[**{v.label}**]  ·  `{v.hypothesis_id}`")
            if h:
                st.write(plain_spec(h.spec))
            if discovery and discovery.dup_of and discovery.dup_of != v.hypothesis_id:
                st.caption(f"Cluster representative {discovery.dup_of} supplied this verdict.")
            text = v.explanation
            st.write(text if text and check_numbers(text, facts) else explain_label(v.label, facts))
            for reason in v.reasons:
                badge = "PASS" if reason.passed else "FAIL"
                st.text(f"{badge}  {reason.test} · {reason.detail}")
            mde = number(v.power.mde_sharpe_80 if v.power.years > 0 else None)
            record = number(v.power.mintrl_years, 1)
            power_line = f"This sample could only detect Sharpe ≥ {mde} at 80% power." if mde != "—" else "Detectable Sharpe unavailable."
            if record != "—":
                power_line += f" Minimum track record for the observed Sharpe: {record} years."
            st.caption(power_line)
            post = trials.get((v.hypothesis_id, "post_cutoff"))
            if v.dsr is not None:
                st.caption(f"Deflated Sharpe probability · {pct(v.dsr)}")
            if v.holdout_opened:
                post_result = next((r for r in v.reasons if r.test == "hindsight"), None)
                post_value = post.sharpe_net_ann if post else post_result.value if post_result else None
                state = "Passed" if v.holdout_passed else "Failed" if v.holdout_passed is False else "Consumed"
                st.caption(f"One-shot holdout · {state}"
                           f"  |  Post-cutoff net Sharpe · {number(post_value)}")
            with st.expander("Raw JSON evidence"):
                st.json({"verdict": v.model_dump(mode="json"),
                         "hypothesis": h.model_dump(mode="json") if h else None,
                         "discovery_trial": discovery.model_dump(mode="json") if discovery else None})


def edit_future(result):
    with st.expander("Edit the future", expanded=True):
        st.caption("Only observations strictly after the chosen decision month are changed. The planted control sits outside the ledger and search.")
        hs = [CONTROL] + result.hypotheses
        by_id = {h.id: h for h in hs}
        selected = st.selectbox("Hypothesis", list(by_id), format_func=lambda key: (
            "Planted leaky control · centred SPX average" if key == CONTROL.id
            else f"{key} · {by_id[key].spec.series} / {by_id[key].spec.feature}"
        ))
        months = [day.date() for day in pd.date_range(*SPLITS["discovery"], freq="ME")]
        stamp = st.select_slider("Decision month", options=months, value=months[len(months)//2])
        multiplier = st.slider("Future multiplier", .5, 1.5, 1.3, step=.05)
        audit_source = "fixture" if result.demo else active_data_source()
        audit_key = (selected, stamp, multiplier, result.demo, audit_source)
        if st.button("Rerun with edited future"):
            try:
                h = by_id[selected]
                if result.demo:
                    evidence = sample_edit(h.spec, demo_panel(), stamp, multiplier)
                else:
                    audit = require("data.audit")
                    evidence = audit.edit_future_demo(h.spec, require("data.pit").decision_panel(), stamp, multiplier)
                st.session_state["future_edit"] = (audit_key, evidence)
            except (IntegrationMissing, FileNotFoundError, KeyError, ValueError) as exc:
                st.error(f"Future edit unavailable · {exc}")
        prior = st.session_state.get("future_edit")
        if prior and prior[0] == audit_key:
            if audit_source == "synthetic_cache":
                st.warning("This counterexample uses the synthetic cache currently on disk.")
            e = prior[1]
            before, after = e["before"], e["after"]
            if "—" in {number(before), number(after)}:
                st.warning("The feature is unavailable for this decision month.")
            else:
                a, b = st.columns(2)
                a.metric("Feature before", number(before, 6))
                b.metric("Feature after", number(after, 6))
                st.caption(f"Position before {number(e['position_before'])} · after {number(e['position_after'])}")
                if e["passed"]:
                    st.success(f"PASS · Edits after {stamp:%b %Y} left the feature and position at that month unchanged.")
                else:
                    st.error(f"LEAK · Editing only observations after {stamp:%b %Y} changed evidence at that decision month.")


shell("Discover")
heading("01", "Discover", "A counted search with independent checks and one-shot holdout evidence.")
text = st.text_area("Market thesis", DEFAULT_THESIS, max_chars=500)
a, b = st.columns(2)
n = a.slider("Hypotheses", 10, 30, N_HYPOTHESES_DEFAULT)
mode = b.radio("Mode", ["thesis", "explore"], horizontal=True)
left, right, fixture = st.columns(3)
if left.button("Generate & test", type="primary", disabled=offline()):
    with st.status("Running research batch", expanded=True) as status:
        try:
            st.session_state["discovery"] = live_discovery(text, n, mode, status.write)
            st.session_state["m3_live"] = True
            status.update(label="Batch complete", state="complete")
        except BlockingIOError:
            status.update(label="A research batch is already running", state="error")
        except (LLMError, IntegrationMissing, FileNotFoundError, ValueError, RuntimeError) as exc:
            status.update(label="Batch unavailable", state="error")
            status.write(str(exc))
if right.button("Replay cached run", type="secondary"):
    with st.status("Loading recorded evidence", expanded=False) as status:
        st.session_state["discovery"] = cached_discovery()
        st.session_state["m3_live"] = False
        status.update(label="Cached run loaded", state="complete")
if fixture.button("Run engine on fixtures"):
    with st.status("Running M1 on synthetic inputs", expanded=False) as status:
        try:
            st.session_state["discovery"] = fixture_discovery(n)
            st.session_state["m3_live"] = False
            status.update(label="Fixture batch complete", state="complete")
        except (IntegrationMissing, FileNotFoundError, ValueError, RuntimeError) as exc:
            status.update(label="Fixture batch unavailable", state="error")
            status.write(str(exc))
if offline():
    st.caption("Offline mode · replay the recorded batch below.")
if result := st.session_state.get("discovery"):
    if result.data_source == "engine_fixture":
        st.warning("Synthetic inputs · these verdicts were computed by the M1 engine.")
    else:
        source_note(result.demo, "Discovery replay", result.data_source or "unknown")
    show_ledger(result)
    show_hurdle(result)
    edit_future(result)
    show_verdicts(result)
else:
    st.info("Run a thesis or replay the bundled batch to inspect the ledger, luck hurdle and verdicts.")
