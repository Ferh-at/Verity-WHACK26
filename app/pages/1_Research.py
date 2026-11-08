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
from app.present import (LABEL_COLORS, LABEL_ORDER, MINT, RED, heading, histogram, number, plain_spec, plot,
                         shell, source_note, verdict_facts)
from app.runner import IntegrationMissing, active_data_source, live_research, require
from config import N_HYPOTHESES_DEFAULT, NOMINATE_MAX, RESEARCH_ROUNDS, SPLITS

TEST_NAMES = {
    "leak_audit": "Edit-the-future leak audit", "sign": "Right sign", "block_null": "Block bootstrap",
    "luck_hurdle": "Luck hurdle", "by_fdr": "False-discovery control (BY)", "outlier": "Outlier robustness",
    "regime": "Sub-period check", "costs": "Cost stress", "factor": "Factor check", "vendor": "Vendor check",
    "validation": "Validation 2015–2019", "holdout": "One-shot holdout", "hindsight": "Post-cutoff check",
}


def show_loop(result):
    st.subheader("How the AI researched")
    hurdle = result.hurdle or {}
    rounds = result.rounds
    a, b, c, d = st.columns(4)
    a.metric("Ideas tried", hurdle.get("ideas_tried", len(result.hypotheses)))
    b.metric("Effective trials", hurdle.get("effective_trials", "—"))
    c.metric("Rounds", len(rounds) or "—")
    d.metric("Luck hurdle · final NW t", number(hurdle.get("hurdle")))
    if not rounds:
        st.caption("This run was recorded before the AI review loop existed, so it has one round and no reviewer.")
        return
    fig = go.Figure(go.Scatter(
        x=[f"Round {r['round']}<br>{r['ideas_tried']} ideas" for r in rounds], y=[r["hurdle"] for r in rounds],
        mode="lines+markers+text", text=[number(r["hurdle"]) for r in rounds], textposition="top center",
        line=dict(color=RED), marker=dict(size=10)))
    st.plotly_chart(plot(fig, y="Luck hurdle (NW t)", height=240, legend=False), use_container_width=True)
    st.caption("Every idea the AI adds is charged to the same ledger, so the bar for all ideas can rise "
               "but never fall.")
    for r in rounds:
        status = r.get("status", {})
        counts = " · ".join(f"{status[k]} {k}" for k in ("survives", "fragile", "weak", "killed") if status.get(k))
        with st.container(border=True):
            st.markdown(f"**Round {r['round']}** · engine tested {len(r['new'])} new ideas · {counts}")
            who = "Engine fallback" if r.get("fallback") else "AI reviewer"
            st.markdown(f"*{who}:* {r['assessment']}")
            if r["challenges"]:
                st.markdown("**Challenges** (kill-only tests the AI asked for)")
                st.dataframe(pd.DataFrame([{
                    "idea": c["hypothesis_id"], "test": TEST_NAMES.get(c["type"], c["type"]),
                    "why": c["rationale"],
                    "outcome": ("PASS" if c.get("passed") else "FAIL" if c.get("passed") is False else "—")
                               + f" · {c.get('detail', '')}",
                } for c in r["challenges"]]), hide_index=True, use_container_width=True)
            if r["refinements"]:
                st.markdown("**Refinements** (new ideas, charged to the ledger)")
                st.dataframe(pd.DataFrame([{
                    "new idea": x["id"], "responds to": x["based_on"],
                    "because": TEST_NAMES.get(x["because"], x["because"]), "rationale": x["rationale"],
                } for x in r["refinements"]]), hide_index=True, use_container_width=True)
            decision = ("Run another round" if r["decision"] == "refine"
                        else f"Stop and confirm · nominated for the holdout: {', '.join(r['nominate']) or 'none'}")
            st.caption(f"Decision · {decision}")
            if r.get("dropped"):
                st.caption("Refused by code: " + " ".join(r["dropped"]))


def show_hurdle(result):
    st.subheader("The luck hurdle")
    null_max = result.hurdle.get("null_max", [])
    if not null_max:
        st.caption("No search-level null was written for this run.")
        return
    fig = histogram(null_max, result.hurdle.get("hurdle"), "Best NW t found by luck in one scrambled re-run",
                    "Hurdle (95th percentile)")
    observed = result.hurdle.get("observed") or {}
    ranked = sorted(((k, v) for k, v in observed.items() if isinstance(v, (int, float))),
                    key=lambda pair: pair[1], reverse=True)[:5]
    if ranked:
        fig.add_trace(go.Scatter(x=[x for _, x in ranked], y=[0] * len(ranked), text=[h for h, _ in ranked],
                                 mode="markers", marker=dict(color=MINT, size=11, symbol="diamond"),
                                 name="Best real ideas", hovertemplate="%{text}<br>t=%{x:.2f}<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Bars: the best result pure chance produced in each of {len(null_max)} re-runs of the entire search "
               "with the signals shifted out of time. Green diamonds: our five strongest ideas. "
               "Only ideas to the right of the red line beat luck.")


def origin(result, hid):
    rnd = result.round_of.get(hid)
    meta = result.lineage.get(hid)
    if meta:
        return (f"Round {rnd} · refined by the AI from {meta['based_on']} "
                f"(responding to: {TEST_NAMES.get(meta['because'], meta['because'])})")
    return f"Round {rnd} · generated from the thesis" if rnd else None


def card(result, v, h, discovery):
    facts = verdict_facts(v, discovery)
    with st.container(border=True):
        st.markdown(f":{LABEL_COLORS[v.label]}-background[**{v.label}**]  ·  `{v.hypothesis_id}`")
        if where := origin(result, v.hypothesis_id):
            st.caption(where)
        if h:
            st.write(plain_spec(h.spec))
        if discovery and discovery.dup_of and discovery.dup_of != v.hypothesis_id:
            st.caption(f"Duplicate of {discovery.dup_of} (returns more than 90% correlated), so it shares that verdict.")
        text = v.explanation
        st.write(text if text and check_numbers(text, facts) else explain_label(v.label, facts))
        for reason in v.reasons:
            badge = "PASS" if reason.passed else "FAIL"
            st.text(f"{badge}  {TEST_NAMES.get(reason.test, reason.test)} · {reason.detail}")
        mde = number(v.power.mde_sharpe_80 if v.power.years > 0 else None)
        if mde != "—":
            st.caption(f"Power · with {v.power.years:.1f} years of discovery data this test could only detect "
                       f"an annual Sharpe of {mde} or more (80% power).")
        if v.holdout_opened:
            state = "passed" if v.holdout_passed else "failed" if v.holdout_passed is False else "already used"
            st.caption(f"One-shot holdout · {state}")
        elif v.label == "Promising":
            st.caption("One-shot holdout · not opened (not nominated by the reviewer)")
        with st.expander("Raw JSON evidence"):
            st.json({"verdict": v.model_dump(mode="json"), "hypothesis": h.model_dump(mode="json") if h else None,
                     "discovery_trial": discovery.model_dump(mode="json") if discovery else None})


def show_verdicts(result):
    st.subheader("Verdicts")
    by_id = {h.id: h for h in result.hypotheses}
    trials = {(row.hypothesis_id, row.split): row for row in result.trials}
    groups = {label: [v for v in result.verdicts if v.label == label] for label in LABEL_ORDER}
    shown = [label for label in LABEL_ORDER if groups[label]]
    for tab, label in zip(st.tabs([f"{label} ({len(groups[label])})" for label in shown]), shown):
        with tab:
            for v in groups[label]:
                card(result, v, by_id.get(v.hypothesis_id), trials.get((v.hypothesis_id, "discovery")))


def edit_future(result):
    with st.expander("Edit the future (look-ahead leak audit)", expanded=False):
        st.caption("Pick a decision month and scale every value after it. A clean signal cannot change at "
                   "that month. The planted control deliberately uses next month's data and is never traded.")
        hs = [CONTROL] + result.hypotheses
        by_id = {h.id: h for h in hs}
        selected = st.selectbox("Hypothesis", list(by_id), format_func=lambda key: (
            "Planted leaky control · centred SPX average" if key == CONTROL.id
            else f"{key} · {by_id[key].spec.series} / {by_id[key].spec.feature}"))
        months = [day.date() for day in pd.date_range(*SPLITS["discovery"], freq="ME")]
        stamp = st.select_slider("Decision month", options=months, value=months[len(months) // 2])
        multiplier = st.slider("Future multiplier", .5, 1.5, 1.3, step=.05)
        audit_source = "fixture" if result.demo else active_data_source()
        audit_key = (selected, stamp, multiplier, result.demo, audit_source)
        if st.button("Rerun with edited future"):
            try:
                h = by_id[selected]
                if result.demo:
                    evidence = sample_edit(h.spec, demo_panel(), stamp, multiplier)
                else:
                    evidence = require("data.audit").edit_future_demo(
                        h.spec, require("data.pit").decision_panel(), stamp, multiplier)
                st.session_state["future_edit"] = (audit_key, evidence)
            except (IntegrationMissing, FileNotFoundError, KeyError, ValueError) as exc:
                st.error(f"Future edit unavailable · {exc}")
        prior = st.session_state.get("future_edit")
        if prior and prior[0] == audit_key:
            e = prior[1]
            before, after = e["before"], e["after"]
            if "—" in {number(before), number(after)}:
                st.warning("The feature is unavailable for this decision month.")
            else:
                a, b = st.columns(2)
                a.metric("Signal at that month · before", number(before, 6))
                b.metric("Signal at that month · after", number(after, 6))
                if e["passed"]:
                    st.success(f"PASS · Editing data after {stamp:%b %Y} left the signal at {stamp:%b %Y} unchanged.")
                else:
                    st.error(f"LEAK · Editing only data after {stamp:%b %Y} changed the signal at {stamp:%b %Y}.")


def show_ledger(result):
    discovery = [row for row in result.trials if row.split == "discovery"]
    if not discovery:
        return
    by_id = {h.id: h for h in result.hypotheses}
    with st.expander(f"Full trial ledger ({len(discovery)} ideas, discovery period)"):
        st.dataframe(pd.DataFrame([{
            "id": row.hypothesis_id, "round": result.round_of.get(row.hypothesis_id),
            "feature": by_id[row.hypothesis_id].spec.feature if row.hypothesis_id in by_id else None,
            "series": by_id[row.hypothesis_id].spec.series if row.hypothesis_id in by_id else None,
            "asset": by_id[row.hypothesis_id].spec.asset if row.hypothesis_id in by_id else None,
            "duplicate of": row.dup_of, "months": row.n_obs, "net Sharpe": row.sharpe_net_ann, "NW t": row.nw_t,
            "p (bootstrap)": row.p_block, "p (search)": row.p_search, "q (BY)": row.q_by,
        } for row in discovery]), hide_index=True, use_container_width=True)


shell("Research")
heading("01", "Research loop", "The AI proposes, the engine tests, the AI reviews. Only statistics can pass an idea.")
text = st.text_area("Market thesis", DEFAULT_THESIS, max_chars=500)
n = st.slider("Hypotheses in the first round", 10, 30, N_HYPOTHESES_DEFAULT)
st.caption(f"Up to {RESEARCH_ROUNDS} rounds, fixed before the run. The reviewer can nominate at most "
           f"{NOMINATE_MAX} ideas for the one-shot holdout.")
left, right, _ = st.columns([1, 1, 2])
if left.button("Run AI research loop", type="primary", disabled=offline()):
    with st.status("Running the research loop", expanded=True) as status:
        try:
            st.session_state["discovery"] = live_research(text, n, status.write)
            st.session_state["live_run"] = True
            status.update(label="Research run complete", state="complete")
        except BlockingIOError:
            status.update(label="A research run is already in progress", state="error")
        except (LLMError, IntegrationMissing, FileNotFoundError, ValueError, RuntimeError) as exc:
            status.update(label="Research run unavailable", state="error")
            status.write(str(exc))
if right.button("Replay last run", type="secondary"):
    st.session_state["discovery"] = cached_discovery()
    st.session_state["live_run"] = False
if offline():
    st.caption("Offline mode · replay the last recorded run.")

if result := st.session_state.get("discovery"):
    source_note(result.demo, "Research run", result.data_source or "unknown")
    if result.thesis:
        st.markdown(f"**Thesis tested:** {result.thesis}")
    show_loop(result)
    show_hurdle(result)
    show_verdicts(result)
    edit_future(result)
    show_ledger(result)
else:
    st.info("Run a thesis, or replay the last recorded run.")
