from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.artifacts import MOCKS, benchmark, calibration, monitor_benchmark, read, run_file
from app.fixtures import RULES
from app.metrics import alarm_rate, benchmark_metrics
from app.present import AMBER, BLUE, MINT, heading, histogram, number, pct, plot, shell, source_note
from config import DETECT_WITHIN


def scoreboard(rows, demo):
    st.subheader("Planted-trap scoreboard")
    if demo:
        source_note(True, "Planted-trap benchmark")
    else:
        st.caption("Measured engine run on planted synthetic signals and traps.")
    m = benchmark_metrics(rows, None if demo else run_file("benchmark_metrics.json"))
    a, b, c = st.columns(3)
    a.metric("Seeds", m["seeds"])
    b.metric("Hypotheses", m["trials"])
    c.metric("Can't tell yet", m["uncertain"])
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name, info, color in (("Naive t-test", m["naive"], AMBER), ("Discovery Lab", m["ours"], MINT)):
            if not info:
                continue
            err = None
            if info.get("ci"):
                low, high = info["ci"]
                err = dict(type="data", symmetric=False, array=[high - info["mean"]],
                           arrayminus=[info["mean"] - low])
            fig.add_bar(x=[name], y=[info["mean"]], error_y=err, marker_color=color, name=name,
                        hovertemplate="%{x}<br>FDR %{y:.1%}<extra></extra>")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(plot(fig, y="False discovery rate", legend=False), use_container_width=True)
        st.caption("Mean across seeds. Error bars show the 95% seed bootstrap interval when multiple seeds are available.")
        if m["naive"] is None:
            st.caption("Naive decisions are missing from the team benchmark artefacts.")
    with right:
        recall = sorted((float(sr), rate) for sr, rate in m["recall"].items())
        if recall:
            fig = go.Figure(go.Bar(x=[str(sr) for sr, _ in recall], y=[val for _, val in recall], marker_color=BLUE))
            fig.update_xaxes(type="category")
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            st.plotly_chart(plot(fig, "Planted annual Sharpe", "Recall", legend=False), use_container_width=True)
        else:
            st.caption("Recall by effect size needs planted Sharpe values from the engine.")
    table = pd.DataFrame.from_dict(m["grid"], orient="index").fillna(0).astype(int).sort_index()
    table.index.name = "Trap type"
    noise = run_file("noise_search.json") or read(MOCKS / "m1_benchmark/noise_search.json", {})
    if noise.get("n_searches"):
        st.caption(f"Separate pure-noise run. {pct(noise['share_with_trusted'])} of "
                   f"{noise['n_searches']} searches produced a trusted result "
                   f"with {noise['candidates']} candidates per search.")
    st.dataframe(table, use_container_width=True)


def monitor_proof():
    st.subheader("Monitor calibration")
    cal, cal_demo = calibration()
    records, break_demo = monitor_benchmark()
    if cal_demo != break_demo:
        if cal_demo:
            cal = {}
        else:
            records = []
    rules = list(dict.fromkeys([*cal, *(row.type for row, _ in records if not break_demo)]))
    rule = st.selectbox("Calibration rule", rules, format_func=lambda key: RULES.get(key, key))
    selected = cal.get(rule, {})
    left, right = st.columns(2)
    with left:
        values = selected.get("max_c", [])
        if selected:
            source_note(cal_demo, "Stable simulations", selected.get("data_source", "unknown"))
        st.metric("False alarms per 20 years", number(alarm_rate(selected)))
        if values:
            st.plotly_chart(histogram(values, selected.get("h"), "Maximum CUSUM under stability", "Locked h"),
                            use_container_width=True)
            if selected.get("quantile") is not None:
                st.caption(f"{selected.get('n_sims', len(values)):,} stable simulations. Threshold quantile {selected['quantile']:.3f}.")
        else:
            st.caption("No stable simulation maxima are available for this rule.")
    with right:
        breaks = [(row, raw) for row, raw in records if row.kind == "break" and row.type == rule]
        if not breaks:
            st.caption("No planted-break simulations are available for this rule.")
            return
        provenance = breaks[0][1]
        if cal_demo != break_demo or not break_demo:
            source_note(break_demo, "Planted-break simulations", provenance.get("data_source", "unknown"))
        break_h = provenance.get("calibration_h")
        if break_h is not None and break_h != selected.get("h"):
            st.caption(f"These break results used h = {number(break_h)}. Rerun the break job to use the current calibration.")
        delays = [row.delay_m for row, _ in breaks if row.delay_m is not None]
        within = sum(delay <= DETECT_WITHIN for delay in delays) / len(breaks)
        st.metric(f"Detected within {DETECT_WITHIN} months", pct(within))
        st.plotly_chart(histogram(delays, DETECT_WITHIN, "Months to first alarm after planted break", "Detection window"),
                        use_container_width=True)
        median = number(float(np.median(delays)), 1) if delays else "—"
        st.caption(f"Median among detections {median} months. Misses {len(breaks)-len(delays)} of {len(breaks)}.")
        stable = [row for row, _ in records if row.kind == "stable" and row.type == rule and row.false_alarm is not None]
        if stable:
            share = sum(row.false_alarm for row in stable) / len(stable)
            st.caption(f"{pct(share)} of {len(stable)} separate stable simulations had at least one alarm.")


def probe_chart():
    st.subheader("Model memorisation probe")
    result = run_file("probe.json")
    demo = not result
    result = result or run_file("probe.json", True, {})
    if not result.get("series"):
        st.caption("No completed probe is available.")
        return
    source_note(demo, "Probe answers and truth")
    fig = go.Figure()
    series = result["series"]
    for phase, color in (("pre", BLUE), ("post", MINT)):
        fig.add_bar(name=f"{phase.title()}-cutoff", x=list(series),
                    y=[stats[phase]["share_within_2pct"] for stats in series.values()],
                    marker_color=color, hovertemplate="%{x}<br>%{y:.0%} within tolerance<extra></extra>")
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    fig.update_layout(barmode="group")
    st.plotly_chart(plot(fig, y="Answers within 2% of truth"), use_container_width=True)
    stats = []
    for key, series_stats in series.items():
        for phase in ("pre", "post"):
            item = series_stats[phase]
            error = item["median_error"]
            stats.append({
                "Series": key, "Period": phase, "Answered": f"{item['answered']} / {item['n']}",
                "Median error": (f"{number(error, 3)} pp" if item["unit"] == "pp"
                                 else number(100 * error, 2) + "%" if error is not None else "—"),
                "Pattern flag": series_stats["memorises_pre_cutoff"],
            })
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption("Null and unavailable answers count as misses for accuracy. Median error uses numeric answers. Probe answers are isolated from discovery.")


shell("Proof")
heading("02", "Proof", "Compare against planted ground truth and inspect the alarm calibration.")
records, demo = benchmark()
scoreboard(records, demo)
st.divider()
monitor_proof()
st.divider()
probe_chart()
