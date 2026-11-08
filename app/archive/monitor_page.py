from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agents.explainer import explain_frame
from app.artifacts import monitor_benchmark, replay, run_file
from app.fixtures import RULES
from app.present import AMBER, MINT, RED, heading, number, plot, shell, source_note
from config import CALIB_WINDOW, CUSUM_K, REPLAY_START, ROLL_WIN, ROLL_WIN_BTC


def current_rows(frames, stamp):
    return {f.rule_id: f for f in frames if f.as_of == stamp}


def readouts(current):
    rows = []
    for key, label in RULES.items():
        f = current.get(key)
        if f is None:
            rows.append({"Rule": label, "Data": "Unavailable", "Strength [range]": "—", "CUSUM / h": "—", "State": "—"})
            continue
        strength, change = f.strength, f.change or {}
        detector = change and not change.get("display_only")
        rows.append({
            "Rule": label, "Data": f.integrity.get("status", "unknown").replace("_", " ").title(),
            "Strength [range]": f"{number(strength.get('est'))} [{number(strength.get('lo'))}, {number(strength.get('hi'))}]",
            "CUSUM / h": f"{number(change.get('cusum'))} / {number(change.get('threshold'))}" if detector else "—",
            "State": ("ALARM" if change.get("alarm") else
                      "Quiet" if detector and change.get("cusum") is not None else
                      "Warm-up" if detector else "Display only"),
        })
    st.dataframe(rows, hide_index=True, use_container_width=True)


def charts(history):
    x = [f.as_of for f in history]
    fig = go.Figure()
    fig.add_scatter(x=x, y=[f.strength.get("lo") for f in history], line=dict(width=0),
                    hoverinfo="skip", showlegend=False)
    fig.add_scatter(x=x, y=[f.strength.get("hi") for f in history], line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(86,221,179,0.17)", name="Bootstrap range")
    fig.add_scatter(x=x, y=[f.strength.get("est") for f in history], line=dict(color=MINT), name="Strength")
    st.plotly_chart(plot(fig, y="Rule strength"), use_container_width=True, key="strength")
    if any(f.change and not f.change.get("display_only") for f in history):
        fig = go.Figure()
        fig.add_scatter(x=x, y=[(f.change or {}).get("cusum") for f in history],
                        line=dict(color=AMBER), name="CUSUM")
        fig.add_scatter(x=x, y=[(f.change or {}).get("threshold") for f in history],
                        line=dict(color=RED, dash="dash"), name="Locked threshold")
        st.plotly_chart(plot(fig, y="Change evidence"), use_container_width=True, key="cusum")


def relative_delay(months, reference):
    if months == 0:
        return f"in the same month as {reference}"
    return f"{abs(months)} months {'after' if months > 0 else 'before'} {reference}"


def freeze_card(frame, current, history, demo, meta, locked, calibration_end):
    change = frame.change or {}
    strength = frame.strength
    control = current.get("vix_control")
    control_status = ("ALARM" if (control.change or {}).get("alarm") else "Quiet") if control else "Unavailable"
    with st.container(border=True):
        st.markdown(":red-background[**FIRST ALARM · REPLAY PAUSED**]")
        title = "Calibration-period evidence on" if frame.as_of <= calibration_end else "As known on"
        st.subheader(f"{title} {frame.as_of:%d %B %Y}")
        st.write(RULES.get(frame.rule_id, frame.rule_id))
        a, b, c = st.columns(3)
        a.metric("Strength", number(strength.get("est")))
        b.metric("CUSUM / threshold", f"{number(change.get('cusum'))} / {number(change.get('threshold'))}")
        c.metric("VIX control", control_status)
        st.caption(f"Data integrity · {frame.integrity.get('status', 'unknown')}   |   "
                   f"Strength range · {number(strength.get('lo'))} to {number(strength.get('hi'))}")
        facts = {
            "rule": RULES.get(frame.rule_id, frame.rule_id), "est": number(strength.get("est")),
            "lo": number(strength.get("lo")), "hi": number(strength.get("hi")),
            "cusum": number(change.get("cusum")), "h": number(change.get("threshold")),
            "alarm": bool(change.get("alarm")), "integrity": frame.integrity.get("status"),
        }
        st.write(explain_frame(frame, {key: val for key, val in facts.items() if val != "—"}))
        if len(history) > 1:
            first = history[0].as_of
            months = (frame.as_of.year - first.year) * 12 + frame.as_of.month - first.month
            st.caption(f"First alarm arrived {months} months after the displayed replay began.")
        if meta.get("first_alarm") == frame.as_of.isoformat():
            if (months := meta.get("delay_vs_2022_01_m")) is not None:
                st.caption(f"The alarm came {relative_delay(months, 'January 2022')}.")
            if (months := meta.get("delay_vs_marker_m")) is not None and (marker := meta.get("expost_marker")):
                st.caption(f"The alarm came {relative_delay(months, marker)}, a break marker identified retrospectively.")
        rows, break_demo = monitor_benchmark()
        delays = [
            row.delay_m for row, raw in rows
            if row.kind == "break" and row.type == frame.rule_id and row.delay_m is not None
            and (
                (demo and break_demo) or (
                    not demo and not break_demo and locked.get("input_id")
                    and locked["input_id"] == raw.get("input_id")
                    and locked.get("h") == raw.get("calibration_h")
                )
            )
        ]
        if delays:
            st.caption(f"Planted-break median delay among detections · {pd.Series(delays).median():.1f} months.")


shell("Monitor")
heading("03", "Monitor", "Replay market rules one month at a time with the detector settings locked.")
frames, demo = replay()
meta = {} if demo else run_file("frames_meta.json", default={})
source_note(demo, "Monitor replay", meta.get("data_source", "unknown"))
available = list(dict.fromkeys(f.rule_id for f in frames))
ordered = [name for name in RULES if name in available] + [name for name in available if name not in RULES]
rule = st.selectbox("Rule", ordered, format_func=lambda key: RULES.get(key, key), key="monitor_rule")
rule_meta = meta.get("rules", {}).get(rule, {})
cal = rule_meta.get("locked") or run_file("calib.json", demo, {}).get(rule, {})
calibration_window = cal.get("calib_window") or meta.get("calib_window") or CALIB_WINDOW
calibration_end = date.fromisoformat(calibration_window[1])
st.markdown("#### Locked parameters")
if rule == "curve_recession":
    st.caption("Display-only rule. The replay does not fit a CUSUM threshold for this rule.")
else:
    window = rule_meta.get("window", ROLL_WIN_BTC if rule == "btc_diversifier" else ROLL_WIN)
    st.caption(f"Window {window} months  ·  k {cal.get('k', CUSUM_K)}  ·  h {number(cal.get('h'))}  ·  "
               f"μ₀ {number(cal.get('mu0'))}  ·  σ₀ {number(cal.get('sd0'))}  ·  "
               f"Calibration {calibration_window[0]} to {calibration_window[1]}")
rule_frames = [f for f in frames if f.rule_id == rule and f.as_of >= date.fromisoformat(REPLAY_START)]
dates = sorted({f.as_of for f in frames if f.as_of >= date.fromisoformat(REPLAY_START)})
if not dates:
    st.info("No replay frames are available for the configured date range.")
    st.stop()
first_alarm = next((f.as_of for f in rule_frames
                    if f.first_alarm == f.as_of), None)

if st.session_state.get("previous_monitor_rule") != rule:
    st.session_state["previous_monitor_rule"] = rule
    st.session_state["playing"] = False
    st.session_state["alarm_seen"] = False
if st.session_state.get("replay_date") not in dates:
    st.session_state["replay_date"] = dates[0]


@st.fragment(run_every=.3 if st.session_state["playing"] else None)
def timeline():
    playing = st.session_state["playing"]
    if playing:
        cursor = st.session_state["playback_cursor"]
        next_i = min(dates.index(cursor) + 1, len(dates) - 1)
        cursor = dates[next_i]
        st.session_state["playback_cursor"] = cursor
        st.session_state["replay_date"] = cursor
        if (cursor == first_alarm and not st.session_state["alarm_seen"]) or next_i == len(dates) - 1:
            st.session_state["playing"] = False
            st.session_state["alarm_seen"] = cursor == first_alarm
            st.rerun()

    stamp = st.select_slider("As-of month", dates, key="replay_date", format_func=lambda t: f"{t:%b %Y}",
                             disabled=playing)
    if playing:
        stamp = st.session_state["playback_cursor"]
    left, middle, _ = st.columns([1, 1, 3])
    if left.button("Pause" if playing else "Play", disabled=stamp == dates[-1]):
        st.session_state["playing"] = not playing
        st.session_state["playback_cursor"] = stamp
        if stamp == first_alarm:
            st.session_state["alarm_seen"] = True
        st.rerun()
    if middle.button("Reset replay"):
        st.session_state["reset_replay"] = True
        st.rerun()
    st.markdown(f"#### Evidence through {stamp:%B %Y}")
    if rule != "curve_recession" and stamp <= calibration_end:
        st.caption("Calibration-period context. The locked detector uses data through "
                   f"{calibration_end:%B %Y}, which was still being observed at this replay date.")
    current = current_rows(frames, stamp)
    history = [f for f in rule_frames if f.as_of <= stamp]
    if stamp == first_alarm and rule in current:
        freeze_card(current[rule], current, history, demo, rule_meta, cal, calibration_end)
    readouts(current)
    if history:
        charts(history)
    else:
        st.caption("The selected rule has insufficient history at this date.")


if st.session_state.pop("reset_replay", False):
    st.session_state["playing"] = False
    st.session_state["alarm_seen"] = False
    st.session_state["replay_date"] = dates[0]
timeline()
