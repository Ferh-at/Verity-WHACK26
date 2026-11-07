from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.artifacts import benchmark, calibration, run_file
from app.metrics import alarm_rate, benchmark_metrics
from app.present import heading, number, pct, shell, source_note

shell("Home")
heading("RESEARCH TERMINAL", "Discovery Lab", "A research workflow for testing how much of a discovery survives scrutiny.")
st.markdown(
    "Give the lab a market thesis. It generates precise hypotheses, records what it tried, "
    "and measures how often chance would have found a stronger result. The same lab replays "
    "market rules one month at a time to show when the evidence changes."
)
st.markdown(
    "> Finding patterns in data is easy; knowing which ones to trust is hard. Create an AI researcher "
    "that tests its own ideas and can tell a real discovery from a lucky one, for example in markets, "
    "sport, medicine, online experiments..."
)
left, right, _ = st.columns([1, 1, 2])
if left.button("Discover", type="primary", use_container_width=True):
    st.switch_page("pages/1_Discover.py")
if right.button("Monitor", use_container_width=True):
    st.switch_page("pages/3_Monitor.py")

st.divider()
st.subheader("Evidence at a glance")
bench, is_demo = benchmark()
metrics = benchmark_metrics(bench, None if is_demo else run_file("benchmark_metrics.json"))
cal, cal_demo = calibration()
bonds = cal.get("bonds_protect", {})
if is_demo and cal_demo:
    source_note(True, "Benchmark and calibration")
else:
    if is_demo:
        source_note(True, "Planted-trap benchmark")
    else:
        st.caption("Measured engine run on planted synthetic signals and traps.")
    source_note(cal_demo, "Monitor calibration", bonds.get("data_source", "unknown"))
a, b, c = st.columns(3)
ours = metrics["ours"]["mean"] if metrics["ours"] else None
naive = metrics["naive"]["mean"] if metrics["naive"] else None
a.metric("False discovery rate · lab / naive", f"{pct(ours, 2)} / {pct(naive, 2)}")
b.metric("Planted leaks caught", pct(metrics["leak_catch"]))
c.metric("False alarms per 20 years · bonds", number(alarm_rate(bonds)))
st.page_link("pages/2_Proof.py", label="Inspect the benchmark and calibration")
