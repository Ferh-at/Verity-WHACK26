from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.artifacts import MOCKS, benchmark, read, run_file
from app.metrics import benchmark_metrics
from app.present import heading, pct, shell, source_note

FLOW = """
digraph {
  rankdir=LR; bgcolor="transparent"; nodesep=0.5; ranksep=0.55;
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=15, color="#2B4362",
        fontcolor="#E9F1FB", margin="0.2,0.12"];
  edge [color="#9AACC5", fontcolor="#C7DBEC", fontname="Helvetica", fontsize=13];
  T [label="Trader\\nthesis", fillcolor="#142235"];
  G [label="AI generator\\nhypotheses", fillcolor="#1E3A66"];
  E [label="Deterministic engine\\n(discovery data)", fillcolor="#134236"];
  R [label="AI reviewer", fillcolor="#1E3A66"];
  C [label="Confirm once\\nvalidation + holdout", fillcolor="#134236"];
  V [label="Verdict", fillcolor="#142235"];
  {rank=same; R; C}
  T -> G -> E -> R;
  R -> G [label="refine (charged)"];
  R -> E [label="challenge (kill-only)"];
  R -> C [label="nominate"];
  C -> V;
}
"""

CHECKS = [
    ("Luck hurdle", "The whole search is re-run 200 times on scrambled data. An idea counts only if it beats "
                    "the best result that luck produced."),
    ("The bar rises", "Every idea the AI tries is counted, across every round. Searching harder makes it "
                      "harder to impress."),
    ("Edit the future", "If changing future data changes a past signal, the signal is cheating and is rejected."),
    ("One-shot holdout", "2020 to 2026 is sealed. Each idea gets one look, only after passing everything else."),
    ("Honest power", "Every verdict says which Sharpe ratio the sample could have detected. Sometimes the "
                     "answer is \"Can't tell yet\"."),
]

shell("Home")
heading("RESEARCH TERMINAL", "Verity",
        "An AI researcher that tests its own ideas and reports which ones survive.")
st.markdown(
    "> Finding patterns in data is easy; knowing which ones to trust is hard. Create an AI researcher "
    "that tests its own ideas and can tell a real discovery from a lucky one."
)
st.markdown(
    "Give Verity a market thesis. The **AI** proposes precise hypotheses, a **deterministic engine** tests them, "
    "and the AI reviews the evidence: it challenges ideas, refines them or nominates the best for a sealed holdout. "
    "The AI can make an idea fail. Only the statistics can make it pass."
)
left, right, _ = st.columns([1, 1, 2])
if left.button("Run the research loop", type="primary", use_container_width=True):
    st.switch_page("pages/1_Research.py")
if right.button("See the proof", use_container_width=True):
    st.switch_page("pages/2_Proof.py")

st.subheader("How it works")
st.graphviz_chart(FLOW, use_container_width=True)
st.caption("Blue boxes are the language model. Green boxes are deterministic code. "
           "The reviewer never sees validation or holdout results.")

st.subheader("Five checks behind every verdict")
for column, (title, text) in zip(st.columns(len(CHECKS)), CHECKS):
    with column:
        st.markdown(f"**{title}**")
        st.caption(text)

st.divider()
st.subheader("Evidence at a glance")
bench, is_demo = benchmark()
metrics = benchmark_metrics(bench, None if is_demo else run_file("benchmark_metrics.json"))
noise = run_file("noise_search.json") or read(MOCKS / "recorded_benchmark/noise_search.json", {})
if is_demo:
    source_note(True, "Planted-trap benchmark")
else:
    st.caption("Measured on synthetic data with planted real signals and planted traps, so the right answer is known.")
a, b, c = st.columns(3)
ours = metrics["ours"]["mean"] if metrics["ours"] else None
naive = metrics["naive"]["mean"] if metrics["naive"] else None
a.metric("False discovery rate · Verity / naive t-test", f"{pct(ours, 2)} / {pct(naive, 2)}")
b.metric("Planted look-ahead leaks caught", pct(metrics["leak_catch"]))
c.metric("Pure-noise searches with a trusted result",
         f"{round(noise['share_with_trusted'] * noise['n_searches'])} of {noise['n_searches']}"
         if noise.get("n_searches") else "—")
st.page_link("pages/2_Proof.py", label="Inspect the benchmark")
