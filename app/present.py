"""Formatting shared by the UI and the guarded explainer."""
import math

import plotly.graph_objects as go
import streamlit as st

from agents.llm import offline

MINT = "#56DDB3"
RED = "#FF7676"
BLUE = "#7BA9FF"
AMBER = "#F5C579"
MUTED = "#9AACC5"
LABEL_ORDER = ["Replicated", "Promising", "Fragile", "Can't tell yet", "Rejected"]
LABEL_COLORS = dict(zip(LABEL_ORDER, ["green", "blue", "orange", "gray", "red"]))


def shell(title):
    st.set_page_config(page_title=f"{title} · Discovery Lab", page_icon="◆", layout="wide")
    st.sidebar.markdown("### ◆ Discovery Lab")
    st.sidebar.caption("WHACK 26  /  Brevan Howard track")
    if offline():
        st.sidebar.markdown(":orange-background[OFFLINE]")
    elif st.session_state.get("m3_live"):
        st.sidebar.markdown(":green-background[LIVE]")
    else:
        st.sidebar.markdown(":blue-background[CACHED]")
    st.markdown(
        """<style>
        .block-container{padding-top:2.4rem;max-width:1260px}
        div[data-testid="stMetric"]{padding:1rem 1.05rem;background:#142235;border-radius:10px}
        div[data-testid="stMetricValue"]{color:#56DDB3}
        div[data-testid="stAlert"] p{color:#C7DBEC}
        </style>""", unsafe_allow_html=True,
    )


def heading(number, title, description):
    st.markdown(f":green[**{number}**]  /  WHACK 26 · BREVAN HOWARD")
    st.title(title)
    st.caption(description)


def source_note(demo, name, source=None):
    if demo:
        st.info(f"Synthetic demo fixtures · illustrative outputs for {name.lower()}.")
    elif source == "synthetic_cache":
        st.warning(f"Synthetic cache · {name}. These figures are for testing.")
    elif source == "market_cache":
        st.caption(f"Market-data cache · {name}")
    elif source == "external_cache":
        st.warning(f"External cache · {name}. Verify the input source before citing results.")
    elif source == "unknown":
        st.warning(f"Unverified source · {name}. Regenerate the artefacts to record their provenance.")
    else:
        st.caption(f"Team run artefacts · {name}")


def number(value, digits=2):
    try:
        return f"{value:.{digits}f}" if value is not None and math.isfinite(value) else "—"
    except TypeError:
        return "—"


def pct(value, digits=0):
    return "—" if value is None else f"{value:.{digits}%}"


def plain_spec(s, *, spell_out=False):
    words = {1: "one", 3: "three", 6: "six", 12: "twelve", 24: "twenty-four", 36: "thirty-six"}
    window = words.get(s.lookback, "trailing") if spell_out else s.lookback
    names = {
        "mom": f"{window}-month momentum in {s.series}",
        "level": f"the month-end level of {s.series}",
        "change": f"the {window}-month change in {s.series}",
        "zscore": f"the trailing {window}-month z-score of {s.series}",
        "corr": f"the {window}-month correlation of changes in {s.series} and {s.series2}",
        "centered_ma": f"the centred {window}-month average of changes in {s.series}",
        "fullsample_z": f"the full-sample z-score of {s.series}",
    }
    position = "long" if s.direction == 1 else "short"
    reverse = "short" if s.direction == 1 else "long"
    sizing = "Size by trailing z-score." if s.transform == "clip_z" else "Size by sign."
    return f"Trade {s.asset} next month using {names[s.feature]}. Go {position} on a positive signal and {reverse} on a negative signal. {sizing}"


def verdict_facts(v, trial=None):
    facts = {
        "label": v.label, "mde_sharpe_80": number(v.power.mde_sharpe_80 if v.power.years > 0 else None),
        "mintrl_years": number(v.power.mintrl_years, 1), "years": number(v.power.years, 1),
        "power_pct": "80%", "dsr": pct(v.dsr),
    }
    if trial:
        facts.update(sharpe_net_ann=number(trial.sharpe_net_ann), p_block=number(trial.p_block, 3))
    return {key: val for key, val in facts.items() if val != "—"}


def plot(fig, x=None, y=None, height=300, legend=True):
    fig.update_layout(
        template="plotly_dark", height=height, margin=dict(l=12, r=18, t=25, b=10),
        paper_bgcolor="#0B1320", plot_bgcolor="#101C2D",
        xaxis_title=x, yaxis_title=y, showlegend=legend,
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_xaxes(gridcolor="#24364E")
    fig.update_yaxes(gridcolor="#24364E")
    return fig


def histogram(values, threshold, x_label, line_label):
    fig = go.Figure(go.Histogram(x=values, marker_color=BLUE, name="Simulations"))
    if threshold is not None:
        fig.add_vline(x=threshold, line_color=RED, line_dash="dash", annotation_text=line_label)
    return plot(fig, x_label, "Count", legend=False)
