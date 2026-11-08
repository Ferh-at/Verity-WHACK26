from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import os
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from app.present import heading, shell
from catalog import CATALOG
from config import SPLITS

VENDORS = {
    "yahoo": ("Yahoo Finance", "https://finance.yahoo.com/quote/{symbol}"),
    "stooq": ("Stooq", "https://stooq.com/q/?s={symbol}"),
    "fred": ("FRED, Federal Reserve Bank of St. Louis", "https://fred.stlouisfed.org/series/{symbol}"),
    "nber": ("NBER Business Cycle Dating Committee", "https://www.nber.org/research/business-cycle-dating"),
}
METHODS = [
    ("Luck hurdle (re-run the whole search on null data)",
     "White, H. (2000). A Reality Check for Data Snooping. Econometrica 68(5)."),
    ("Block bootstrap p-values that respect autocorrelation",
     "Politis, D. & Romano, J. (1994). The Stationary Bootstrap. JASA 89(428); block length: "
     "Politis & White (2004), Patton, Politis & White (2009)."),
    ("t-statistics robust to correlated months",
     "Newey, W. & West, K. (1987). Econometrica 55(3); lag rule: Newey & West (1994). Rev. Econ. Studies 61(4)."),
    ("False-discovery control across all ideas tried",
     "Benjamini, Y. & Yekutieli, D. (2001). The control of the false discovery rate under dependency. "
     "Annals of Statistics 29(4)."),
    ("Minimum track record and deflated Sharpe (shown in the raw evidence)",
     "Bailey, D. & López de Prado, M. (2012, 2014). The Sharpe Ratio Efficient Frontier; The Deflated Sharpe Ratio."),
]
LIBRARIES = ("pandas, NumPy, SciPy, statsmodels, arch (block length), yfinance, fredapi, pandas-datareader, "
             "pydantic, OpenAI Python SDK (OpenAI-compatible endpoint), Streamlit, Plotly")


@st.cache_data(show_spinner=False)
def coverage():
    from data.store import cache_source, read_cache
    rows = []
    for sid, entry in CATALOG.items():
        for vendor, symbol in entry["sources"].items():
            df = read_cache(vendor, sid)
            if df is None or df.empty:
                continue
            values = df.dropna(subset=["value"])
            name, link = VENDORS[vendor]
            rows.append({
                "Series": sid, "Description": entry["desc"], "Source": name,
                "Symbol": symbol if vendor != "nber" else "announcement table",
                "Link": link.format(symbol=symbol),
                "First": f"{values['date'].min():%Y-%m}", "Last": f"{values['date'].max():%Y-%m}",
                "Usable from": ("next month-end (release lag)" if entry["freq"] == "monthly"
                                else "announcement date" if entry["freq"] == "events" else "same month-end"),
            })
    return pd.DataFrame(rows), cache_source()


shell("Data and methods")
heading("03", "Data and methods", "Where every number comes from, and the published methods behind each check.")

st.subheader("Market and macro data")
table, source = coverage()
if source == "synthetic_cache":
    st.warning("The app is currently reading the synthetic test cache, not market data. Run `python -m data.fetch`.")
st.markdown(
    "All series are downloaded once with `python -m data.fetch`, stored as month-end values in `data/cache/` with "
    "the date each value became public, and read from that cache by the app. A value is only used from the month "
    "it was published, so no test can see data that was not yet known."
)
if not table.empty:
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Link": st.column_config.LinkColumn("Link", display_text="source")})
st.caption("Prices are Yahoo Finance daily closes (auto-adjusted), resampled to the last trading day of each month. "
           "The monthly risk-free rate is the 3-month Treasury bill yield (FRED DGS3MO) divided by 1,200 and "
           "lagged one month. Data are used for non-commercial research and education under each provider's terms. "
           "Nothing here is investment advice.")

st.subheader("Time periods")
st.dataframe(pd.DataFrame([
    {"Period": "Discovery", "Dates": f"{SPLITS['discovery'][0][:7]} to {SPLITS['discovery'][1][:7]}",
     "Who sees it": "Engine and AI reviewer (summary statistics only)"},
    {"Period": "Validation", "Dates": f"{SPLITS['validation'][0][:7]} to {SPLITS['validation'][1][:7]}",
     "Who sees it": "Engine only, once, after the reviewer has finished"},
    {"Period": "Holdout (sealed)", "Dates": f"{SPLITS['holdout'][0][:7]} to {SPLITS['holdout'][1][:7]}",
     "Who sees it": "Engine only, one look per nominated idea, logged with a pre-registration hash"},
]), hide_index=True, use_container_width=True)
st.caption("Assets with short histories (Bitcoin) split their own history 60/20/20 instead.")

st.subheader("The language model")
model = os.getenv("LLM_MODEL") or "not configured"
host = urlparse(os.getenv("LLM_BASE_URL") or "").hostname or "provider default"
st.markdown(
    f"Model `{model}` via an OpenAI-compatible endpoint ({host}). It plays two roles, the **generator** and the "
    "**reviewer**. Neither ever sees raw data, validation results or holdout results. Every prompt passes a code "
    "check that blocks any date from 2020 onwards and any number outside the typed evidence fields. Every call is "
    "logged to `runs/llm_calls.jsonl`, and the reviewer's limits (budget, nominations, kill-only challenges) are "
    "enforced in code, not trusted to the model."
)

st.subheader("Methods and references")
st.dataframe(pd.DataFrame(METHODS, columns=["Check", "Reference"]), hide_index=True, use_container_width=True)

st.subheader("Software")
st.caption(f"Open-source libraries: {LIBRARIES}. Built for WHACK 26 by Ferhat Celik, "
           "Efe Ulker and Reyansh Mathur.")
