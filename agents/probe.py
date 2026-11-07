"""Separate opt-in CLI for the historical memorisation probe."""
import argparse
import math
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, FiniteFloat

from agents.llm import LLMError, ROOT, chat_json, write_json
from config import DATA_END, LLM_CUTOFF, TEMPERATURE

SERIES = {
    "UNRATE": ("US unemployment rate in percent", "fred", "pp"),
    "CPIAUCSL": ("year-on-year percentage change in the US consumer price index", "fred", "pp"),
    "DGS10": ("US ten-year Treasury yield at month-end in percent", "fred", "pp"),
    "SPX": ("S&P 500 index at month-end close", "yahoo", "relative"),
}
PRE = ["2018-03", "2018-11", "2019-06", "2020-04", "2020-12", "2021-07",
       "2022-03", "2022-10", "2023-05", "2023-12"]
SYSTEM = "Answer with JSON only."


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: FiniteFloat | None


def prompt(series_id, t):
    desc = SERIES[series_id][0]
    return (
        f'What was {desc} for {t:%B %Y}? If you do not know, answer null. '
        'Reply as {"value": number|null}'
    )


def truth_series(series_id, cache_dir=ROOT / "data/cache"):
    source = SERIES[series_id][1]
    base = cache_dir / f"{source}_{series_id}"
    if base.with_suffix(".parquet").exists():
        frame = pd.read_parquet(base.with_suffix(".parquet"))
    elif base.with_suffix(".csv").exists():
        frame = pd.read_csv(base.with_suffix(".csv"))
    else:
        raise FileNotFoundError(f"Missing probe truth series {source}_{series_id} in {cache_dir}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["published"] = pd.to_datetime(frame["published"])
    frame = frame[frame["published"].notna() & (frame["published"] <= pd.Timestamp(DATA_END))]
    frame = frame.sort_values("published").drop_duplicates("date", keep="last")
    series = frame.set_index("date")["value"].astype(float).sort_index()
    if series_id == "CPIAUCSL":
        series = series.asfreq("ME").pct_change(12, fill_method=None) * 100
    return series


def _summary(rows, unit):
    # Treat abstentions as misses for accuracy. The median is over numeric
    # predictions, and coverage is displayed alongside it.
    rel = [row["relative_error"] for row in rows if row["relative_error"] is not None]
    errors = [row["error"] for row in rows if row["error"] is not None]
    return {
        "n": len(rows), "answered": sum(row["pred"] is not None for row in rows),
        "median_error": float(pd.Series(errors).median()) if errors else None,
        "median_relative_error": float(pd.Series(rel).median()) if rel else None,
        "share_within_2pct": sum(r <= 0.02 for r in rel) / len(rows) if rows else None,
        "unit": unit,
    }


def run_probe(cache_dir=ROOT / "data/cache", out=ROOT / "runs/probe.json"):
    rows = []
    by_series = {}
    for series_id, (_, _, unit) in SERIES.items():
        series = truth_series(series_id, cache_dir)
        pre = pd.PeriodIndex(PRE, freq="M").to_timestamp(how="end").normalize()
        post = series.loc[series.index > pd.Timestamp(LLM_CUTOFF)].dropna().index[-10:]
        summary = {}
        for group, dates in (("pre", pre), ("post", post)):
            part = []
            for stamp in dates:
                truth = series.get(stamp)
                if truth is None or not math.isfinite(truth):
                    continue
                try:
                    ans = chat_json("probe", SYSTEM, prompt(series_id, stamp),
                                    Answer, TEMPERATURE["probe"])
                    pred, status = ans.value, "abstained" if ans.value is None else "answered"
                except LLMError:
                    pred, status = None, "unavailable"
                absolute = abs(pred - truth) if pred is not None else None
                rel = absolute / abs(truth) if absolute is not None and truth != 0 else None
                part.append({
                    "series": series_id, "group": group, "date": stamp.date().isoformat(),
                    "truth": float(truth), "pred": pred, "status": status, "relative_error": rel,
                    "error": absolute if unit == "pp" else rel,
                })
            summary[group] = _summary(part, unit)
            rows.extend(part)
        pre_med, post_med = (summary[group]["median_relative_error"] for group in ("pre", "post"))
        summary["memorises_pre_cutoff"] = (
            pre_med is not None and post_med is not None and pre_med < 0.02 and post_med > 3 * pre_med
        )
        by_series[series_id] = summary
    value = {"rows": rows, "series": by_series, "cutoff": LLM_CUTOFF}
    write_json(out, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/cache")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/probe.json")
    args = parser.parse_args()
    result = run_probe(args.cache_dir, args.out)
    answered = sum(row["status"] == "answered" for row in result["rows"])
    print(f"Wrote {len(result['rows'])} rows ({answered} numeric responses) to {args.out}")


if __name__ == "__main__":
    main()
