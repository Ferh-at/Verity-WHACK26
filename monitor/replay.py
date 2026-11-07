"""M2.10 Point-in-time replay and freeze frames.

    python -m monitor.replay                      # all rules, REPLAY_START..DATA_END -> runs/frames.jsonl

For every month t and rule, the frame is computed only from pit.panel(t) ("as known on t").
Locked parameters (mu0, sd0, h) come from runs/calib.json, fitted on 2003-2019 data and stable
simulations; they are fixed before the replay starts and shown on screen first.

first_alarm = the first month on or after REPLAY_START where an alarm episode STARTS (C crosses
above h), as known at t. Episodes that started before REPLAY_START are listed as historical alarms
in runs/frames_meta.json.

Files:
  runs/frames.jsonl       one ReplayFrame per (rule, month). change also carries s, z, mu0, sd0, k.
  runs/frames_meta.json   locked parameters, statements, first alarms, historical alarms, delays.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import REPLAY_START, DATA_END, CALIB_WINDOW, CUSUM_K, SEED
from schemas import ReplayFrame
from data import pit
from data.store import ROOT, as_ts, cache_source
from monitor.calibrate import get_params, CALIB_PATH
from monitor.cusum import run_cusum, onsets
from monitor.rules import RULES, strength, strength_series, integrity, curve_spells

FRAMES_PATH = ROOT / "runs" / "frames.jsonl"
META_PATH = ROOT / "runs" / "frames_meta.json"


def _months(start, end, last_available) -> pd.DatetimeIndex:
    end = min(as_ts(end), last_available)
    return pd.date_range(as_ts(start), end, freq="ME")


def _r(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def expost_marker(s_full: pd.Series, start=REPLAY_START, run: int = 6) -> str | None:
    """EX-POST reference month (hindsight; only for reporting a delay, never used by the alarm):
    first month on/after `start` that begins a run of >= `run` months with 12-month strength < 0."""
    neg = (s_full.loc[as_ts(start):] < 0).astype(int)
    streak = neg.groupby((neg != neg.shift()).cumsum()).cumsum() * neg
    hits = streak[streak >= run]
    if hits.empty:
        return None
    end = hits.index[0]
    return str((end - pd.offsets.MonthEnd(run - 1)).date())


def replay(rule_ids=None, start=REPLAY_START, end=DATA_END, out: Path = FRAMES_PATH,
           calib_path: Path = CALIB_PATH) -> None:
    rule_ids = list(rule_ids or RULES.keys())
    full = pit.panel()
    last_avail = full.dropna(how="all", subset=["SPX"]).index.max()
    months = _months(start, end, last_avail)
    params = {r: get_params(r, calib_path) for r in rule_ids if RULES[r]["alarm"]}
    frames, meta = [], {"start": str(as_ts(start).date()), "end": str(months[-1].date()),
                        "data_source": cache_source(), "calib_window": list(CALIB_WINDOW), "rules": {}}
    first = {r: None for r in rule_ids}
    for t in months:
        p = pit.panel(t)
        for r in rule_ids:
            rule = RULES[r]
            st = strength(r, t, panel=p)
            integ = integrity(r, t)
            if not rule["alarm"]:
                change = {"cusum": None, "threshold": None, "alarm": False, "display_only": True}
            else:
                c = params[r]
                s = strength_series(r, p)
                cs = run_cusum(s, c["mu0"], c["sd0"], c["h"], CUSUM_K) if len(s) else None
                if cs is None or t not in cs.index:
                    change = {"cusum": None, "threshold": c["h"], "alarm": False}
                else:
                    on = pd.Series(onsets(cs["alarm"].values), index=cs.index)
                    if first[r] is None:
                        new = on.loc[as_ts(start):t]
                        if new.any():
                            first[r] = new[new].index[0]
                    change = {"cusum": _r(cs.loc[t, "C"]), "threshold": _r(c["h"]), "alarm": bool(cs.loc[t, "alarm"]),
                              "s": _r(s.loc[t]), "z": _r(cs.loc[t, "z"]), "mu0": _r(c["mu0"]), "sd0": _r(c["sd0"]),
                              "k": CUSUM_K}
            frames.append(ReplayFrame(rule_id=r, as_of=t.date(), integrity=integ, strength=st, change=change,
                                      first_alarm=first[r].date() if first[r] is not None else None))
    # ---- meta (computed on the full panel AFTER the replay; for display only)
    for r in rule_ids:
        rule = RULES[r]
        m = {"statement": rule["statement"], "window": rule["window"], "alarm_rule": rule["alarm"],
             "control": bool(rule.get("control", False)),
             "first_alarm": str(first[r].date()) if first[r] is not None else None}
        if rule["alarm"]:
            c = params[r]
            s = strength_series(r, full)
            cs = run_cusum(s, c["mu0"], c["sd0"], c["h"], CUSUM_K)
            on = pd.Series(onsets(cs["alarm"].values), index=cs.index)
            m["locked"] = {"mu0": c["mu0"], "sd0": c["sd0"], "h": c["h"], "k": CUSUM_K,
                           "calib_window": c.get("calib_window"), "n_sims": c.get("n_sims"),
                           "fa_rate_per_20y": c.get("fa_rate_per_20y"), "quantile": c.get("quantile"),
                           "input_id": c.get("input_id"), "data_source": c.get("data_source")}
            m["all_alarm_onsets"] = [str(d.date()) for d in on[on].index]
            m["historical_alarms"] = [d for d in m["all_alarm_onsets"] if d < str(as_ts(start).date())]
            m["n_alarm_months_replay"] = int(cs.loc[as_ts(start):, "alarm"].sum())
            if r == "bonds_protect":
                # ex-post marker: 12-month strength, same statistic, hindsight-only
                from monitor.rules import pairs
                P = pairs(r, full)
                s12 = -P["a"].rolling(12, min_periods=12).corr(P["b"])
                mk = expost_marker(s12)
                m["expost_marker"] = mk
                m["expost_marker_rule"] = "first month from 2015 starting >=6 months of negative 12m strength (hindsight)"
                if mk and first[r] is not None:
                    m["delay_vs_marker_m"] = (first[r].year - as_ts(mk).year) * 12 + first[r].month - as_ts(mk).month
                if first[r] is not None:
                    m["delay_vs_2022_01_m"] = (first[r].year - 2022) * 12 + first[r].month - 1
        else:
            m["spells"] = curve_spells(months[-1], full)
        meta["rules"][r] = m
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text("\n".join(f.model_dump_json() for f in frames) + "\n")
    tmp.replace(out)
    meta_path = out.with_name(f"{out.stem}_meta.json")
    meta_tmp = meta_path.with_suffix(".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=1))
    meta_tmp.replace(meta_path)


def load_frames(path: Path = FRAMES_PATH, rule_id: str | None = None) -> list[ReplayFrame]:
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            f = ReplayFrame.model_validate_json(line)
            if rule_id is None or f.rule_id == rule_id:
                out.append(f)
    return out


def freeze_frame(rule_id: str, path: Path = FRAMES_PATH) -> ReplayFrame | None:
    """The frame at the rule's first alarm, with the last 12 months of strength attached."""
    fr = load_frames(path, rule_id)
    hit = next((f for f in fr if f.first_alarm is not None and f.as_of == f.first_alarm), None)
    if hit is None:
        return None
    hist = [f for f in fr if f.as_of <= hit.as_of][-12:]
    st = dict(hit.strength)
    st["last12"] = [{"as_of": str(f.as_of), "est": f.strength.get("est"), "lo": f.strength.get("lo"),
                     "hi": f.strength.get("hi")} for f in hist]
    return hit.model_copy(update={"strength": st})


def explainer_facts(frame: ReplayFrame, path: Path = FRAMES_PATH) -> dict:
    """Flat, display-rounded facts for M3's guarded explainer."""
    ce = pd.Timestamp(CALIB_WINDOW[1])
    t = pd.Timestamp(frame.as_of)
    vix = next((f for f in load_frames(path, "vix_control") if f.as_of == frame.as_of), None)

    def f2(x):
        return None if x is None else f"{x:.2f}"
    return {"rule": RULES[frame.rule_id]["statement"], "as_of": str(frame.as_of),
            "est": f2(frame.strength.get("est")), "lo": f2(frame.strength.get("lo")), "hi": f2(frame.strength.get("hi")),
            "cusum": f2(frame.change.get("cusum")), "h": f2(frame.change.get("threshold")),
            "alarm": frame.change.get("alarm"), "integrity": frame.integrity.get("status"),
            "months_since_calib_end": (t.year - ce.year) * 12 + t.month - ce.month,
            "control_rule_alarm": None if vix is None else vix.change.get("alarm")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", nargs="*", default=None)
    ap.add_argument("--start", default=REPLAY_START)
    ap.add_argument("--end", default=DATA_END)
    a = ap.parse_args(argv)
    t0 = time.time()
    replay(a.rules, a.start, a.end)
    meta = json.loads(META_PATH.read_text())
    print(f"replay done in {time.time() - t0:.1f}s -> {FRAMES_PATH}")
    for r, m in meta["rules"].items():
        extra = ""
        if "delay_vs_marker_m" in m:
            extra = f" | ex-post marker {m['expost_marker']} -> delay {m['delay_vs_marker_m']} m"
        print(f"  {r:16s} first alarm {m['first_alarm']} | historical {m.get('historical_alarms')}{extra}")


if __name__ == "__main__":
    main()
