"""NBER recession dates, point in time.

Convention (same as FRED USREC): recession months are the months AFTER the peak up to and
including the trough. A recession is known only once its peak has been announced; if the trough
has not yet been announced at `as_of`, the recession runs to `as_of`.

Never use FRED USREC in the point-in-time replay (it is revised with hindsight).
"""
from __future__ import annotations

import pandas as pd

from data.store import month_end, as_ts

# (event, month, announcement date)  [verify against nber.org/research/business-cycle-dating]
NBER_TABLE = [
    ("peak",   "2001-03", "2001-11-26"),
    ("trough", "2001-11", "2003-07-17"),
    ("peak",   "2007-12", "2008-12-01"),
    ("trough", "2009-06", "2010-09-20"),
    ("peak",   "2020-02", "2020-06-08"),
    ("trough", "2020-04", "2021-07-19"),
]
HISTORY_START = "2000-01-31"


def _cycles():
    """[(peak_month_end, peak_announced, trough_month_end|None, trough_announced|None)]"""
    out, cur = [], None
    for ev, m, ann in NBER_TABLE:
        if ev == "peak":
            cur = [month_end(m + "-01"), pd.Timestamp(ann), None, None]
            out.append(cur)
        else:
            cur[2], cur[3] = month_end(m + "-01"), pd.Timestamp(ann)
    return out


def recession_known(as_of, start: str = HISTORY_START) -> pd.Series:
    """Monthly bool Series (index start..month-end of as_of): recession months known at as_of."""
    t = as_ts(as_of)
    idx = pd.date_range(start, month_end(t), freq="ME")
    s = pd.Series(False, index=idx)
    for peak, peak_ann, trough, trough_ann in _cycles():
        if peak_ann > t:
            continue
        end = trough if (trough is not None and trough_ann <= t) else month_end(t)
        s[(s.index > peak) & (s.index <= end)] = True
    return s


def recession_starts_known(as_of) -> list[pd.Timestamp]:
    """First recession month (peak + 1) of every recession whose peak was announced by as_of."""
    t = as_ts(as_of)
    return [peak + pd.offsets.MonthEnd(1) for peak, ann, _, _ in _cycles() if ann <= t]
