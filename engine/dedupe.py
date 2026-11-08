"""Duplicate clustering and effective N: single linkage in creation order on the
correlation of discovery net returns; the first-created member represents the cluster."""
import numpy as np
import pandas as pd

from config import DUP_CORR
from schemas import Hypothesis

MIN_OVERLAP = 24


def cluster_returns(order: list[str], nets: dict[str, pd.Series], reps: list[str] | None = None,
                    threshold: float = DUP_CORR) -> dict[str, str]:
    """order: hypothesis ids in creation order; nets: discovery net returns per id.
    reps: existing representatives (e.g. earlier batches of the session), checked first.
    Returns {hid: rep_hid} (reps map to themselves)."""
    reps = [r for r in (reps or []) if r in nets]
    out = {r: r for r in reps}
    ids = list(dict.fromkeys(reps + [h for h in order if h in nets]))
    if not ids:
        return {**out, **{h: h for h in order}}
    M = pd.concat([nets[h].rename(h) for h in ids], axis=1).to_numpy(float)
    col = {h: j for j, h in enumerate(ids)}
    rep_cols: list[int] = [col[r] for r in reps]
    for hid in order:
        if hid in out:
            continue
        best, best_c = None, -1.0
        if hid in col and rep_cols:
            x = M[:, col[hid]][:, None]
            R = M[:, rep_cols]
            mask = ~np.isnan(x) & ~np.isnan(R)
            n = mask.sum(axis=0)
            xm = np.where(mask, x, 0.0)
            rm = np.where(mask, R, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, mr = xm.sum(0) / n, rm.sum(0) / n
                dx, dr = np.where(mask, x - mx, 0.0), np.where(mask, R - mr, 0.0)
                c = (dx * dr).sum(0) / np.sqrt((dx ** 2).sum(0) * (dr ** 2).sum(0))
            c = np.where((n >= MIN_OVERLAP) & np.isfinite(c), c, -np.inf)
            j = int(np.argmax(c))
            if np.isfinite(c[j]):
                best, best_c = ids[rep_cols[j]], float(c[j])
        if best is not None and best_c > threshold:
            out[hid] = best
        else:
            out[hid] = hid
            if hid in col:
                rep_cols.append(col[hid])
    return out


def cluster(hyps: list[Hypothesis], panel: pd.DataFrame, splits: dict | None = None) -> dict[str, str]:
    from engine.backtest import eligible_net, strategy_returns
    hs = sorted(hyps, key=lambda h: h.created_at)
    nets = {h.id: eligible_net(strategy_returns(h.spec, panel, "discovery", splits)) for h in hs}
    return cluster_returns([h.id for h in hs], nets)


def effective_n(mapping: dict[str, str]) -> int:
    return len(set(mapping.values()))
