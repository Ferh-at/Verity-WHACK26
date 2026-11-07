"""Multiple testing across the ledger (M1.7): Benjamini-Yekutieli q-values, Holm for display."""
from statsmodels.stats.multitest import multipletests

from config import FDR_Q


def _adjust(p: dict, method: str) -> dict:
    if not p:
        return {}
    keys = list(p)
    q = multipletests([p[k] for k in keys], alpha=FDR_Q, method=method)[1]
    return {k: float(v) for k, v in zip(keys, q)}


def by_qvalues(p: dict) -> dict:
    """BY (2001): reject p_(i) <= i*q / (m*c(m)), c(m) = sum_{j<=m} 1/j. Input: p_block of cluster reps."""
    return _adjust(p, "fdr_by")


def bh_qvalues(p: dict) -> dict:
    return _adjust(p, "fdr_bh")


def holm_qvalues(p: dict) -> dict:
    return _adjust(p, "holm")
