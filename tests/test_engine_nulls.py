"""Null size and luck-hurdle known-answer tests (M1.5, M1.6; must-haves 4-5)."""
import numpy as np
import pytest
from scipy.signal import lfilter

from engine.hurdle import hurdle_from, null_stats, shift_draws
from engine.nulls import circular_shift, p_block
from engine.stats import naive_p, nw_t


@pytest.mark.slow
def test_block_null_size_white_noise():
    rej = np.mean([p_block(np.random.default_rng(s).normal(0, 1, 144)) < 0.05 for s in range(500)])
    assert 0.03 <= rej <= 0.07, rej


@pytest.mark.slow
def test_block_null_size_ar1():
    xs = [lfilter([1], [1, -0.5], np.random.default_rng(1000 + s).normal(size=144)) for s in range(500)]
    rej = np.mean([p_block(x) < 0.05 for x in xs])
    naive = np.mean([naive_p(x) < 0.05 for x in xs])
    print(f"AR(1) phi=0.5: block-bootstrap rejection {rej:.3f}, naive iid t-test {naive:.3f}")
    assert rej <= 0.08 and naive > rej


def test_p_block_detects_strong_signal():
    assert p_block(np.random.default_rng(0).normal(0.5, 1, 144)) < 0.01


def test_circular_shift():
    import pandas as pd
    s = pd.Series([1, 2, 3, 4])
    assert list(circular_shift(s, 1)) == [4, 1, 2, 3]


@pytest.mark.slow
def test_luck_hurdle_size_pure_noise():
    """200 searches of 100 pure-noise candidates (T=180): share with max S_i > hurdle in [0.02, 0.08];
    the hurdle for 100 candidates lies in [2.0, 3.5]."""
    T, K, n = 180, 100, 200
    hits, hurdles = [], []
    for s in range(200):
        rng = np.random.default_rng(10_000 + s)
        eta = rng.standard_normal((K, T))
        pos = lfilter([np.sqrt(0.75)], [1, -0.5], eta, axis=1)
        pos = np.clip(pos, -2, 2) / 2
        r = rng.normal(0.005, 0.04, (K, T))
        u = shift_draws(n, 42 + 1000 * s)
        stats = {i: null_stats(pos[i], r[i], u) for i in range(K)}
        obs = {}
        for i in range(K):
            turn = np.abs(np.diff(pos[i], prepend=0.0))
            obs[i] = nw_t(pos[i] * r[i] - 2e-4 * turn)
        hd = hurdle_from(stats, obs)
        hurdles.append(hd["hurdle"])
        hits.append(max(obs.values()) > hd["hurdle"])
    share = float(np.mean(hits))
    assert 0.02 <= share <= 0.08, share
    assert 2.0 <= float(np.median(hurdles)) <= 3.5
