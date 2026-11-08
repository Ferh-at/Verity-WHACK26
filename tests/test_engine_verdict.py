"""Verdict labels, vault errors, kill tests and a pipeline run."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from data.pit import panel as mock_panel
from engine.dsl import make_hypothesis
from engine.pipeline import Ledger, run_batch
from engine.vault import Vault, VaultError
from engine.verdict import classify, decide, outlier_test, regime_test
from schemas import Hypothesis, Power, Spec, TestResult, TrialLogEntry, Verdict

H = make_hypothesis(Spec(feature="mom", series="SPX", lookback=12, transform="sign", asset="SPX", direction=1),
                    "thesis", "trend")


def tr(test, passed, value=None, kill=False):
    return TestResult(hypothesis_id=H.id, test=test, passed=passed, kill=kill, value=value, detail="",
                      evidence_ref="")


def pw(mde):
    return Power(years=12, mde_sharpe_80=mde, mintrl_years=None, power_at_05=0.4)


GATES_OK = [tr("block_null", True, 0.01), tr("luck_hurdle", True), tr("by_fdr", True), tr("validation", True)]


@pytest.mark.parametrize("results,mde,holdout,label", [
    ([tr("leak_audit", False, kill=True)] + GATES_OK, 0.6, None, "Rejected"),
    ([tr("block_null", False, 0.2)], 0.6, None, "Rejected"),
    ([tr("block_null", False, 0.2)], 0.9, None, "Can't tell yet"),
    ([tr("block_null", True, 0.01), tr("luck_hurdle", False), tr("by_fdr", True), tr("validation", True)],
     0.6, None, "Fragile"),
    (GATES_OK, 0.6, None, "Promising"),
    (GATES_OK, 0.6, tr("holdout", False), "Promising"),
    (GATES_OK, 0.6, tr("holdout", True), "Replicated"),
    ([tr("sign", False, -0.001, kill=True)] + GATES_OK, 0.6, None, "Rejected"),
])
def test_verdict_table(results, mde, holdout, label):
    assert decide(H, results, pw(mde), holdout).label == label


def test_killed_by_first_failure():
    assert classify([tr("block_null", True, 0.01), tr("luck_hurdle", True), tr("by_fdr", False),
                     tr("validation", False)], pw(0.6), None) == ("Fragile", "by_fdr")


# --- kill tests
def _net(values):
    return pd.Series(values, index=pd.date_range("2003-01-31", periods=len(values), freq="ME"))


def test_outlier_kills_single_month_result():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.02, 144)
    x[50] = 0.6
    r = outlier_test(_net(x), "h", base_p=0.01)
    assert not r.passed and r.kill


def test_outlier_passes_robust_signal_and_skips_noise():
    rng = np.random.default_rng(4)
    assert outlier_test(_net(rng.normal(0.02, 0.02, 144)), "h").passed
    r = outlier_test(_net(rng.normal(0.0, 0.02, 144)), "h", base_p=0.4)
    assert r.passed and "not applicable" in r.detail


def test_regime_kills_one_period_signal():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.normal(0.03, 0.01, 48), rng.normal(-0.01, 0.02, 96)])
    assert not regime_test(_net(x), "h", base_p=0.01).passed


# --- vault
def _panel_hyp():
    p = mock_panel()
    return p, make_hypothesis(Spec(feature="level", series="T10Y3M", lookback=1, transform="sign", asset="SPX",
                                   direction=1), "thesis", "curve")


def test_vault_requires_prereg_and_is_one_shot():
    p, h = _panel_hyp()
    v = Vault(None)
    with pytest.raises(VaultError):
        v.open_holdout(h, p)
    v.register(h)
    res = v.open_holdout(h, p)
    assert res.test == "holdout"
    with pytest.raises(VaultError):
        v.open_holdout(h, p)


def test_vault_rejects_modified_spec():
    p, h = _panel_hyp()
    v = Vault(None)
    v.register(h)
    modified = h.model_copy(update={"spec": h.spec.model_copy(update={"direction": -1})})
    with pytest.raises(VaultError):
        v.open_holdout(modified, p)


def test_vault_files(tmp_path):
    p, h = _panel_hyp()
    v = Vault(tmp_path)
    v.register(h)
    v.open_holdout(h, p)
    assert (tmp_path / "prereg.jsonl").exists() and (tmp_path / "vault.jsonl").exists()
    with pytest.raises(VaultError):
        Vault(tmp_path).open_holdout(h, p)


# --- pipeline integration #1
def test_run_batch_writes_valid_jsonl(tmp_path):
    from app.fixtures import sample_hypotheses
    hyps = sample_hypotheses()[:5]
    vs = run_batch(None, hyps, mock_panel(), None, ledger=Ledger(), runs_dir=tmp_path, n_boot=200, n_hurdle=50)
    assert len(vs) == len(hyps)
    for line in (tmp_path / "verdicts.jsonl").read_text().splitlines():
        Verdict.model_validate_json(line)
    for line in (tmp_path / "trials.jsonl").read_text().splitlines():
        TrialLogEntry.model_validate_json(line)
    for line in (tmp_path / "tests.jsonl").read_text().splitlines():
        TestResult.model_validate(json.loads(line))
    hd = json.loads((tmp_path / "hurdle.json").read_text())
    assert len(hd["null_max"]) == 50 and set(hd["observed"]) <= {h.id for h in hyps}
    assert all(v.power.mde_sharpe_80 > 0 for v in vs)


def test_leaky_control_rejected_by_pipeline(tmp_path):
    p = mock_panel()
    h = make_hypothesis(Spec(feature="centered_ma", series="SPX", lookback=3, transform="sign", asset="SPX",
                             direction=1), "planted", "leaky demo")
    v = run_batch(None, [h], p, None, ledger=Ledger(), runs_dir=None, vault=Vault(None), n_boot=200, n_hurdle=50)[0]
    assert v.label == "Rejected"
    assert next(r for r in v.reasons if r.test == "leak_audit").passed is False


def test_hurdle_rises_with_ledger():
    p = mock_panel()
    led = Ledger()
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    specs = [Spec(feature="mom", series=s, lookback=L, transform="sign", asset=a, direction=d)
             for s in ("SPX", "NASDAQ", "GOLD") for L in (1, 3, 12) for a in ("SPX", "TLT") for d in (1,)]
    hyps = [make_hypothesis(s, "thesis", "", created_at=t0 + timedelta(seconds=i)) for i, s in enumerate(specs)]
    info1, info2 = {}, {}
    run_batch(None, hyps[:3], p, None, ledger=led, runs_dir=None, vault=Vault(None), n_boot=100, n_hurdle=100,
              audit=False, info=info1)
    run_batch(None, hyps[3:], p, None, ledger=led, runs_dir=None, vault=Vault(None), n_boot=100, n_hurdle=100,
              audit=False, info=info2)
    assert info2["hurdle"]["hurdle"] >= info1["hurdle"]["hurdle"]
    assert led.n_tried == len(hyps)
