"""Pre-registration and the one-shot holdout vault, including the hindsight condition."""
import json
import os
from threading import RLock
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import ALPHA, HINDSIGHT_MIN_N, N_BOOT, SEED
from engine.backtest import eligible_net, strategy_returns
from engine.dsl import spec_hash
from engine.nulls import p_block
from engine.locks import file_lock
from engine.stats import sharpe
from schemas import Hypothesis, TestResult

PLAN = "block_null+hurdle+by+validation+holdout"
RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


class VaultError(Exception):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def hindsight_test(h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None) -> TestResult:
    """Post-cutoff (months after LLM_CUTOFF) reversal check. Never kills on its own: it is a
    condition inside the holdout. Untestable with fewer than HINDSIGHT_MIN_N months."""
    def res(passed, value, detail):
        return TestResult(hypothesis_id=h.id, test="hindsight", passed=passed, kill=False, value=value,
                          detail=detail, evidence_ref="runs/vault.jsonl")
    if splits is not None and "post_cutoff" not in splits:
        return res(True, None, "untestable (no post-cutoff period)")
    net = eligible_net(strategy_returns(h.spec, panel, "post_cutoff", splits))
    n = len(net)
    s = sharpe(net) if n >= 2 else None
    s_txt = "n/a" if s is None or pd.isna(s) else f"{s:.2f}"
    if n < HINDSIGHT_MIN_N:
        return res(True, s, f"untestable (n<{HINDSIGHT_MIN_N}); post-cutoff net Sharpe {s_txt}, n={n}")
    if s < 0:
        return res(False, s, f"post-cutoff reversal: net Sharpe {s_txt}, n={n}")
    return res(True, s, f"post-cutoff net Sharpe {s_txt}, n={n}")


class Vault:
    """runs_dir=None keeps the prereg/vault ledgers in memory (benchmark); otherwise they are the
    append-only files runs/prereg.jsonl and runs/vault.jsonl."""

    def __init__(self, runs_dir: str | Path | None = RUNS_DIR):
        self.dir = None if runs_dir is None else Path(runs_dir)
        self._prereg: list[dict] = []
        self._vault: list[dict] = []
        self._reserved: list[dict] = []
        self._thread_lock = RLock()

    def _lock(self):
        return self._thread_lock if self.dir is None else file_lock(self.dir / ".vault.lock")

    def _memory(self, name):
        return {"prereg": self._prereg, "vault": self._vault, "vault_reservations": self._reserved}[name]

    def _read(self, name: str) -> list[dict]:
        if self.dir is None:
            return self._memory(name)
        f = self.dir / f"{name}.jsonl"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _append(self, name: str, row: dict) -> None:
        if self.dir is None:
            self._memory(name).append(row)
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{name}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def register(self, h: Hypothesis) -> dict:
        row = {"hypothesis_id": h.id, "spec_hash": spec_hash(h.spec), "plan": PLAN, "ts": _now().isoformat()}
        with self._lock():
            self._append("prereg", row)
        return row

    def is_opened(self, h: Hypothesis) -> bool:
        digest = spec_hash(h.spec)
        return any(row["spec_hash"] == digest
                   for name in ("vault_reservations", "vault") for row in self._read(name))

    def open_holdout_full(self, h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None,
                          n_boot: int = N_BOOT, seed: int = SEED) -> tuple[TestResult, TestResult]:
        """Returns (holdout result, hindsight result). Raises VaultError without a matching earlier
        prereg entry, or if this spec_hash was already opened."""
        with self._lock():
            now = _now()
            hsh = spec_hash(h.spec)
            # The row must already be on the ledger. <= covers a registration written
            # in the same clock tick, which Windows can report as equal to now.
            if not any(r["hypothesis_id"] == h.id and r["spec_hash"] == hsh
                       and datetime.fromisoformat(r["ts"]) <= now for r in self._read("prereg")):
                raise VaultError(f"{h.id}: no pre-registration with spec_hash {hsh[:10]}")
            if self.is_opened(h):
                raise VaultError(f"{h.id}: holdout already opened for spec_hash {hsh[:10]}")
            self._append("vault_reservations", {"hypothesis_id": h.id, "spec_hash": hsh, "opened_at": now.isoformat()})
            net = eligible_net(strategy_returns(h.spec, panel, "holdout", splits))
            n = len(net)
            s = sharpe(net) if n >= 2 else float("nan")
            p = p_block(net, n=n_boot, seed=seed) if n >= 8 else 1.0
            hind = hindsight_test(h, panel, splits)
            ok_stats = n >= 8 and net.mean() > 0 and p < ALPHA and s > 0
            passed = bool(ok_stats and hind.passed)
            why = "" if passed else (" (post-cutoff reversal)" if ok_stats else "")
            self._append("vault", {"hypothesis_id": h.id, "spec_hash": hsh, "opened_at": now.isoformat(),
                                   "passed": passed})
            tr = TestResult(hypothesis_id=h.id, test="holdout", passed=passed, kill=False,
                            value=None if pd.isna(s) else float(s),
                            detail=f"one-shot holdout: net Sharpe {s:.2f}, one-sided p={p:.3f}, n={n}{why}",
                            evidence_ref="runs/vault.jsonl")
            return tr, hind

    def open_holdout(self, h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None,
                     n_boot: int = N_BOOT, seed: int = SEED) -> TestResult:
        return self.open_holdout_full(h, panel, splits, n_boot, seed)[0]


_default = Vault()


def register(h: Hypothesis) -> dict:
    return _default.register(h)


def open_holdout(h: Hypothesis, panel: pd.DataFrame, splits: dict | None = None) -> TestResult:
    return _default.open_holdout(h, panel, splits)
