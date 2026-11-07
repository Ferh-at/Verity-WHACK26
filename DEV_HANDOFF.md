# Developer handoff · WHACK 26

## Checkout and dependencies

This is the combined source project. It uses the M1 engine from the supplied WHACK26 archive, M2’s data and monitor, and M3’s Streamlit UI and agents. Install the pinned requirements from a fresh Python 3.11 or 3.12 environment. The virtual environment included in the original M1 archive is machine-specific.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

For Windows PowerShell, run `\.venv\Scripts\Activate.ps1` and `Copy-Item .env.example .env`. Set environment overrides with `$env:OFFLINE = "1"` for offline runs or `$env:OFFLINE = "0"` when fetching data and using live agents.

## Offline workflows

```bash
OFFLINE=1 python -m streamlit run app/Home.py
OFFLINE=1 python -m engine.pipeline --fixtures --n 20
OFFLINE=1 python -m pytest -q
```

Discover has two offline paths. **Run engine on fixtures** runs the real statistics engine with bundled hypotheses and synthetic data, writing under `runs/fixture_engine/`. **Replay cached run** opens the latest live batch when available or the bundled 20-card demonstration. The slides and recording in `demo/` use the bundled demonstration.

To regenerate M2’s monitor artefacts offline with an empty `data/cache/`, run:

```bash
OFFLINE=1 python -m monitor.calibrate --sims 1000
OFFLINE=1 python -m monitor.breaks --sims 200
OFFLINE=1 python -m monitor.replay
```

Refresh the app to use those outputs. Generated synthetic alarms can occur on different dates from the canned demo. The UI identifies the source for both.

## Market-data and live-agent setup

Set `LLM_API_KEY` and, if needed, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_MODEL_FALLBACK` and `FRED_API_KEY` in `.env`. Fetch data from a machine with vendor access.

```bash
OFFLINE=0 python -m data.fetch
python -m data.fetch --check
python -m pytest -q tests/test_m2_*.py
OFFLINE=1 python -m monitor.calibrate --sims 1000
OFFLINE=1 python -m monitor.breaks --sims 200
OFFLINE=1 python -m monitor.replay
OFFLINE=0 python -m streamlit run app/Home.py
```

Confirm `data/cache/` holds usable primary series. Restart the app after refreshing the cache so M2 reloads it. The optional `WHACK_CACHE_DIR` override takes precedence. Live Discover blocks the bundled synthetic fallback. For the live CLI, run `OFFLINE=0 python -m engine.pipeline --thesis "An inverted yield curve predicts equity drawdowns" --n 20`.

## Integration boundaries

| Component | Contract |
| --- | --- |
| M2 → M1 | `data.pit.decision_panel()` puts released values on their first usable month-end. The raw observation panel remains available to M2. |
| M1 → M3 | `engine.pipeline.run_batch(thesis, hyps, panel, objections_fn, info=...)` returns verdicts. The batch's trial rows and JSON-safe hurdle record come from `info`. |
| M1 → Proof | `runs/benchmark.jsonl` and `runs/benchmark_naive.jsonl` provide the two comparators. The packaged 50-seed snapshot lives under `mocks/m1_benchmark/` and is used until a new benchmark is generated. |
| M2 → Proof | `runs/calib.json` and `runs/benchmark_monitor.jsonl` provide threshold, stability and planted-break data. |
| M2 → Monitor | `runs/frames.jsonl` and `runs/frames_meta.json` provide time-truncated frames, onset, the locked settings and retrospective delay references. |

The supplied M1 engine uses a studentized stationary bootstrap. Its search ledger accumulates in the current process. The vault journal persists across restarts, and a reservation is written before evaluating holdout returns so an interruption still consumes that access. Fixture reservations are isolated from market runs. Duplicate cards identify their cluster representative.

The default monitor quantile is 0.632. To compare 0.368, pass `--quantile 0.368` to calibration and rerun breaks and replay. Early replay dates through the calibration end use detector parameters fitted over the full calibration window. The UI flags these dates as calibration-period context.

## Benchmark commands and evidence

```bash
python -m engine.benchmark --seeds 50 --out runs/benchmark.jsonl
python -m engine.benchmark --noise-search 200
```

M1’s benchmark enables factor attribution by default. Live discovery uses the setting in `config.py`. Add `--no-factor` to the benchmark command to disable it, or use `--resume` to continue an interrupted run with the same settings.

The packaged M1 result is a supplied, recorded synthetic run. Its trial ledgers contain 10,000 candidates over 50 seeds. Recomputing the aggregates from those ledgers gives 0.33% mean FDR versus 78.13% naive and 97% recall at planted annual Sharpe 2.0. It contains zero “Can’t tell yet” outcomes. The separate offline fixture engine path exercises that label.

The supplied 200-search pure-noise result reports zero searches with a trusted outcome. The live provider call and the real-market VIX acceptance test require external credentials and market data.
