# WHACK 26 · Verity

Verity is a quantitative research workbench for the Brevan Howard track at WHACK 26. Give it a market thesis. A language model turns that thesis into precise, testable hypotheses. A deterministic statistics engine then tries to knock them down. The model can challenge, refine or nominate an idea. Only the statistics can let one through.

Built by Ferhat Celik, Efe Ulker and Reyansh Mathur.

## What a research run does

A run is fixed at three rounds before it starts. Each round has the same shape.

1. **Propose.** The generator writes hypotheses in a small domain language: a feature (`mom`, `level`, `change`, `zscore` or `corr`), one or two catalog series, a lookback, a transform, a tradable asset and a direction. Leaky transforms (`centered_ma`, `fullsample_z`) exist only as audit controls and are never offered to the model.
2. **Explore on discovery data only (2003–2014).** Each new idea is checked for look-ahead, the sign of its net return, a stationary-bootstrap null and a few extreme months. Near-duplicate strategies are clustered, and the luck hurdle plus Benjamini–Yekutieli adjustment are recomputed over every idea tried so far, including earlier rounds. That discovery summary is the only evidence the reviewer sees.
3. **Review.** The reviewer may challenge an idea with a kill-only test (regime, costs, factor or vendor), spend a limited budget on refined hypotheses, or nominate at most three ideas for the sealed holdout. It never sees validation or holdout results. Prompts are checked in code so they cannot carry a date from 2020 onward or a number outside the typed evidence fields. Every call is logged to `runs/llm_calls.jsonl`.
4. **Confirm once.** After the reviewer finishes, the engine runs validation (2015–2019), the optional factor check and a one-shot holdout (2020–2026) for nominated ideas. A holdout look is written to the vault before the return is evaluated, so an interrupted run still consumes that look.

Verdicts are `Rejected`, `Can't tell yet`, `Fragile`, `Promising` or `Replicated`. Every card includes a power line: the annual Sharpe the discovery sample could have detected. `Can't tell yet` means the result is not significant and the sample is too short to resolve a Sharpe of 0.75. Assets with fewer than 60 discovery months, such as Bitcoin, use a 60/20/20 split of their own history instead of the fixed calendar.

The position at month-end `t` earns the next month's return, after a cost of 2 basis points per unit of one-way turnover. The risk-free rate is the 3-month Treasury yield, lagged one month.

## The app

```bash
python -m streamlit run app/Home.py
```

| Page | What it shows |
| --- | --- |
| Home | The research loop, the five checks behind a verdict, and the packaged benchmark at a glance |
| Research | A thesis, the round-by-round ledger, the luck hurdle, verdict cards and an "edit the future" control |
| Proof | A planted-trap benchmark: real signals of known strength against leaks, outliers, regimes, disguised beta, duplicates and noise, next to a naive t-test |
| Data and methods | Series coverage, the three time periods, the configured model endpoint and the published methods |

The sidebar reads **LIVE** after a research run in this session, **CACHED** when a page is serving saved artefacts, and **OFFLINE** when `OFFLINE=1`.

On Research, **Run AI research loop** calls the model and the engine on the market cache. **Replay last run** opens the latest saved batch, or the bundled demonstration when no batch has been saved. Offline mode disables the live button and keeps the replay.

## Repository layout

| Path | Role |
| --- | --- |
| `catalog.py` | Series definitions: vendor symbols, release convention, whether a series can be traded |
| `config.py` | Splits, costs, bootstrap settings, monitor calibration and model temperatures |
| `schemas.py` | Shared models for hypotheses, trials, tests, verdicts and replay frames |
| `data/` | Fetch, point-in-time panel, leak audit, Yahoo-versus-Stooq check, NBER dates |
| `engine/` | Features, backtest, nulls, luck hurdle, multiple testing, verdicts, holdout vault, benchmark |
| `monitor/` | Rule strength, CUSUM alarms, calibration, planted breaks and month-by-month replay |
| `agents/` | Model client, generator, reviewer, skeptic, explainer and the number guard |
| `app/` | Streamlit pages |
| `mocks/` | Synthetic cache, bundled hypotheses and the recorded benchmark snapshot |
| `tests/` | Unit and integration tests |
| `demo/runbook.md` | Timed walkthrough for the demonstration |
| `runs/` | Local outputs. Not committed |

`data/cache/` holds the market files and is created by the fetch command. Until those files exist, loaders fall back to `mocks/cache/` and log that the numbers are synthetic. A live Research run does not use that fallback.

## Setup

Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

A shell variable overrides `.env`. Leave `$env:OFFLINE` unset when the file should apply.

| Variable | Role |
| --- | --- |
| `LLM_API_KEY` | Model calls for hypothesis generation, review, objections and explanations |
| `LLM_BASE_URL` | OpenAI-compatible endpoint. Leave blank to use the provider default |
| `LLM_MODEL` | Primary model. The example file uses `gpt-4o-mini` |
| `LLM_MODEL_FALLBACK` | Tried once if the primary model fails validation. The example uses `gpt-4o` |
| `FRED_API_KEY` | FRED series. A keyless CSV fallback is used when this is blank |
| `OFFLINE` | `0` allows the network. `1` uses caches only |
| `USE_CACHE` | `1` reuses a saved response when the prompt is identical. A new prompt still calls the API |

Fetch the market cache, check coverage, then start the app:

```bash
python -m data.fetch
python -m data.fetch --check
python -m streamlit run app/Home.py
```

`python -m data.fetch --check` should show more than 200 months for each primary series, and more than 120 for Bitcoin. Restart Streamlit after a refresh so every page reads the new files. `WHACK_CACHE_DIR` overrides the cache location when set.

## Command line

Run a thesis through the same engine the app uses:

```bash
python -m engine.pipeline --thesis "An inverted yield curve predicts equity drawdowns" --n 20
```

`--n` is the number of hypotheses in the batch. The command writes `runs/trials.jsonl`, `runs/tests.jsonl`, `runs/verdicts.jsonl`, `runs/hurdle.json`, `runs/prereg.jsonl` and `runs/vault.jsonl`.

An offline pass through the bundled hypotheses, with no model call and no market download:

```bash
python -m engine.pipeline --fixtures --n 20
```

That writes under `runs/fixture_engine/` so the synthetic holdout reservations stay separate from market runs.

### Benchmark

The Proof page opens on a recorded 50-seed snapshot in `mocks/recorded_benchmark/` until a new benchmark is generated. Regenerate it with:

```bash
python -m engine.benchmark --seeds 50 --out runs/benchmark.jsonl
python -m engine.benchmark --noise-search 200
```

The first command also writes `runs/benchmark_naive.jsonl` and `runs/benchmark_metrics.json`. `--resume` continues an interrupted run. `--no-factor` turns off factor attribution for that benchmark. Live research uses `FACTOR_ENABLED` in `config.py`, which is off unless you change it. The factor test becomes a kill test only when that flag is on.

Each seed plants 200 synthetic hypotheses: 10 real signals at known Sharpe ratios and 190 fakes (noise, look-ahead, a single extreme month, a signal that dies, disguised market beta and near-duplicates). The naive comparator is a one-sided t-test at 5 percent and nothing else.

### Monitor

The monitor replays six rules month by month using only data that had been released by that month: bonds protect stocks, the dollar as a haven, gold versus real yields, Bitcoin versus the Nasdaq, curve inversion and recession, and a VIX control that should not alarm. Thresholds are locked on 2003–2019 data and on stable simulations. The default quantile is 0.632, which targets well under one false alarm per 20 years. Replay months inside the calibration window use parameters fitted on that whole window; later months are fully point in time.

```bash
python -m monitor.calibrate --sims 1000
python -m monitor.breaks --sims 200
python -m monitor.replay
```

Those commands write `runs/calib.json`, `runs/breaks.json`, `runs/benchmark_monitor.jsonl`, `runs/frames.jsonl` and `runs/frames_meta.json`. A 10,000-simulation calibration checkpoints every 500 draws under `runs/calib_ckpt/` and resumes if you repeat the command. Pass `--quantile 0.368` to compare a lower threshold; `config.py` is unchanged by that flag.

Rebuild the synthetic cache, the wide mock panel and the sample replay with:

```bash
python -m mocks.make_mocks
```

## Tests

```bash
python -m pytest -q
```

Add `-m "not slow"` to skip the longer statistical size checks. The suite covers the hypothesis language, the backtest, null size, the leak audit, verdicts, the data loader, monitor calibration and the Streamlit pages.

## Data

Prices come from Yahoo Finance, with Stooq as the second vendor for the integrity check. Yields, unemployment, CPI and the funds rate come from FRED. Recession dates are the NBER announcement table in `data/nber.py`, not the revised FRED USREC series. Monthly macro series are treated as known at the following month-end. The catalog and the coverage table on Data and methods list every series.

Discovery is January 2003 through December 2014, validation is 2015–2019, and the holdout is January 2020 through October 2026. One month is dropped at each boundary.

## Offline

Set `OFFLINE=1` in `.env` and restart.

```bash
python -m streamlit run app/Home.py
python -m engine.pipeline --fixtures --n 20
python -m pytest -q
```

Monitor artefacts can be rebuilt from the synthetic cache the same way, with `OFFLINE=1` in the environment. Generated synthetic alarms need not fall on the same months as the bundled demonstration. The pages label which source a figure came from.
