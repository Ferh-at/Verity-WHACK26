The repository hosting the code for the WHACK 26 competition, following the Brevan Howard track.

Members: Ferhat Celik, Efe Ulker, Reyansh Mathur

## Setup (Python 3.11)

```
pip install uv
uv python install 3.11
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt   # Windows
.venv\Scripts\activate
python -m mocks.make_mocks      # rebuild the M2 synthetic cache
pytest -q                       # all tests (add -m "not slow" to skip the statistical size tests)
```

## Layout

`config.py`, `schemas.py`, `catalog.py` are the shared contract (BUILD_SPEC C3-C5). `engine/` (M1) is the
statistics engine, `data/` + `monitor/` (M2), `agents/` + `app/` (M3). See [DEV_HANDOFF.md](DEV_HANDOFF.md) for the combined application and data sources.

## Engine (M1)

- `python -m engine.pipeline --thesis "..." --n 20` runs thesis -> hypotheses -> verdicts and writes
  `runs/trials.jsonl`, `runs/tests.jsonl`, `runs/verdicts.jsonl`, `runs/hurdle.json`, `runs/prereg.jsonl`,
  `runs/vault.jsonl`.
- `python -m engine.benchmark --seeds 50 --out runs/benchmark.jsonl` runs the planted-trap benchmark
  (`--resume` continues a partial run; `--no-factor` disables the factor check). Writes
  `runs/benchmark.jsonl`, `runs/benchmark_naive.jsonl` and `runs/benchmark_metrics.json`.
- `python -m engine.benchmark --noise-search 200` runs the pure-noise search test (`runs/noise_search.json`).

Python entry point for M3: `engine.pipeline.run_batch(thesis, hyps, panel, objections_fn)`.
