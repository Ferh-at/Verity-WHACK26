# M2 — data, point-in-time loader, leak audit, vendor check, monitor

## First thing on a laptop with internet
```bash
pip install -r requirements.txt
cp .env.example .env                 # add FRED_API_KEY (optional: keyless FRED CSV fallback is built in)
OFFLINE=0 python -m data.fetch       # downloads everything to data/cache/ (+ CSV mirror)
python -m data.fetch --check         # every id should show n > 200 (BTC > 120)
pytest -q tests/test_m2_*.py
python -m monitor.calibrate --sims 1000          # -> runs/calib.json
python -m monitor.breaks --sims 200              # -> runs/breaks.json, runs/benchmark_monitor.jsonl
python -m monitor.replay                         # -> runs/frames.jsonl, runs/frames_meta.json (< 10 s)
```
Until `data/cache/` has files, every loader silently falls back to **SYNTHETIC** `mocks/cache/`
(a warning is logged; `data.store.using_mock()` tells you). Mock numbers are made up — never
put them on a slide. Rebuild mocks with `python -m mocks.make_mocks`. In the combined build,
M2 writes `m2_panel.parquet` and `sample_m2_frames.jsonl` to keep M3’s demo fixtures separate.
Generated calibration, break and replay artefacts record the input source. See [DEV_HANDOFF.md](DEV_HANDOFF.md).

Overnight (K4): `python -m monitor.calibrate --sims 10000` — chunks of 500 are checkpointed in
`runs/calib_ckpt/`; rerunning the same command resumes and gives identical numbers.

## Interfaces for M1 / M3
| Function | Notes |
|---|---|
| `data.pit.panel(t=None, ids=None)` | wide month-end panel + `RF`; cells with `published > t` are NaN, rows after t dropped |
| `data.pit.as_of(id, t)` | values published by t |
| `data.audit.leak_audit(feature_fn, panel, dates=None, seed=SEED, hypothesis_id=...)` | extra kwarg `hypothesis_id` (TestResult needs it). Evidence → `runs/audit_evidence.jsonl`, `evidence_ref` points there. Helper: `leak_audit_spec(spec, panel, hypothesis_id)` |
| `data.audit.edit_future_demo(spec, panel, t, multiplier)` | `{before, after, position_before, position_after, passed, detail, t, edited_range}` |
| `data.vendor.vendor_test(h, panel)` / `vendor_check(asset, end, window)` | Stooq missing → `single_vendor`, passes |
| `monitor.replay.replay(rule_ids, start, end)` | writes `runs/frames.jsonl` (+ `frames_meta.json`: locked params, historical alarms, delays) |
| `monitor.replay.freeze_frame(rule_id)` | frame at first alarm, `strength["last12"]` attached; `explainer_facts(frame)` gives M3's explainer dict |
| `runs/calib.json` | per rule: `mu0, sd0, h, fa_rate_per_20y` (= `false_alarm_rate_per_20y`), CI, `max_c` array for the histogram |
| `runs/breaks.json`, `runs/benchmark_monitor.jsonl` | break/stable `BenchmarkRun` rows kept separate from M1's `runs/benchmark.jsonl` — the Proof page should read both |

`data/features_ref.py` is a copy of the M1.2 feature table used only when `engine.features` is
not importable; once M1's module exists, the audit and vendor test use M1's code automatically.

## Decisions / spec issues to agree as a team
1. **Calibration quantile contradicts its own rationale.** `CALIB_QUANTILE = 0.632` gives
   P(any false alarm in 20 y) = 0.368, i.e. ~0.45 alarms per 20 y — meets the must-have (≤ 1) but
   misses the M2.11 band [0.5, 1.5] and slows detection. The rationale (P(≥1 alarm) = 1 − e⁻¹)
   needs quantile **0.368**. On mock data:

   | quantile | bonds h | false alarms / 20 y | median break delay | detected ≤ 24 m |
   |---|---|---|---|---|
   | 0.632 (config) | 52 | 0.43 | 23 m | 58 % |
   | 0.368 | 30 | 0.80 | 17 m | 90 % |

   Try it with `python -m monitor.calibrate --quantile 0.368`; config.py is unchanged until agreed.
2. **Strength is a 36-month rolling statistic**, so consecutive z values are highly correlated and
   h comes out large (≈ 30–60, not ≈ 5). That is correct for this design, but explain it in Q&A.
3. **Locked parameters use 2003–2019 data**, so replay frames for 2015–2019 use μ0/σ0/h fitted on
   data partly after their date. Frames from 2020 on are fully point-in-time. Say so if asked.
4. **Monthly macro lag:** with `published = next month-end`, at month-end t the value dated t−1 is
   known and the value dated t is not (the spec text says t−1 and t are missing; the rule wins).
5. **first_alarm = first alarm *onset* on/after 2015-01** (C crossing above h). Earlier episodes are
   in `frames_meta.json → historical_alarms`.
6. **Delay for the real 2022 alarm** is reported two ways in `frames_meta.json`: vs Jan 2022, and vs
   an ex-post marker (first month from 2015 starting ≥ 6 months of negative 12-month strength).
   Both are hindsight references used only for reporting, never by the alarm.
7. `ReplayFrame.change` for `curve_recession` is a dict with `None` values plus `display_only: true`
   (the schema type is `dict`, so `None` would not validate).
8. Leak audit tolerance is relative (`1e-9 × max(1, |value|)`) so index levels in the thousands do
   not trip on float rounding. A value that exists only when the future is visible (e.g. the last
   point of a centred window) also counts as a leak.
