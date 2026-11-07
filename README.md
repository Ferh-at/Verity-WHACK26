# WHACK 26 · Discovery Lab

A quantitative research workbench for the Brevan Howard track. The project combines Ferhat Celik’s M1 statistics engine, Efe Ulker’s M2 data and monitor, and Reyansh Mathur’s M3 agents and Streamlit app.

## Quick start

Run from this directory with Python 3.11 or 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
OFFLINE=1 python -m streamlit run app/Home.py
```

On Windows PowerShell, activate with `\.venv\Scripts\Activate.ps1`, then run `$env:OFFLINE = "1"` before launching Streamlit.

In **Discover**, choose **Run engine on fixtures** to compute 20 verdicts offline or **Replay cached run** to step through the demonstration. **Proof** opens with the supplied 50-seed planted-trap benchmark. **Monitor** opens with a synthetic rehearsal and can load a fresh replay from the M2 jobs.

[DEV_HANDOFF.md](DEV_HANDOFF.md) covers live setup, the CLI, output formats, benchmarks, and the verification limits.
