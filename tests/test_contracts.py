import json
from pathlib import Path

import pytest

from schemas import BenchmarkRun, Hypothesis, ReplayFrame, TrialLogEntry, Verdict

MOCKS = Path(__file__).resolve().parents[1] / "mocks"


@pytest.mark.parametrize(("name", "model"), [
    ("hypotheses", Hypothesis), ("trials", TrialLogEntry), ("verdicts", Verdict),
    ("frames", ReplayFrame), ("benchmark", BenchmarkRun),
])
def test_demo_rows_parse(name, model):
    lines = (MOCKS / f"sample_{name}.jsonl").read_text().splitlines()
    assert lines
    for line in lines:
        model.model_validate(json.loads(line))
