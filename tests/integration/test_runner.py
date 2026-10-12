import asyncio
from pathlib import Path

import pytest

from agentreclab.core.config import ROOT, load_run
from agentreclab.core.runner import run_experiment
from agentreclab.data.fixtures import make_fixture
from agentreclab.eval.compare import compare_runs
from agentreclab.state.snapshot import read_json, write_json


def config(tmp_path, model, n=10):
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml", model=model)
    cfg.update(
        manifest=str(make_fixture(tmp_path / "data", n)),
        outputs=str(tmp_path / "runs"),
        run_id=model,
    )
    return cfg


@pytest.mark.parametrize("model,expected_calls", [("iagent", 20), ("i2agent", 50), ("memrec", 50)])
def test_upstream_end_to_end_and_resume(tmp_path, model, expected_calls):
    cfg = config(tmp_path, model)
    result = asyncio.run(run_experiment(cfg))
    assert result["status"] == "completed", result
    assert result["metrics"]["n_evaluated"] == 10
    assert result["metrics"]["total_cost_cny"] == 0
    run = Path(result["run"])
    before = (run / "traces.jsonl").read_text(encoding="utf-8")
    again = asyncio.run(run_experiment(cfg, resume=True))
    assert again["metrics"] == result["metrics"]
    assert (run / "traces.jsonl").read_text(encoding="utf-8") == before
    assert len((run / "predictions.jsonl").read_text().splitlines()) == 10
    manifest = read_json(run / "manifest.json")
    assert len(manifest["cases"]) == 10
    assert manifest["reproduction_status"] == "synthetic_mock"
    # Identical prompts can share a response; call upper bound, no forced duplicate calls.
    assert result["metrics"]["total_calls"] <= expected_calls
    with pytest.raises(ValueError, match="fingerprint"):
        asyncio.run(run_experiment({**cfg, "seed": 1}, resume=True))


def test_paired_compare_and_reject(tmp_path):
    runs = [
        asyncio.run(run_experiment(config(tmp_path, model)))["run"]
        for model in ("iagent", "i2agent", "memrec")
    ]
    comparison = compare_runs(runs)
    assert comparison["n_paired"] == 10
    path = Path(runs[1]) / "manifest.json"
    manifest = read_json(path)
    manifest["cases"][0]["candidates"].reverse()
    write_json(path, manifest)
    with pytest.raises(ValueError, match="paired"):
        compare_runs(runs)


def test_gates_and_dry_run(tmp_path):
    cfg = config(tmp_path, "iagent")
    assert asyncio.run(run_experiment(cfg, dry_run=True))["planned_calls_without_retries"] == 20
    cfg["provider"]["name"] = "openai"
    with pytest.raises(ValueError, match="allow-paid"):
        asyncio.run(run_experiment(cfg))
    cfg["protocol_mode"] = "paper_reproduction"
    with pytest.raises(ValueError, match="not_ready"):
        asyncio.run(run_experiment(cfg))
    cfg["model"] = "mars"
    with pytest.raises(NotImplementedError, match="not_ready"):
        asyncio.run(run_experiment(cfg))


def test_budget_resume_replays_completed_requests(tmp_path):
    cfg = config(tmp_path, "iagent")
    cfg["max_calls"] = 1
    result = asyncio.run(run_experiment(cfg))
    assert result["status"] == "completed_with_failures"
    assert result["metrics"]["n_failed"] == 10
    assert result["metrics"]["total_calls"] == 1


def test_serial_parallel_same_predictions(tmp_path):
    cfg = config(tmp_path, "iagent")
    first = asyncio.run(run_experiment(cfg))
    second = asyncio.run(run_experiment({**cfg, "run_id": "parallel", "concurrency": 4}))
    import json

    def predictions(run):
        return [
            json.loads(line)["prediction"]["ordered_item_ids"]
            for line in (Path(run) / "predictions.jsonl").read_text().splitlines()
        ]

    assert predictions(first["run"]) == predictions(second["run"])
