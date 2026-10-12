import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agentreclab.cli import app
from agentreclab.core.config import ROOT, load_run
from agentreclab.core.runner import run_experiment
from agentreclab.data.fixtures import make_fixture
from agentreclab.providers.mock import MockProvider
from agentreclab.state.snapshot import digest
from agentreclab.state.store import StateStore
from agentreclab.tracing.cache import TracedProvider
from agentreclab.tracing.recorder import Recorder


def test_interrupted_case_uses_durable_response_replay(tmp_path):
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml")
    cfg.update(manifest=str(make_fixture(tmp_path / "data")), outputs=str(tmp_path / "runs"))
    first = asyncio.run(run_experiment(cfg))
    directory = Path(first["run"])
    store = StateStore(directory / "state.sqlite")
    completed = store.rows()
    store.put_case(completed[0]["case_id"], "running")  # Crash after responses, before case commit.
    store.close()
    second = asyncio.run(run_experiment(cfg, resume=True))
    assert second["metrics"]["total_calls"] == first["metrics"]["total_calls"]
    assert second["metrics"]["cache_hits"] == 2
    assert second["metrics"]["exclude"] == first["metrics"]["exclude"]
    assert len((directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()) == 10


def test_unknown_dispatch_outcome_blocks_paid_repeat(tmp_path):
    cfg = {"name": "mock", "model": "v1"}
    store = StateStore(tmp_path / "state.sqlite")
    recorder = Recorder(tmp_path)
    provider = TracedProvider(MockProvider(), cfg, store, recorder)
    key = digest({"provider": cfg, "messages": [], "parameters": {}, "cache_version": 1})
    store.begin_request(key)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        asyncio.run(provider.generate([]))
    assert provider.provider.calls == 0
    store.close()


@pytest.mark.parametrize("mode", ["truncated", "duplicate", "unknown", "empty"])
def test_invalid_response_does_not_become_success(tmp_path, monkeypatch, mode):
    class Invalid(MockProvider):
        async def generate(self, messages, **kwargs):
            result = await super().generate(messages, **kwargs)
            if "rerank_list" in kwargs["schema"]["properties"]:
                if mode == "truncated":
                    return replace(result, text='{"rerank_list":[')
                if mode == "empty":
                    return replace(result, text="")
                value = json.loads(result.text)
                value["rerank_list"][0] = value["rerank_list"][1] if mode == "duplicate" else 999999
                return replace(result, text=json.dumps(value))
            return result

    monkeypatch.setattr("agentreclab.core.runner.create", lambda config: Invalid())
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml")
    cfg.update(manifest=str(make_fixture(tmp_path / "data")), outputs=str(tmp_path / "runs"))
    result = asyncio.run(run_experiment(cfg))
    assert result["metrics"]["n_failed"] == 10
    assert result["metrics"]["exclude"]["MRR"] is None
    assert result["metrics"]["count_as_failure"]["MRR"] == 0


def test_cli_target_option_names(tmp_path):
    runner = CliRunner()
    fixture = make_fixture(tmp_path / "data")
    result = runner.invoke(app, ["validate", "--manifest", str(fixture)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["n_cases"] == 10
    for command, option in (("run", "--config"), ("report", "--out"), ("evaluate", "--run")):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert option in result.output
