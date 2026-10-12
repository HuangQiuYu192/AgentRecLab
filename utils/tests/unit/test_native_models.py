import asyncio
import copy
import json
from dataclasses import replace

import pytest

from models import create_model
from utils.config import ROOT, load_run
from utils.data import instant, load_dataset, make_fixture
from utils.llm import MockProvider
from utils.schema import ExperimentContext


def model_context(tmp_path, name, provider=None):
    cases, events, catalog, _ = load_dataset(make_fixture(tmp_path / "data"))
    config = load_run(ROOT / "configs/experiments/smoke_books.yaml", model=name)
    context = ExperimentContext(
        catalog,
        {e.event_id: e for e in events},
        cases,
        provider or MockProvider(),
        None,
        None,
        "controlled_comparison",
        config,
    )
    model = create_model(name)
    model.prepare(context)
    return model, context


def test_warmup_cutoff_and_read_only_evaluation(tmp_path):
    model, context = model_context(tmp_path, "memrec")
    case = context.cases[0]
    target = model.history(case)[-1]
    graph = model.graph(target.timestamp)
    assert all(
        instant(e.timestamp) < instant(target.timestamp)
        for events in graph.values()
        for e in events
    )
    assert target.event_id not in {e.event_id for events in graph.values() for e in events}
    asyncio.run(model.warmup())
    assert model.state["n_updates"] == 10
    assert len(model.state["completed_warmup_users"]) == 10
    for plan in model.state["warmup_manifest"]:
        training = {e.item_id for e in model.histories[plan["user_id"]]}
        target_item = context.interactions[plan["target_event_id"]].item_id
        assert set(plan["candidates"]) & training == {target_item}
    before = copy.deepcopy(model.state)
    asyncio.run(model.predict(case))
    assert model.state == before
    snapshot = tmp_path / "prepared.json"
    model.save_state(snapshot)
    restored = create_model("memrec")
    restored.prepare(context)
    restored.load_state(snapshot)
    assert asyncio.run(restored.predict(case)) == asyncio.run(model.predict(case))


def test_neighbor_memory_write_and_atomic_invalid_update(tmp_path):
    class Updates(MockProvider):
        invalid = False

        async def generate(self, messages, **kwargs):
            result = await super().generate(messages, **kwargs)
            if "user_memory" in kwargs["schema"]["properties"]:
                data = json.loads(result.text)
                data["neighbor_updates"] = (
                    [
                        {
                            "neighbor_id": "User-999999",
                            "memory_update": "untrusted",
                            "rationale": "test",
                        }
                    ]
                    if self.invalid
                    else []
                )
                return replace(result, text=json.dumps(data))
            return result

    provider = Updates()
    model, _ = model_context(tmp_path, "memrec", provider)
    before = copy.deepcopy(model.state)
    provider.invalid = True
    with pytest.raises(ValueError, match="allowed"):
        asyncio.run(model.warmup())
    assert model.state["user_profiles"] == before["user_profiles"]
    assert model.state["item_descriptions"] == before["item_descriptions"]
    assert model.state["n_updates"] == 0


@pytest.mark.parametrize("name", ["iagent", "i2agent", "memrec"])
def test_no_upstream_runtime_dependency(tmp_path, monkeypatch, name):
    # Missing reference paths cannot affect native prediction; also reject dynamic imports.
    import importlib.util
    from pathlib import Path

    original = Path.exists
    monkeypatch.setattr(Path, "exists", lambda p: False if "third_party" in str(p) else original(p))
    monkeypatch.setattr(
        importlib.util,
        "spec_from_file_location",
        lambda *a, **k: pytest.fail("native model dynamically loaded a file"),
    )
    model, context = model_context(tmp_path, name)
    result = asyncio.run(model.predict(context.cases[0]))
    assert set(result.ordered_item_ids) == set(context.cases[0].candidates)
