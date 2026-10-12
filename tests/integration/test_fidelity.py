import asyncio
import copy
import logging
from dataclasses import replace

import pandas as pd
import pytest

from agentreclab.core.config import ROOT, load_run
from agentreclab.core.registry import create_model
from agentreclab.core.schemas import ExperimentContext
from agentreclab.data.fixtures import make_fixture
from agentreclab.data.instructrec import convert
from agentreclab.data.manifest import DOMAINS, load_dataset
from agentreclab.eval.ranking import evaluate_case
from agentreclab.models.base import SyncBridge, item_mapping, load_module, upstream
from agentreclab.providers.mock import MockProvider
from agentreclab.state.snapshot import digest


class RecordingMock(MockProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def generate(self, messages, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "schema": kwargs["schema"]}))
        return await super().generate(messages, **kwargs)


@pytest.mark.parametrize("model", ["iagent", "i2agent"])
def test_official_differential_valid_predictions(tmp_path, model):
    cases, events, catalog, _ = load_dataset(make_fixture(tmp_path / "data"))
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml", model=model)
    provider = RecordingMock()
    context = ExperimentContext(
        catalog,
        {e.event_id: e for e in events},
        cases,
        provider,
        None,
        None,
        "controlled_comparison",
        cfg,
    )
    adapter = create_model(model)
    adapter.prepare(context)
    original_class = getattr(adapter.module, adapter.agent_class)
    captured = []

    class Capture(original_class):
        def __init__(self, task, logger):
            captured.append(copy.deepcopy(task))
            super().__init__(task, logger)

    setattr(adapter.module, adapter.agent_class, Capture)

    async def execute():
        prediction = await adapter.predict(cases[0])
        original = load_module("_diff_original_" + model, upstream("iAgent") / "model.py")
        original.OpenAI = lambda **kwargs: None
        direct_provider = RecordingMock()
        task = captured[0]
        assert task["answer"] is None  # Label removed from the adapter's algorithm input.
        task["answer"] = int(cases[0].positive_item_id)
        bridge = SyncBridge(
            direct_provider,
            asyncio.get_running_loop(),
            "direct",
            [int(i) for i in cases[0].candidates],
        )
        direct = getattr(original, adapter.agent_class)(task, logging.getLogger("test"))
        direct.client = bridge
        result = await asyncio.to_thread(direct.run)
        assert digest(provider.requests) == digest(direct_provider.requests)
        assert prediction.ordered_item_ids == tuple(str(i) for i in bridge.rankings[-1])
        assert result["MRR"] == evaluate_case(cases[0], prediction)["MRR"]

    asyncio.run(execute())


@pytest.mark.parametrize("domain", DOMAINS)
def test_converter_preserves_original_order_and_index(tmp_path, domain):
    raw = tmp_path / "raw.pkl"
    mapping = tmp_path / "catalog.csv"
    pd.DataFrame(
        [
            {
                "asin": [20, 21, 22, 3],
                "instruction": "Synthetic instruction",
                "ranked_lists": [9, 8, 7, 6, 5, 4, 3, 2, 1, 10],
                "title": ["a", "b", "c", "d"],
                "description": ["a", "b", "c", "d"],
                "reviewText": ["train1", "train2", "train3", "TEST_REVIEW_MUST_NOT_APPEAR"],
            }
        ],
        index=[97],
    ).to_pickle(raw)
    pd.DataFrame(
        [{"index": i, "title": f"item {i}", "description": "description"} for i in range(1, 31)]
    ).to_csv(mapping, index=False)
    manifest = convert(
        domain,
        {
            "pickle": str(raw),
            "mapping": str(mapping),
            "out": str(tmp_path / "canonical"),
            "trusted_pickle": True,
        },
    )
    cases, events, _, provenance = load_dataset(manifest)
    assert cases[0].candidates == ("9", "8", "7", "6", "5", "4", "3", "2", "1", "10")
    assert cases[0].positive_item_id == "3"
    assert cases[0].metadata["original_row_index"] == "97"
    assert cases[0].user_id == f"instructrec:{domain}:row:0"
    assert "TEST_REVIEW_MUST_NOT_APPEAR" not in str(events)
    assert provenance["provenance"]["candidate_source"] == "original"
    with pytest.raises(ValueError, match="trusted"):
        convert(domain, {"pickle": str(raw)})


def test_identifier_mapping_is_bijective():
    ids = ("01", "1", "namespace:item:1", "用户:物品")
    mapped = item_mapping(dict.fromkeys(ids))
    inverse = {v: k for k, v in mapped.items()}
    assert tuple(inverse[mapped[i]] for i in ids) == ids
    assert len(set(mapped.values())) == len(ids)


def test_label_change_does_not_change_model_requests(tmp_path):
    cases, events, catalog, _ = load_dataset(make_fixture(tmp_path / "data"))
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml")

    async def execute(case):
        provider = RecordingMock()
        adapter = create_model("iagent")
        adapter.prepare(
            ExperimentContext(
                catalog,
                {e.event_id: e for e in events},
                cases,
                provider,
                None,
                None,
                "controlled_comparison",
                cfg,
            )
        )
        prediction = await adapter.predict(case)
        return digest(provider.requests), prediction.ordered_item_ids

    assert asyncio.run(execute(cases[0])) == asyncio.run(
        execute(replace(cases[0], positive_item_id=cases[0].candidates[-1]))
    )
