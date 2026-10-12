"""Optional audit against pinned source; normal tests never require official checkouts."""

import asyncio
import copy
import importlib.util
import logging
import os
from types import SimpleNamespace

import pytest

from agentreclab.config import ROOT, load_run
from agentreclab.data import load_dataset, make_fixture
from agentreclab.evaluation import evaluate_case
from agentreclab.llm import MockProvider
from agentreclab.models import create_model
from agentreclab.schema import ExperimentContext
from agentreclab.storage import digest

pytestmark = pytest.mark.skipif(
    os.environ.get("AGENTRECLAB_REFERENCE_TESTS") != "1",
    reason="optional reference audit; no upstream required",
)


class RecordingMock(MockProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    async def generate(self, messages, **kwargs):
        self.requests.append(copy.deepcopy({"messages": messages, "schema": kwargs["schema"]}))
        return await super().generate(messages, **kwargs)


@pytest.mark.parametrize("name", ["iagent", "i2agent"])
def test_reference_requests_and_ranking(tmp_path, name):
    pytest.importorskip("openai")
    path = ROOT / "third_party/iAgent/model.py"
    if not path.exists():
        pytest.skip("optional reference checkout absent")
    spec = importlib.util.spec_from_file_location("reference_iagent", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases, events, catalog, _ = load_dataset(make_fixture(tmp_path / "data"))
    cfg = load_run(ROOT / "configs/runs/smoke_books.yaml", model=name)
    provider = RecordingMock()
    native = create_model(name)
    native.prepare(
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
    case = cases[0]
    history = native.history(case)
    _, candidate_text, ids = native.inputs(case, history)
    negative = catalog[case.metadata["warmup_negative"]]
    task = dict(
        instruction=case.instruction,
        title=[catalog[e.item_id]["title"] for e in history],
        description=[catalog[e.item_id]["description"] for e in history],
        asin=[native.item_ids[e.item_id] for e in history],
        reviewText=[e.review or "" for e in history],
        answer=native.item_ids[case.positive_item_id],
        ranked_list_str=candidate_text,
        pure_ranked_list=ids,
        neg_sample_title=negative["title"],
        neg_sample_descript=negative["description"],
    )

    async def execute():
        prediction = await native.predict(case)
        reference_provider = RecordingMock()
        loop = asyncio.get_running_loop()

        def completion(**kwargs):
            response = asyncio.run_coroutine_threadsafe(
                reference_provider.generate(
                    kwargs["messages"],
                    schema=kwargs["response_format"]["json_schema"]["schema"],
                    candidate_ids=ids,
                ),
                loop,
            ).result(10)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=response.text))]
            )

        module.OpenAI = lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=completion))
        )
        original = getattr(module, "iAgent" if name == "iagent" else "i2Agent")(
            task, logging.getLogger("reference")
        )
        result = await asyncio.to_thread(original.run)
        assert digest(provider.requests) == digest(reference_provider.requests)
        assert result["MRR"] == evaluate_case(case, prediction)["MRR"]

    asyncio.run(execute())
