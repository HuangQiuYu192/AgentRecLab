import asyncio

import pytest

from utils.experiment.evaluation import user_bootstrap
from utils.experiment.llm import price


def test_bootstrap_unit_is_user():
    rows = [
        {"user_id": "one", "metrics": {"MRR": 1}},
        {"user_id": "one", "metrics": {"MRR": 0}},
        {"user_id": "two", "metrics": {"MRR": 0.5}},
    ]
    result = user_bootstrap(rows, n=100)
    assert result["unit"] == "user"
    assert result["lower"] == result["upper"] == 0.5


def test_pricing_estimates_and_cost_budget(tmp_path):
    from utils.core.schema import BudgetExceeded, LLMUsage
    from utils.experiment.llm import MockProvider, TracedProvider
    from utils.experiment.storage import Recorder, StateStore

    pricing = {
        "currency": "CNY",
        "effective_date": "2026-10-12",
        "input_per_million": 2,
        "cached_per_million": 1,
        "output_per_million": 4,
    }
    assert price({"name": "test", "pricing": pricing}, LLMUsage(1000, 100, 500)) == (0.0019, False)
    cfg = {"name": "test", "model": "mock_transport_only", "pricing": pricing, "max_tokens": 1000}
    store = StateStore(tmp_path / "s.db")
    mock = MockProvider()
    p = TracedProvider(mock, cfg, store, Recorder(tmp_path), budget_cny=0)
    with pytest.raises(BudgetExceeded):
        asyncio.run(p.generate([]))
    assert mock.calls == 0
    store.close()
