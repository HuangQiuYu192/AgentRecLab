import asyncio
import json

import httpx
import pytest

from agentreclab.core.errors import BudgetExceeded
from agentreclab.providers.mock import MockProvider
from agentreclab.providers.openai_compatible import OpenAICompatible
from agentreclab.state.store import StateStore
from agentreclab.tracing.cache import TracedProvider
from agentreclab.tracing.recorder import Recorder
from agentreclab.tracing.redaction import redact


def test_cache_provider_isolation_and_budget(tmp_path):
    store, recorder = StateStore(tmp_path / "state.db"), Recorder(tmp_path)
    mock = MockProvider()
    p = TracedProvider(mock, {"name": "mock", "model": "v1"}, store, recorder, max_calls=1)
    first = asyncio.run(p.generate([], schema={"properties": {"knowledge": {"type": "string"}}}))
    assert (
        asyncio.run(p.generate([], schema={"properties": {"knowledge": {"type": "string"}}})).text
        == first.text
    )
    assert mock.calls == 1
    with pytest.raises(BudgetExceeded):
        asyncio.run(p.generate([{"role": "user", "content": "different"}]))
    q = TracedProvider(mock, {"name": "mock", "model": "v2"}, store, recorder, max_calls=2)
    asyncio.run(q.generate([], schema={"properties": {"knowledge": {"type": "string"}}}))
    assert mock.calls == 2
    assert recorder.summary()["cache_hits"] == 1
    store.close()


@pytest.mark.parametrize("error", [429, 500, "timeout", 400])
def test_http_retry_and_missing_usage(tmp_path, monkeypatch, error):
    monkeypatch.setenv("TEST_KEY", "synthetic-only-secret")
    count = []

    def handler(request):
        count.append(request)
        if len(count) == 1:
            if error == "timeout":
                raise httpx.ReadTimeout("test", request=request)
            return httpx.Response(error, json={"error": "test"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"knowledge":"ok"}'}}], "id": "r"}
        )

    cfg = {
        "name": "openai",
        "model": "test",
        "api_key_env": "TEST_KEY",
        "retry_budget": 1,
        "retry_backoff": 0,
    }
    raw = OpenAICompatible(cfg, httpx.MockTransport(handler))
    store, recorder = StateStore(tmp_path / "state.db"), Recorder(tmp_path)
    p = TracedProvider(raw, cfg, store, recorder)
    if error == 400:
        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(p.generate([]))
        assert len(count) == 1
    else:
        result = asyncio.run(p.generate([]))
        assert result.usage is None
        assert len(count) == 2
        assert recorder.summary()["unknown_cost_calls"] == 1
    assert "synthetic-only-secret" not in (tmp_path / "traces.jsonl").read_text()
    store.close()


def test_azure_request(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "synthetic-key")
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    p = OpenAICompatible(
        {
            "name": "azure",
            "model": "gpt-test",
            "deployment": "deployment",
            "api_version": "2024-08-01-preview",
            "endpoint": "https://example.invalid",
            "api_key_env": "TEST_KEY",
        },
        httpx.MockTransport(handler),
    )
    asyncio.run(p.generate([]))
    assert "/deployments/deployment/chat/completions" in str(captured[0].url)
    assert "api-version=" in str(captured[0].url)
    assert captured[0].headers["api-key"] == "synthetic-key"


def test_empty_response_and_redaction(tmp_path):
    class Empty(MockProvider):
        async def generate(self, messages, **kwargs):
            from dataclasses import replace

            return replace(await super().generate(messages, **kwargs), text="")

    store = StateStore(tmp_path / "s.db")
    p = TracedProvider(Empty(), {"name": "mock"}, store, Recorder(tmp_path))
    with pytest.raises(ValueError, match="empty"):
        asyncio.run(p.generate([]))
    value = redact(
        {
            "api_key": "secret",
            "Authorization": "Bearer sensitive",
            "nested": "api_key=abc sk-1234567890",
        }
    )
    assert "abc" not in json.dumps(value)
    assert value["api_key"] == "[REDACTED]"
    store.close()
