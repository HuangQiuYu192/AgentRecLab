import asyncio
import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import date
from urllib.parse import quote, urlparse

import httpx

from utils.schema import LLMResult, LLMUsage
from utils.storage import digest


class MockProvider:
    name = "mock"
    model = "deterministic-v1"

    def __init__(self, config=None):
        self.calls = 0

    async def generate(self, messages, **kwargs):
        self.calls += 1
        ids = list(kwargs.get("candidate_ids", []))
        schema = kwargs.get("schema", {})
        properties = schema.get("properties", schema)

        def fill(s):
            kind = s.get("type", "string")
            if kind == "object":
                return {k: fill(v) for k, v in s.get("properties", {}).items()}
            if kind == "array":
                return []
            if kind in ("number", "integer"):
                return 0
            return "Synthetic mock output; no real recommendation evidence."

        data = {k: fill(v) for k, v in properties.items()}
        if "rerank_list" in data:
            data["rerank_list"] = [int(i) for i in ids]
        if "ordered_item_ids" in data:
            data["ordered_item_ids"] = [str(i) for i in ids]
        if "scores" in data:
            data["scores"] = [
                {
                    "item_id": int(i),
                    "score": 1 - j / max(1, len(ids)),
                    "rationale": "Mock candidate order",
                }
                for j, i in enumerate(ids)
            ]
        text = json.dumps(data)
        return LLMResult(
            text,
            LLMUsage(len(json.dumps(messages)) // 4, len(text) // 4),
            self.name,
            self.model,
            f"mock-{self.calls}",
            "",
            0.0,
        )


class OpenAICompatible:
    def __init__(self, config, transport=None):
        self.config = config
        self.name, self.model = config["name"], config["model"]
        self.transport = transport

    async def generate(self, messages, **kwargs):
        config = self.config
        endpoint = config.get("endpoint") or os.environ.get(
            config.get("endpoint_env", "OPENAI_BASE_URL"),
            "" if config["name"] == "azure" else "https://api.openai.com/v1",
        )
        if config["name"] == "azure" and not endpoint:
            raise ValueError("Azure endpoint is required")
        secret = os.environ.get(config.get("api_key_env", "OPENAI_API_KEY"), "")
        if not secret:
            raise ValueError("provider API key environment variable is not configured")
        host = urlparse(endpoint).hostname
        if urlparse(endpoint).scheme != "https" and host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("remote provider endpoints must use HTTPS")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": config.get("temperature", 0),
            "max_tokens": config.get("max_tokens", 4000),
        }
        caps = config.get("capabilities", {})
        if caps.get("supports_seed") and config.get("seed") is not None:
            payload["seed"] = config["seed"]
        if config.get("thinking") is not None:
            payload.update(config["thinking"])
        if caps.get("supports_json_schema") and kwargs.get("schema"):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "agentreclab", "strict": True, "schema": kwargs["schema"]},
            }
        elif config.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
            # JSON mode alone does not communicate the requested keys/types.
            if kwargs.get("schema"):
                payload["messages"] = [
                    {
                        "role": "system",
                        "content": "Return JSON matching this schema: "
                        + json.dumps(kwargs["schema"]),
                    }
                ] + messages
        if config["name"] == "azure":
            deployment = config.get("deployment")
            version = config.get("api_version")
            if not deployment or not version:
                raise ValueError("Azure deployment and API version are required")
            url = f"{endpoint.rstrip('/')}/openai/deployments/{quote(deployment, safe='')}/chat/completions"
            headers, params = {"api-key": secret}, {"api-version": version}
        else:
            url = endpoint.rstrip("/") + "/chat/completions"
            headers, params = {"Authorization": f"Bearer {secret}"}, None
        start = time.perf_counter()
        async with httpx.AsyncClient(
            timeout=config.get("timeout", 60), transport=self.transport
        ) as client:
            response = await client.post(url, headers=headers, params=params, json=payload)
            response.raise_for_status()
            data = response.json()
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("empty provider response")
        usage = data.get("usage")
        parsed = (
            LLMUsage(
                usage["prompt_tokens"],
                usage["completion_tokens"],
                usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
            )
            if usage
            else None
        )
        return LLMResult(
            text,
            parsed,
            self.name,
            data.get("model", self.model),
            data.get("id"),
            "",
            (time.perf_counter() - start) * 1000,
        )


def price(config, usage, input_estimate=0, output_estimate=0):
    if config["name"] == "mock":
        return 0.0, True
    prices = config.get("pricing")
    if not prices:
        return None, True
    date.fromisoformat(prices["effective_date"])
    if prices.get("currency") != "CNY":
        raise ValueError("pricing must be CNY; explicitly supply dated FX-converted prices")
    incoming = usage.input_tokens if usage else input_estimate
    outgoing = usage.output_tokens if usage else output_estimate
    cached = (usage.cached_input_tokens or 0) if usage else 0
    return (
        (incoming - cached) * prices["input_per_million"]
        + cached * prices.get("cached_per_million", prices["input_per_million"])
        + outgoing * prices["output_per_million"]
    ) / 1e6, usage is None


def validate_schema(value, schema):
    """Validate the small JSON-schema subset used by audited upstream prompts."""
    kind = schema.get("type")
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
    }
    if kind in types and (
        not isinstance(value, types[kind])
        or (kind in ("number", "integer") and isinstance(value, bool))
    ):
        raise ValueError(f"structured response requires {kind}")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        if any(k not in value for k in schema.get("required", [])):
            raise ValueError("structured response missing required fields")
        if schema.get("additionalProperties") is False and any(k not in properties for k in value):
            raise ValueError("structured response has unexpected fields")
        for key, spec in properties.items():
            if key in value:
                validate_schema(value[key], spec)
    if isinstance(value, list):
        for item in value:
            validate_schema(item, schema.get("items", {}))


def create_provider(config):
    if config["name"] == "mock":
        return MockProvider(config)
    if config["name"] in ("openai", "azure", "qwen", "compatible"):
        return OpenAICompatible(config)
    raise ValueError(f"unknown provider {config['name']}")


class TracedProvider:
    """Durable request replay. Each retry consumes a call reservation before dispatch."""

    def __init__(self, provider, config, store, recorder, max_calls=1000, budget_cny=None):
        self.provider, self.config, self.store, self.recorder = provider, config, store, recorder
        self.max_calls, self.budget_cny = max_calls, budget_cny
        self.lock = asyncio.Lock()

    async def generate(self, messages, **kwargs):
        phase, cid = kwargs.get("phase", "rerank"), kwargs.get("case_id", "prepare")
        # Phase/case labels do not affect response bytes; candidate IDs do affect mock output.
        semantic = {k: v for k, v in kwargs.items() if k not in ("case_id", "phase")}
        request = {
            "provider": self.config,
            "messages": messages,
            "parameters": semantic,
            "cache_version": 1,
        }
        key = digest(request)
        async with self.lock:
            cached = self.store.get_response(key)
            if cached:
                result = LLMResult(
                    **{**cached, "usage": LLMUsage(**cached["usage"]) if cached["usage"] else None}
                )
                self.recorder.record(
                    {
                        "event": "response",
                        "case_id": cid,
                        "phase": phase,
                        "cache_key": key,
                        "cached": True,
                        "input_hash": digest(messages),
                        "output_hash": digest(result.text),
                        "raw_response_ref": result.raw_response_ref,
                        "cost_cny": 0,
                        "usage": asdict(result.usage) if result.usage else None,
                    }
                )
                return result
            # Include the schema added for JSON-only endpoints in the byte/token bound.
            input_estimate = len(json.dumps(request, ensure_ascii=False).encode())
            reserved_cost, _ = price(
                self.config, None, input_estimate, self.config.get("max_tokens", 4000)
            )
            if self.budget_cny is not None and reserved_cost is None:
                raise ValueError("a cost budget requires a dated provider price table")
            for attempt in range(self.config.get("retry_budget", 2) + 1):
                self.store.begin_request(key)
                rid = str(uuid.uuid4())
                try:
                    self.store.reserve(rid, reserved_cost or 0, self.max_calls, self.budget_cny)
                except Exception:
                    self.store.finish_request(key)
                    raise
                self.recorder.record(
                    {
                        "event": "dispatch",
                        "case_id": cid,
                        "phase": phase,
                        "cache_key": key,
                        "attempt": attempt,
                        "input_hash": digest(messages),
                        "request": request,
                        "reservation_id": rid,
                    }
                )
                try:
                    result = await self.provider.generate(messages, **kwargs)
                    if not result.text or not result.text.strip():
                        raise ValueError("empty LLM response")
                except Exception as exc:
                    self.store.finish_request(key)
                    self.recorder.record(
                        {
                            "event": "error",
                            "case_id": cid,
                            "phase": phase,
                            "cache_key": key,
                            "attempt": attempt,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    transient = isinstance(exc, (httpx.TimeoutException, httpx.ConnectError)) or (
                        isinstance(exc, httpx.HTTPStatusError)
                        and (exc.response.status_code == 429 or exc.response.status_code >= 500)
                    )
                    if not transient or attempt >= self.config.get("retry_budget", 2):
                        raise
                    await asyncio.sleep(min(self.config.get("retry_backoff", 0.1) * 2**attempt, 5))
                    continue
                amount, estimated = price(
                    self.config, result.usage, input_estimate, len(result.text.encode())
                )
                # Preserve response before downstream parsing/state updates: failed JSON is replayable.
                ref = self.recorder.raw(
                    {
                        "text": result.text,
                        "usage": asdict(result.usage) if result.usage else None,
                        "model": result.model,
                        "request_id": result.request_id,
                    }
                )
                result = LLMResult(
                    **{**asdict(result), "usage": result.usage, "raw_response_ref": ref}
                )
                self.store.cache_response(key, asdict(result))
                self.store.settle(rid, amount or 0)
                self.store.finish_request(key)
                self.recorder.record(
                    {
                        "event": "response",
                        "case_id": cid,
                        "phase": phase,
                        "cache_key": key,
                        "attempt": attempt,
                        "cached": False,
                        "input_hash": digest(messages),
                        "output_hash": digest(result.text),
                        "raw_response_ref": ref,
                        "latency_ms": result.latency_ms,
                        "usage": asdict(result.usage) if result.usage else None,
                        "cost_cny": amount,
                        "estimated": estimated,
                    }
                )
                return result
