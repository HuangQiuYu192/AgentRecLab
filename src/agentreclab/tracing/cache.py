import asyncio
import json
import uuid
from dataclasses import asdict

import httpx

from agentreclab.core.schemas import LLMResult, LLMUsage
from agentreclab.providers.pricing import price
from agentreclab.state.snapshot import digest


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
            input_estimate = len(
                json.dumps(messages, ensure_ascii=False).encode()
            )  # Conservative token bound.
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
