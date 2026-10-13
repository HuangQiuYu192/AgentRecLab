"""Explicit stage batches, durable job IDs, and result matching by custom_id."""

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import httpx

from utils.core.schema import BudgetExceeded, LLMResult, LLMUsage
from utils.experiment.llm import MockProvider, chat_payload, price
from utils.experiment.storage import canonical, digest, read_json, write_json


class BatchPending(RuntimeError):
    pass


def validate_batch(config):
    if not isinstance(config.get("use_batch", False), bool):
        raise ValueError("use_batch must be true or false")
    if not config.get("use_batch", False):
        return
    if config["provider"]["name"] not in ("mock", "openai"):
        raise ValueError("batch currently supports mock and OpenAI only")
    if config.get("invalid_prediction_policy", "fail") == "one_retry":
        raise ValueError("batch supports fail/append_missing; one_retry is online only")
    options = config.get("batch", {})
    if options.get("completion_window", "24h") != "24h":
        raise ValueError("batch completion_window must be 24h")
    if not 1 <= int(options.get("max_requests_per_file", 10000)) <= 50000:
        raise ValueError("max_requests_per_file must be in [1, 50000]")
    if not 1 <= int(options.get("max_input_bytes", 180_000_000)) <= 200_000_000:
        raise ValueError("max_input_bytes must be in [1, 200000000]")
    if not 0 < float(options.get("poll_interval_seconds", 30)) <= 60:
        raise ValueError("poll_interval_seconds must be in (0, 60]")
    if float(options.get("poll_timeout_seconds", 120)) <= 0:
        raise ValueError("poll_timeout_seconds must be positive")
    if config.get("budget_cny") is not None and not options.get("pricing"):
        raise ValueError("batch budget_cny requires explicitly dated batch.pricing")


class BatchClient:
    def __init__(self, config, directory, store, recorder, transport=None):
        validate_batch(config)
        self.config = config
        self.provider = config["provider"]
        self.options = config.get("batch", {})
        self.directory = Path(directory) / "batches"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.store, self.recorder, self.transport = store, recorder, transport
        self.mock = MockProvider() if self.provider["name"] == "mock" else None
        # Batch prices are supplied explicitly; never reuse standard prices silently.
        self.pricing = {**self.provider, "pricing": self.options.get("pricing")}

    async def http(self, method, path, **kwargs):
        endpoint = self.provider.get("endpoint") or os.environ.get(
            self.provider.get("endpoint_env", "OPENAI_BASE_URL"), "https://api.openai.com/v1"
        )
        host = urlparse(endpoint).hostname
        if urlparse(endpoint).scheme != "https" and host not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("remote batch endpoints must use HTTPS")
        secret = os.environ.get(self.provider.get("api_key_env", "OPENAI_API_KEY"), "")
        if not secret:
            raise ValueError("provider API key environment variable is not configured")
        async with httpx.AsyncClient(
            timeout=self.provider.get("timeout", 60), transport=self.transport
        ) as client:
            response = await client.request(
                method,
                endpoint.rstrip("/") + path,
                headers={"Authorization": f"Bearer {secret}"},
                **kwargs,
            )
            response.raise_for_status()
            return response

    async def generate_many(self, requests):
        # Recover a previous file before regrouping a partially consumed stage.
        identities = [digest({"provider": self.provider, "request": r}) for r in requests]
        for cid in identities:
            if self.store.get_response("batch:" + cid):
                continue
            index = self.directory / "request_jobs" / (cid + ".json")
            if index.exists():
                job = read_json(index)["job"]
                previous = read_json(self.directory / job / "requests.json")
                await self.execute_chunk(previous)
        pending, cached = [], {}
        for position, (request, cid) in enumerate(zip(requests, identities, strict=True)):
            value = self.store.get_response("batch:" + cid)
            if value:
                cached[position] = self.cached_result(value)
            else:
                pending.append((position, request))
        results, chunk, size = [], [], 0
        maximum = int(self.options.get("max_requests_per_file", 10000))
        max_bytes = int(self.options.get("max_input_bytes", 180_000_000))
        for _position, request in pending:
            body = chat_payload(self.provider, request["messages"], request["kwargs"]["schema"])
            item = {
                "custom_id": digest({"provider": self.provider, "request": request}),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            length = len((canonical(item) + "\n").encode())
            if length > max_bytes:
                raise ValueError("a single request exceeds batch max_input_bytes")
            if chunk and (len(chunk) >= maximum or size + length > max_bytes):
                results.extend(await self.execute_chunk(chunk))
                chunk, size = [], 0
            chunk.append((request, item))
            size += length
        if chunk:
            results.extend(await self.execute_chunk(chunk))
        for (position, _request), reply in zip(pending, results, strict=True):
            cached[position] = reply
        return [cached[i] for i in range(len(requests))]

    @staticmethod
    def cached_result(value):
        if "batch_error" in value:
            return RuntimeError(value["batch_error"])
        data = value["batch_result"]
        return LLMResult(**{**data, "usage": LLMUsage(**data["usage"]) if data["usage"] else None})

    def reserve(self, chunk):
        # Reserve the entire file atomically before submitting any paid request.
        with self.store.lock, self.store.db:
            calls, spent = self.store.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(cost),0) FROM accounting"
            ).fetchone()
            additions = []
            for _request, item in chunk:
                rid = "batch:" + item["custom_id"]
                if self.store.db.execute("SELECT 1 FROM accounting WHERE id=?", (rid,)).fetchone():
                    continue
                amount, _ = price(
                    self.pricing,
                    None,
                    len(canonical(item).encode()),
                    self.provider.get("max_tokens", 4000),
                )
                additions.append((rid, amount or 0, "reserved"))
            if calls + len(additions) > self.config.get("max_calls", 1000):
                raise BudgetExceeded("batch request budget exhausted before submission")
            budget = self.config.get("budget_cny")
            if budget is not None and spent + sum(row[1] for row in additions) > budget:
                raise BudgetExceeded("batch cost budget exhausted before submission")
            self.store.db.executemany("INSERT INTO accounting VALUES (?,?,?)", additions)
        new_ids = {row[0] for row in additions}
        for request, item in chunk:
            if "batch:" + item["custom_id"] in new_ids:
                self.recorder.record(
                    {
                        "event": "dispatch",
                        "case_id": request["kwargs"]["case_id"],
                        "phase": request["kwargs"]["phase"],
                        "batch": True,
                        "custom_id": item["custom_id"],
                        "request": request,
                    }
                )

    async def execute_chunk(self, chunk):
        job_id = digest([item for _, item in chunk])
        directory = self.directory / job_id
        directory.mkdir(exist_ok=True)
        state_path, output_path = directory / "job.json", directory / "results.json"
        if not (directory / "requests.json").exists():
            write_json(directory / "requests.json", chunk)
        for _request, item in chunk:
            write_json(
                self.directory / "request_jobs" / (item["custom_id"] + ".json"), {"job": job_id}
            )
        state = read_json(state_path) if state_path.exists() else {}
        if not output_path.exists():
            self.reserve(chunk)
            if self.mock:
                output = {}
                for request, item in reversed(chunk):
                    result = await self.mock.generate(request["messages"], **request["kwargs"])
                    output[item["custom_id"]] = {"result": asdict(result)}
                write_json(output_path, output)
                write_json(
                    state_path, {"status": "completed", "request_count": len(chunk), "mock": True}
                )
            else:
                if not state.get("input_file_id"):
                    content = "".join(canonical(item) + "\n" for _, item in chunk).encode()
                    (directory / "input.jsonl").write_bytes(content)
                    response = await self.http(
                        "POST",
                        "/files",
                        data={"purpose": "batch"},
                        files={"file": ("input.jsonl", content, "application/jsonl")},
                    )
                    state["input_file_id"] = response.json()["id"]
                    write_json(state_path, state)
                if not state.get("batch_id"):
                    if state.get("submission_started"):
                        raise RuntimeError(
                            f"batch submission outcome unknown; recover batch_id in {state_path} before resume"
                        )
                    state["submission_started"] = True
                    write_json(state_path, state)
                    response = await self.http(
                        "POST",
                        "/batches",
                        json={
                            "input_file_id": state["input_file_id"],
                            "endpoint": "/v1/chat/completions",
                            "completion_window": "24h",
                            "metadata": {"run_id": self.config["run_id"], "job_key": job_id},
                        },
                    )
                    state["batch_id"] = response.json()["id"]
                    write_json(state_path, state)
                started = time.monotonic()
                while True:
                    response = await self.http("GET", "/batches/" + state["batch_id"])
                    status = response.json()
                    state.update(status=status["status"], remote=status)
                    write_json(state_path, state)
                    if status["status"] in ("completed", "failed", "expired", "cancelled"):
                        break
                    remaining = self.options.get("poll_timeout_seconds", 120) - (
                        time.monotonic() - started
                    )
                    if remaining <= 0:
                        raise BatchPending(f"{state['batch_id']} pending; run again with --resume")
                    await asyncio.sleep(
                        min(self.options.get("poll_interval_seconds", 30), remaining)
                    )
                output = {}
                expected = {item["custom_id"] for _, item in chunk}
                for field in ("output_file_id", "error_file_id"):
                    if not status.get(field):
                        continue
                    response = await self.http("GET", "/files/" + status[field] + "/content")
                    (directory / (field + ".jsonl")).write_text(response.text, encoding="utf-8")
                    for line in response.text.splitlines():
                        row = json.loads(line)
                        cid = row["custom_id"]
                        if cid not in expected or cid in output:
                            raise ValueError("unknown or duplicate batch custom_id")
                        output[cid] = row
                for cid in expected - output.keys():
                    output[cid] = {
                        "error": {"message": f"batch {status['status']}: missing result"}
                    }
                write_json(output_path, output)
        output = read_json(output_path)
        replies = []
        for request, item in chunk:
            cid = item["custom_id"]
            row = output[cid]
            try:
                if "result" in row:
                    data = row["result"]
                    result = LLMResult(
                        **{**data, "usage": LLMUsage(**data["usage"]) if data["usage"] else None}
                    )
                elif row.get("error") or (row.get("response") or {}).get("status_code") != 200:
                    self.store.cache_response("batch:" + cid, {"batch_error": canonical(row)})
                    replies.append(RuntimeError(canonical(row)))
                    continue
                else:
                    body = row["response"]["body"]
                    usage = body.get("usage")
                    result = LLMResult(
                        body["choices"][0]["message"]["content"],
                        LLMUsage(
                            usage["prompt_tokens"],
                            usage["completion_tokens"],
                            (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                            (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                        )
                        if usage
                        else None,
                        self.provider["name"],
                        body.get("model", self.provider["model"]),
                        body.get("id"),
                        "",
                        0,
                    )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                error = f"invalid batch response for {cid}: {exc}"
                self.store.cache_response("batch:" + cid, {"batch_error": error})
                replies.append(RuntimeError(error))
                continue
            rid = "batch:" + cid
            recorded = self.store.db.execute(
                "SELECT status FROM accounting WHERE id=?", (rid,)
            ).fetchone()
            if recorded and recorded[0] != "completed":
                amount, estimated = price(
                    self.pricing,
                    result.usage,
                    len(canonical(item).encode()),
                    len(str(result.text).encode()),
                )
                ref = self.recorder.raw(asdict(result))
                self.recorder.record(
                    {
                        "event": "response",
                        "case_id": request["kwargs"]["case_id"],
                        "phase": request["kwargs"]["phase"],
                        "batch": True,
                        "custom_id": cid,
                        "cached": False,
                        "cost_cny": amount,
                        "estimated": estimated,
                        "usage": asdict(result.usage) if result.usage else None,
                        "raw_response_ref": ref,
                    }
                )
                self.store.settle(rid, amount or 0)
            self.store.cache_response("batch:" + cid, {"batch_result": asdict(result)})
            replies.append(result)
        return replies
