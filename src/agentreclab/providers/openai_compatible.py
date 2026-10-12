import os
import time
from urllib.parse import quote, urlparse

import httpx

from agentreclab.core.schemas import LLMResult, LLMUsage


class OpenAICompatible:
    def __init__(self, config, transport=None):
        self.config = config
        self.name, self.model = config["name"], config["model"]
        self.transport = transport

    async def generate(self, messages, **kwargs):
        config = self.config
        endpoint = config.get("endpoint") or os.environ.get(
            config.get("endpoint_env", "OPENAI_BASE_URL"), "https://api.openai.com/v1"
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
