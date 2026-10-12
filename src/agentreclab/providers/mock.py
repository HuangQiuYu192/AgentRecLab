import json

from agentreclab.core.schemas import LLMResult, LLMUsage


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
