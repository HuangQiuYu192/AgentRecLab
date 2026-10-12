import json

from utils.llm import validate_schema
from utils.storage import read_json, write_json


def item_mapping(catalog):
    ids = list(catalog)
    try:
        if all(str(int(i)) == i for i in ids):
            return {i: int(i) for i in ids}
    except ValueError:
        pass
    return {i: n for n, i in enumerate(ids, 1)}


def object_schema(**properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING = {"type": "string"}
RANK_SCHEMA = object_schema(
    rerank_list={"type": "array", "items": {"type": "integer"}},
    explanation={"type": "array", "items": STRING},
)


class BaseModel:
    def prepare(self, context):
        self.context = context
        self.state = {}
        self.item_ids = item_mapping(context.catalog)
        self.original_ids = {v: k for k, v in self.item_ids.items()}

    def save_state(self, path):
        write_json(path, self.state)

    def load_state(self, path):
        self.state = read_json(path)

    async def ask(self, messages, schema, case_id, phase, candidates=()):
        result = await self.context.provider.generate(
            messages,
            schema=schema,
            candidate_ids=[self.item_ids[i] for i in candidates],
            case_id=case_id,
            phase=phase,
            adapter_version="0.2.0",
            prompt_version="native-v2",
        )
        data = json.loads(result.text)
        validate_schema(data, schema)
        return data

    def history(self, case):
        return [self.context.interactions[i] for i in case.history_event_ids]
