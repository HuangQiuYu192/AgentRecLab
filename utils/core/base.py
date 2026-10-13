import json

from utils.experiment.llm import validate_schema
from utils.experiment.storage import read_json, write_json


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
    supports_batch = False

    def __init__(self, use_batch=False):
        if use_batch and not self.supports_batch:
            raise ValueError(f"{type(self).__name__} does not support batch")
        self.use_batch = use_batch

    def prepare(self, context):
        self.context = context
        self.state = {}
        self.item_ids = item_mapping(context.catalog)
        self.original_ids = {v: k for k, v in self.item_ids.items()}

    def save_state(self, path):
        write_json(path, self.state)

    def load_state(self, path):
        self.state = read_json(path)

    def request(self, messages, schema, case_id, phase, candidates=()):
        return {
            "messages": messages.copy(),
            "kwargs": {
                "schema": schema,
                "candidate_ids": [self.item_ids[i] for i in candidates],
                "case_id": case_id,
                "phase": phase,
                "adapter_version": "0.2.0",
                "prompt_version": "native-v2",
            },
        }

    @staticmethod
    def parse(result, request):
        data = json.loads(result.text)
        validate_schema(data, request["kwargs"]["schema"])
        return data

    async def ask(self, messages, schema, case_id, phase, candidates=()):
        request = self.request(messages, schema, case_id, phase, candidates)
        result = await self.context.provider.generate(request["messages"], **request["kwargs"])
        return self.parse(result, request)

    async def predict(self, case):
        steps = self.workflow(case)
        answer = None
        try:
            while True:
                try:
                    request = steps.send(answer)
                except StopIteration as done:
                    return done.value
                result = await self.context.provider.generate(
                    request["messages"], **request["kwargs"]
                )
                answer = self.parse(result, request)
        finally:
            steps.close()

    async def run_batch_steps(self, cases, batch):
        active = {case.case_id: self.workflow(case) for case in cases}
        answers, results = {}, {}
        try:
            while active:
                requests = {}
                for cid, steps in list(active.items()):
                    try:
                        requests[cid] = steps.send(answers.pop(cid, None))
                    except StopIteration as done:
                        results[cid] = done.value
                        del active[cid]
                    except Exception as exc:
                        results[cid] = exc
                        del active[cid]
                if not requests:
                    continue
                replies = await batch.generate_many(list(requests.values()))
                for cid, request, reply in zip(requests, requests.values(), replies, strict=True):
                    try:
                        if isinstance(reply, Exception):
                            raise reply
                        answers[cid] = self.parse(reply, request)
                    except Exception as exc:
                        results[cid] = exc
                        active.pop(cid).close()
            return results
        finally:
            for steps in active.values():
                steps.close()

    def history(self, case):
        return [self.context.interactions[i] for i in case.history_event_ids]
