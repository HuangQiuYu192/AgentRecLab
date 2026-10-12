import asyncio
import importlib.util
import json
import subprocess
import sys
import types

from agentreclab.core.config import ROOT
from agentreclab.providers.structured import validate_schema
from agentreclab.state.snapshot import file_hash, read_json, write_json


def upstream(name):
    lock = read_json(ROOT / "third_party" / "lock.json")[name]
    directory = ROOT / "third_party" / name
    if not directory.exists():
        raise RuntimeError("upstream checkout absent; run python scripts/setup_upstream.py")
    command = ["git", "-c", f"safe.directory={directory.as_posix()}", "-C", str(directory)]
    head = subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        command + ["status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if head != lock["commit"] or dirty:
        raise ValueError("upstream commit differs or tracked source is dirty")
    return directory


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def namespace(name, path):
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def item_mapping(catalog):
    # Leading zeroes and namespaces must survive round-trip.
    if all(i.isdigit() and str(int(i)) == i for i in catalog):
        return {i: int(i) for i in catalog}
    return {i: j + 1 for j, i in enumerate(catalog)}


class BaseModel:
    adapter_version = "0.1.0"

    def prepare(self, context):
        self.context = context
        self.state = {}

    async def observe(self, feedback):
        raise NotImplementedError("offline_static never reveals test feedback")

    def save_state(self, destination):
        write_json(destination, self.state)

    def load_state(self, source):
        self.state = read_json(source)


class SyncBridge:
    def __init__(self, provider, loop, case_id, candidates, phase="rerank"):
        self.provider, self.loop, self.case_id = provider, loop, case_id
        self.candidates, self.phase = candidates, phase
        self.error = None
        self.rankings = []
        self.model = "traced-provider"
        self.api_endpoint = "configured-provider"
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def generate_json(self, messages, properties, **kwargs):
        schema = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        stage = (
            "rerank"
            if "scores" in properties
            else ("retrieval" if "facets" in properties else "memory_update")
        )
        if self.phase in ("warmup", "memory_update"):
            stage = "warmup/" + stage
        return self.call(messages, schema, stage)

    def call(self, messages, schema, phase):
        if self.error is not None:
            raise self.error
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.provider.generate(
                    messages,
                    schema=schema,
                    candidate_ids=self.candidates,
                    case_id=self.case_id,
                    phase=phase,
                    adapter_version="0.1.0",
                    prompt_version=file_hash(ROOT / "third_party" / "lock.json"),
                ),
                self.loop,
            )
            result = future.result()
            value = json.loads(result.text)
            validate_schema(value, schema)
            if "rerank_list" in value:
                self.rankings.append(value["rerank_list"])
            return value
        except Exception as exc:
            self.error = exc
            raise

    def create(self, messages, response_format, **kwargs):
        schema = response_format["json_schema"]["schema"]
        fields = schema["properties"]
        phase = (
            "rerank"
            if "rerank_list" in fields
            else ("warmup" if "profile" in fields or "recommend_content" in fields else "retrieval")
        )
        value = self.call(messages, schema, phase)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(value)))
            ]
        )
