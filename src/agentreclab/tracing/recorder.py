import os
import threading
from pathlib import Path

from agentreclab.state.snapshot import canonical, digest, write_json
from agentreclab.tracing.redaction import redact


class Recorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def raw(self, response):
        key = digest(response)
        path = self.directory / "raw" / f"{key}.json"
        # Raw responses remain local; secrets are still removed, user text is private.
        with self.lock:
            if not path.exists():
                write_json(path, redact(response))
                try:
                    os.chmod(path, 0o600)
                except OSError:
                    pass
        return str(path.resolve())

    def record(self, row):
        with self.lock, (self.directory / "traces.jsonl").open("a", encoding="utf-8") as f:
            f.write(canonical(redact(row)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def summary(self):
        import json

        path = self.directory / "traces.jsonl"
        rows = (
            [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if path.exists()
            else []
        )
        calls = [r for r in rows if r.get("event") == "response" and not r.get("cached")]
        return {
            "total_calls": len([r for r in rows if r.get("event") == "dispatch"]),
            "cache_hits": sum(r.get("cached", False) for r in rows),
            "total_input_tokens": sum((r.get("usage") or {}).get("input_tokens", 0) for r in calls),
            "total_output_tokens": sum(
                (r.get("usage") or {}).get("output_tokens", 0) for r in calls
            ),
            "total_cost_cny": sum(r.get("cost_cny") or 0 for r in calls),
            "cost_estimated": any(r.get("estimated", False) for r in calls),
            "unknown_cost_calls": sum(r.get("cost_cny") is None for r in calls),
            "phases": {
                p: {
                    "calls": sum(r.get("phase") == p for r in calls),
                    "input_tokens": sum(
                        (r.get("usage") or {}).get("input_tokens", 0)
                        for r in calls
                        if r.get("phase") == p
                    ),
                    "output_tokens": sum(
                        (r.get("usage") or {}).get("output_tokens", 0)
                        for r in calls
                        if r.get("phase") == p
                    ),
                    "cost_cny": sum(r.get("cost_cny") or 0 for r in calls if r.get("phase") == p),
                }
                for p in sorted({r.get("phase", "unknown") for r in calls})
            },
        }
