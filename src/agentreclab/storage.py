import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import asdict, is_dataclass
from pathlib import Path


def canonical(value) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class StateStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, status TEXT, result TEXT);
        CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, response TEXT);
        CREATE TABLE IF NOT EXISTS accounting (id TEXT PRIMARY KEY, cost REAL, status TEXT);
        CREATE TABLE IF NOT EXISTS inflight (key TEXT PRIMARY KEY, status TEXT);
        """)
        self.db.commit()

    def close(self):
        self.db.close()

    def put_case(self, cid, status, result=None):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO cases VALUES (?,?,?)", (cid, status, canonical(result))
            )

    def rows(self):
        import json

        with self.lock:
            return [
                {"case_id": cid, "status": status, **(json.loads(result) or {})}
                for cid, status, result in self.db.execute(
                    "SELECT id,status,result FROM cases ORDER BY id"
                )
            ]

    def get_response(self, key):
        import json

        with self.lock:
            row = self.db.execute("SELECT response FROM responses WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_response(self, key, result):
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?)", (key, canonical(result))
            )

    def reserve(self, rid, amount, max_calls, budget):
        from agentreclab.schema import BudgetExceeded

        with self.lock, self.db:
            calls, cost = self.db.execute(
                "SELECT COUNT(*),COALESCE(SUM(cost),0) FROM accounting"
            ).fetchone()
            if calls >= max_calls or (budget is not None and cost + amount > budget):
                raise BudgetExceeded("request/cost budget exhausted before dispatch")
            self.db.execute("INSERT INTO accounting VALUES (?,?,?)", (rid, amount, "reserved"))

    def begin_request(self, key):
        with self.lock, self.db:
            row = self.db.execute("SELECT status FROM inflight WHERE key=?", (key,)).fetchone()
            if row and row[0] == "dispatched":
                raise RuntimeError(
                    "request outcome is unknown after interruption; refusing automatic duplicate dispatch"
                )
            self.db.execute("INSERT OR REPLACE INTO inflight VALUES (?,?)", (key, "dispatched"))

    def finish_request(self, key):
        with self.lock, self.db:
            self.db.execute("UPDATE inflight SET status='resolved' WHERE key=?", (key,))

    def accounting_summary(self):
        with self.lock:
            return {
                "reserved_or_failed_calls": self.db.execute(
                    "SELECT COUNT(*) FROM accounting WHERE status='reserved'"
                ).fetchone()[0],
                "conservative_reserved_cost_cny": self.db.execute(
                    "SELECT COALESCE(SUM(cost),0) FROM accounting WHERE status='reserved'"
                ).fetchone()[0],
            }

    def settle(self, rid, amount):
        with self.lock, self.db:
            self.db.execute(
                "UPDATE accounting SET cost=?,status='completed' WHERE id=?", (amount, rid)
            )


SENSITIVE = re.compile(r"api[_-]?key|authorization|secret|password|access[_-]?token", re.I)


def redact(value):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]"
            if SENSITIVE.search(str(k)) and not str(k).endswith("_env")
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        value = re.sub(r"Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value, flags=re.I)
        value = re.sub(
            r"(?i)(api[_-]?key|authorization|password|secret)\s*[:=]\s*[^\s,;]+",
            r"\1=[REDACTED]",
            value,
        )
        return value
    return value


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
