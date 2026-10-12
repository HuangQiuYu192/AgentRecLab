import sqlite3
import threading
from pathlib import Path

from agentreclab.state.snapshot import canonical


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
        from agentreclab.core.errors import BudgetExceeded

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
