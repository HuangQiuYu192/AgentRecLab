"""Audited reuse of legacy responses across the JSON-object -> strict-schema change."""

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

from utils.core.config import ROOT
from utils.experiment.llm import validate_schema
from utils.experiment.storage import (
    StateStore,
    canonical,
    digest,
    file_hash,
    read_json,
    redact,
    write_json,
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def compatible_providers(old, new):
    if old == new:
        return False
    expected = json.loads(canonical(old))
    caps = expected.setdefault("capabilities", {})
    if caps.get("supports_json_schema", False) or not new.get("capabilities", {}).get(
        "supports_json_schema"
    ):
        raise ValueError("only the JSON-object to strict-schema cache upgrade is allowed")
    caps["supports_json_schema"] = True
    if expected != new:
        raise ValueError("response cache provider mismatch beyond JSON Schema capability")
    return True


def import_response_cache(source, store, config):
    source = resolve(source)
    provider = redact(config["provider"])
    dataset_hash = file_hash(config["manifest"])
    bundle = source / "cache_manifest.json"
    if bundle.exists():
        info = read_json(bundle)
        if (
            info["target_provider"] != provider
            or info["dataset_manifest_sha256"] != dataset_hash
            or info["target_model"] != config["model"]
        ):
            raise ValueError("prepared response cache provider/dataset mismatch")
        if info["database_sha256"] != file_hash(source / "state.sqlite"):
            raise ValueError("prepared response cache checksum mismatch")
        with closing(
            sqlite3.connect(f"file:{(source / 'state.sqlite').as_posix()}?mode=ro", uri=True)
        ) as old:
            responses = old.execute("SELECT key,response FROM responses").fetchall()
            aliases = old.execute(
                "SELECT key,source_key,provenance FROM response_aliases"
            ).fetchall()
            unresolved = old.execute(
                "SELECT key,status FROM inflight WHERE status='dispatched'"
            ).fetchall()
        with store.db:
            store.db.executemany("INSERT OR IGNORE INTO responses VALUES (?,?)", responses)
            store.db.executemany("INSERT OR IGNORE INTO response_aliases VALUES (?,?,?)", aliases)
            store.db.executemany("INSERT OR IGNORE INTO inflight VALUES (?,?)", unresolved)
        return {
            **info["report"],
            "prepared_cache": str(source),
            "cache_manifest_sha256": file_hash(bundle),
        }

    previous = read_json(source / "manifest.json")
    if previous["dataset_manifest_sha256"] != dataset_hash:
        raise ValueError("response cache dataset mismatch")
    if previous["config"]["model"] != config["model"]:
        raise ValueError("response cache model mismatch")
    upgraded = compatible_providers(previous["config"]["provider"], provider)
    with closing(
        sqlite3.connect(f"file:{(source / 'state.sqlite').as_posix()}?mode=ro", uri=True)
    ) as old:
        responses = old.execute("SELECT key,response FROM responses").fetchall()
        unresolved_keys = [
            row[0] for row in old.execute("SELECT key FROM inflight WHERE status='dispatched'")
        ]
    wanted = {key for key, _ in responses} | set(unresolved_keys)
    requests = {}
    with (source / "traces.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("event") == "dispatch" and row.get("cache_key") in wanted:
                request = row["request"]
                if digest(request) != row["cache_key"]:
                    raise ValueError("source request hash mismatch")
                requests[row["cache_key"]] = request
    valid, aliases, excluded, reasons = [], [], [], Counter()
    for key, response in responses:
        try:
            request = requests[key]
            value = json.loads(json.loads(response)["text"])
            schema = request["parameters"]["schema"]
            validate_schema(value, schema)
        except (KeyError, ValueError, TypeError) as exc:
            excluded.append(
                {"source_key": key, "reason": str(exc), "error_type": type(exc).__name__}
            )
            reasons[type(exc).__name__] += 1
            continue
        valid.append((key, response))
        if upgraded:
            target_key = digest({**request, "provider": provider})
            aliases.append(
                (
                    target_key,
                    key,
                    canonical(
                        {
                            "kind": "legacy_json_object_validated",
                            "source_run": str(source),
                            "source_key": key,
                            "schema_sha256": digest(schema),
                            "source_response_sha256": digest(json.loads(response)),
                            "limitation": "Legacy output reused after local validation; not generated with strict schema.",
                        }
                    ),
                )
            )
    unresolved = []
    for key in unresolved_keys:
        if key not in requests:
            raise ValueError("cannot safely remap an unresolved request without its source trace")
        unresolved.append((key, "dispatched"))
        if upgraded:
            unresolved.append((digest({**requests[key], "provider": provider}), "dispatched"))
    with store.db:
        store.db.executemany("INSERT OR IGNORE INTO responses VALUES (?,?)", valid)
        store.db.executemany("INSERT OR IGNORE INTO response_aliases VALUES (?,?,?)", aliases)
        store.db.executemany("INSERT OR IGNORE INTO inflight VALUES (?,?)", unresolved)
    return {
        "source_run": str(source),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "source_trace_sha256": file_hash(source / "traces.jsonl"),
        "schema_upgrade": upgraded,
        "responses": len(valid),
        "aliases": len(aliases),
        "excluded_responses": len(excluded),
        "excluded_reasons": dict(reasons),
        "excluded": excluded,
        "unresolved_requests": len(unresolved_keys),
        "accounting": "new requests only; prior charges remain in the source run",
        "protocol": "legacy validated responses plus new strict-schema responses"
        if upgraded
        else "exact-request reuse",
    }


def prepare_response_cache(source, out, config):
    out = resolve(out)
    if out.exists():
        raise ValueError(
            "cache output already exists; use a new directory to preserve the snapshot"
        )
    store = StateStore(out / "state.sqlite")
    try:
        report = import_response_cache(source, store, config)
        integrity = store.db.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise ValueError(f"cache integrity check failed: {integrity}")
    finally:
        store.close()
    write_json(
        out / "cache_manifest.json",
        {
            "version": 1,
            "target_model": config["model"],
            "target_provider": redact(config["provider"]),
            "dataset_manifest_sha256": file_hash(config["manifest"]),
            "database_sha256": file_hash(out / "state.sqlite"),
            "report": report,
        },
    )
    return {"prepared_cache": str(out), **{k: v for k, v in report.items() if k != "excluded"}}
