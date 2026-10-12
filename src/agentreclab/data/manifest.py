from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from agentreclab.core.schemas import Interaction, RankingCase
from agentreclab.data.validation import validate
from agentreclab.state.snapshot import digest, file_hash, read_json, write_json

DOMAINS = ("books", "movietv", "goodreads", "yelp")


def save_dataset(out, cases, interactions, catalog, provenance=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stats = validate(
        cases, interactions, catalog, strict_time=bool((provenance or {}).get("strict_time", True))
    )
    tables = {
        "cases": [asdict(c) for c in cases],
        "interactions": [asdict(e) for e in interactions],
        "catalog": [{"item_id": k, **v} for k, v in catalog.items()],
    }
    # Metadata is serialized as JSON to avoid heterogeneous nested Parquet schemas.
    from agentreclab.state.snapshot import canonical

    for name, rows in tables.items():
        for row in rows:
            if "metadata" in row:
                row["metadata"] = canonical(row["metadata"])
        pq.write_table(pa.Table.from_pylist(rows), out / f"{name}.parquet")
    manifest = {
        "version": 1,
        "converter_version": "0.1.0",
        "dataset_id": cases[0].dataset_id,
        "files": {
            n: {"path": f"{n}.parquet", "sha256": file_hash(out / f"{n}.parquet")} for n in tables
        },
        "case_ids_sha256": digest([c.case_id for c in cases]),
        "candidate_sha256": digest([asdict(c) for c in cases]),
        "stats": stats,
        "provenance": provenance or {},
    }
    write_json(out / "dataset_manifest.json", manifest)
    return out / "dataset_manifest.json"


def load_dataset(path):
    import json

    path = Path(path)
    manifest = read_json(path)
    if "dataset_manifest" in manifest:
        source = (path.parent / manifest["dataset_manifest"]).resolve()
        if file_hash(source) != manifest["dataset_manifest_sha256"]:
            raise ValueError("dataset manifest checksum mismatch")
        cases, interactions, catalog, original = load_dataset(source)
        by_id = {c.case_id: c for c in cases}
        cases = tuple(by_id[i] for i in manifest["case_ids"])
        if digest([asdict(c) for c in cases]) != manifest["cases_sha256"]:
            raise ValueError("frozen cases checksum mismatch")
        return cases, interactions, catalog, {**original, "frozen": manifest}
    rows = {}
    for name, info in manifest["files"].items():
        target = path.parent / info["path"]
        if file_hash(target) != info["sha256"]:
            raise ValueError(f"{name} checksum mismatch")
        rows[name] = pq.read_table(target).to_pylist()
        for row in rows[name]:
            if "metadata" in row:
                row["metadata"] = json.loads(row["metadata"])
    cases = tuple(
        RankingCase(
            **{
                **r,
                "history_event_ids": tuple(r["history_event_ids"]),
                "candidates": tuple(r["candidates"]),
            }
        )
        for r in rows["cases"]
    )
    events = tuple(Interaction(**r) for r in rows["interactions"])
    catalog = {
        r["item_id"]: {k: v for k, v in r.items() if k != "item_id"} for r in rows["catalog"]
    }
    validate(cases, events, catalog, strict_time=manifest["provenance"].get("strict_time", True))
    return cases, events, catalog, manifest
