import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils.schema import Interaction, RankingCase, ValidationError
from utils.storage import digest, file_hash, read_json, write_json


def instant(text):
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValidationError("timestamps require a timezone")
    return value.astimezone(timezone.utc)


def validate(cases, interactions, catalog, candidate_count=10, strict_time=True):
    seen = set()
    events = {e.event_id: e for e in interactions}
    if len(events) != len(interactions):
        raise ValidationError("duplicate event_id")
    for e in interactions:
        if e.item_id not in catalog:
            raise ValidationError(f"missing catalog item {e.item_id}")
    for c in cases:
        if c.case_id in seen:
            raise ValidationError(f"duplicate case_id {c.case_id}")
        seen.add(c.case_id)
        if len(c.candidates) != candidate_count or len(set(c.candidates)) != candidate_count:
            raise ValidationError(f"{c.case_id}: candidate count/duplicates")
        if c.candidates.count(c.positive_item_id) != 1:
            raise ValidationError(f"{c.case_id}: positive must appear exactly once")
        if any(i not in catalog for i in c.candidates):
            raise ValidationError(f"{c.case_id}: missing catalog item")
        if strict_time and c.evaluation_time is None:
            raise ValidationError("strict time requires evaluation_time")
        for eid in c.history_event_ids:
            if eid not in events:
                raise ValidationError(f"unknown history event {eid}")
            e = events[eid]
            if e.user_id != c.user_id:
                raise ValidationError("history belongs to another user")
            if eid == c.metadata.get("target_event_id"):
                raise ValidationError("target event leaked into history")
            if strict_time and (
                e.timestamp is None or instant(e.timestamp) >= instant(c.evaluation_time)
            ):
                raise ValidationError("history is not strictly before evaluation")
        # Collaborative models can see every user's train graph, not just local history.
        if strict_time and any(
            e.timestamp is None or instant(e.timestamp) >= instant(c.evaluation_time)
            for e in interactions
        ):
            raise ValidationError("global training graph contains future interactions")
    return {
        "n_cases": len(cases),
        "n_interactions": len(events),
        "n_items": len(catalog),
        "n_users": len({e.user_id for e in interactions}),
        "n_test_users": len({c.user_id for c in cases}),
    }


def permutation(ids, candidates, policy="fail"):
    ids = tuple(str(i) for i in ids)
    candidates = tuple(str(i) for i in candidates)
    if len(ids) == len(candidates) and set(ids) == set(candidates):
        return ids
    if policy == "append_missing":
        if any(i not in candidates for i in ids) or len(ids) != len(set(ids)):
            raise ValidationError("append_missing cannot repair duplicate/unknown IDs")
        return ids + tuple(i for i in candidates if i not in ids)
    raise ValidationError("prediction is not a strict candidate permutation")


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
    from utils.storage import canonical

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


def freeze(source, out, seed=42, limit=None, candidates=10):
    cases, _, _, manifest = load_dataset(source)
    if any(len(c.candidates) != candidates for c in cases):
        raise ValueError("freeze cannot resample existing candidates")
    chosen = cases if limit is None else cases[:limit]
    if not chosen:
        raise ValueError("empty case selection")
    write_json(
        out,
        {
            "version": 1,
            "dataset_id": manifest["dataset_id"],
            "dataset_manifest": os.path.relpath(Path(source).resolve(), Path(out).resolve().parent),
            "dataset_manifest_sha256": file_hash(source),
            "seed": seed,
            "selection": "original_order_prefix",
            "candidate_source": manifest["provenance"].get("candidate_source", "original"),
            "case_ids": [c.case_id for c in chosen],
            "cases_sha256": digest([asdict(c) for c in chosen]),
        },
    )
    return Path(out)


def convert(dataset, config):
    """Read trusted original pickle; never silently resample test candidates."""
    import pandas as pd

    if dataset not in DOMAINS:
        raise ValueError(f"unsupported domain {dataset}")
    if not config.get("trusted_pickle"):
        raise ValueError(
            "pickle can execute code: set trusted_pickle only for trusted upstream data"
        )
    raw, mapping = Path(config["pickle"]), Path(config["mapping"])
    df, meta = pd.read_pickle(raw), pd.read_csv(mapping, dtype=str, keep_default_na=False)
    id_column = config.get("item_id_column", "index")
    catalog = {
        str(r[id_column]): {
            "title": str(r["title"]),
            "description": str(r.get("description", "")),
            "tags": [],
        }
        for _, r in meta.iterrows()
    }
    cases, events = [], []
    for ordinal, (index, row) in enumerate(df.iterrows()):
        namespace = f"instructrec:{dataset}"
        uid = (
            f"{namespace}:{row[config['user_id_column']]}"
            if config.get("user_id_column")
            else f"{namespace}:row:{ordinal}"
        )
        items = [str(i) for i in row["asin"]]
        if len(items) < 3:
            raise ValueError(f"row {ordinal} requires at least 3 interactions")
        history = []
        for pos, item in enumerate(items[:-1]):
            eid = f"{namespace}:{ordinal}:{pos}"
            history.append(eid)
            events.append(
                Interaction(
                    eid,
                    uid,
                    item,
                    None,
                    str(row["reviewText"][pos]),
                    metadata={"position": pos, "row_index": str(index)},
                )
            )
        if "ranked_lists" not in row:
            raise ValueError("original ranked_lists absent; explicit controlled sampling required")
        candidates = tuple(str(i) for i in row["ranked_lists"])
        negative_pool = [i for i in catalog if i not in items and i not in candidates]
        if not negative_pool:
            raise ValueError("no train warmup negative available")
        import random

        negative = random.Random(config.get("seed", 42) + ordinal).choice(negative_pool)
        cases.append(
            RankingCase(
                f"{namespace}:{ordinal}",
                dataset,
                uid,
                str(row["instruction"]),
                tuple(history),
                candidates,
                items[-1],
                metadata={
                    "original_row_index": str(index),
                    "row_ordinal": ordinal,
                    "target_event_id": f"{namespace}:{ordinal}:{len(items) - 1}",
                    "warmup_negative": negative,
                    "user_identity": "original"
                    if config.get("user_id_column")
                    else "row_namespace_original_id_unknown",
                },
            )
        )
    return save_dataset(
        config["out"],
        cases,
        events,
        catalog,
        {
            "raw_sources": {str(p): file_hash(p) for p in (raw, mapping)},
            "strict_time": False,
            "time_semantics": "sequence_only_original_timestamps_unknown",
            "candidate_source": "original",
            "seed": config.get("seed", 42),
            "filtering": "none",
            "user_id_column": config.get("user_id_column", "unknown"),
        },
    )


def make_fixture(out, n=10, dataset="books"):
    if n < 1:
        raise ValueError("fixture requires at least one case")

    if dataset not in DOMAINS:
        raise ValueError("unsupported fixture domain")
    catalog = {
        str(i): {
            "title": f"Synthetic book {i}",
            "description": f"Fictional adventure {i}",
            "tags": ["fiction"],
        }
        for i in range(1, 61)
    }
    cases, events = [], []
    for j in range(n):
        uid = f"fixture:{dataset}:user:{j}"
        history = []
        for k in range(4):
            eid = f"{uid}:event:{k}"
            history.append(eid)
            events.append(
                Interaction(
                    eid,
                    uid,
                    str(20 + (j + k) % 20),
                    f"2020-01-0{k + 1}T00:00:00Z",
                    "Synthetic training review",
                )
            )
        candidates = tuple(str(1 + (j + k) % 15) for k in range(10))
        cases.append(
            RankingCase(
                f"fixture:{dataset}:case:{j}",
                dataset,
                uid,
                "Recommend a fictional adventure.",
                tuple(history),
                candidates,
                candidates[j % 10],
                "2020-02-01T00:00:00Z",
                {"target_event_id": f"{uid}:test", "warmup_negative": "60"},
            )
        )
    return save_dataset(
        out,
        cases,
        events,
        catalog,
        {
            "synthetic": True,
            "candidate_source": "synthetic_fixture",
            "strict_time": True,
            "seed": 42,
        },
    )
