import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from utils.core.schema import Interaction, RankingCase, ValidationError
from utils.experiment.storage import digest, file_hash, read_json, write_json


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
    from utils.experiment.storage import canonical

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
    df = pd.read_pickle(raw)
    df["__source_ordinal"] = range(len(df))
    user_column = config.get("user_id_column")
    limit = config.get("max_users")
    if limit is not None:
        import random

        if not user_column or user_column not in df:
            raise ValueError("sampling users requires an original user_id_column")
        users = list(df[user_column].drop_duplicates())
        if int(limit) < 1 or int(limit) > len(users):
            raise ValueError("max_users is outside the available user count")
        selected = random.Random(config.get("seed", 42)).sample(users, int(limit))
        df = df[df[user_column].isin(selected)]
        needed = {str(i) for values in df["asin"] for i in values}
        needed.update(str(i) for values in df["ranked_lists"] for i in values)
    else:
        needed = None
    id_column = config.get("item_id_column", "index")
    catalog, spare = {}, set()
    for chunk in pd.read_csv(mapping, dtype=str, keep_default_na=False, chunksize=10000):
        if needed is not None:
            for item in chunk[id_column]:
                if item not in needed and len(spare) < 100:
                    spare.add(item)
            chunk = chunk[chunk[id_column].isin(needed | spare)]
        for _, r in chunk.iterrows():
            catalog[str(r[id_column])] = {
                "title": str(r["title"]),
                "description": str(r.get("description", "")),
                "tags": [],
            }
    cases, events = [], []
    for index, row in df.iterrows():
        ordinal = int(row["__source_ordinal"])
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
            "filtering": "seeded_user_sample_original_row_order" if limit is not None else "none",
            "max_users": limit,
            "catalog_scope": "selected histories/candidates plus 100 spare items"
            if limit is not None
            else "full_original_mapping",
            "user_id_column": config.get("user_id_column", "unknown"),
        },
    )
