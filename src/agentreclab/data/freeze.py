import os
from dataclasses import asdict
from pathlib import Path

from agentreclab.data.manifest import load_dataset
from agentreclab.state.snapshot import digest, file_hash, write_json


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
