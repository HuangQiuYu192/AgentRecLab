import math
import random
from collections import defaultdict
from pathlib import Path

from utils.core.schema import ProtocolMismatch
from utils.data.dataset import permutation
from utils.experiment.storage import read_json


def at_rank(rank):
    if rank is not None and rank < 1:
        raise ValueError("rank is 1-based")
    result = {f"HR@{k}": float(rank is not None and rank <= k) for k in (1, 3, 5, 10)}
    result.update(
        {
            f"NDCG@{k}": 1 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0
            for k in (3, 10)
        }
    )
    result["MRR"] = 1 / rank if rank else 0.0
    return result


def evaluate_case(case, prediction):
    ids = permutation(prediction.ordered_item_ids, case.candidates)
    return at_rank(ids.index(case.positive_item_id) + 1)


def aggregate(rows):
    success = [r["metrics"] for r in rows if r["status"] == "succeeded"]
    failed = len(rows) - len(success)
    zero = at_rank(None)
    return {
        "n_evaluated": len(success),
        "n_failed": failed,
        "success_rate": len(success) / len(rows) if rows else 0.0,
        "exclude": {
            k: sum(r[k] for r in success) / len(success) if success else None for k in zero
        },
        "count_as_failure": {
            k: sum(r[k] for r in success) / len(rows) if rows else None for k in zero
        },
    }


def user_bootstrap(rows, metric="MRR", n=1000, seed=42):
    groups = defaultdict(list)
    for r in rows:
        groups[r["user_id"]].append(r["metrics"][metric])
    if not groups:
        raise ValueError("no evaluated users")
    rng = random.Random(seed)
    users = list(groups)
    estimates = []
    for _ in range(n):
        sampled = [v for u in rng.choices(users, k=len(users)) for v in groups[u]]
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    return {
        "unit": "user",
        "metric": metric,
        "lower": estimates[int(n * 0.025)],
        "upper": estimates[min(n - 1, int(n * 0.975))],
        "seed": seed,
        "resamples": n,
    }


def compare_runs(runs, paired_only=True):
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    manifests = [read_json(Path(p) / "manifest.json") for p in runs]
    baseline = manifests[0]
    for m in manifests[1:]:
        if m["protocol_mode"] != baseline["protocol_mode"]:
            raise ProtocolMismatch("cannot mix paper_reproduction and controlled_comparison")
        if paired_only and m["cases"] != baseline["cases"]:
            raise ProtocolMismatch(
                "paired comparison requires identical cases, labels, candidate order and histories"
            )
        if m["train_snapshot_sha256"] != baseline["train_snapshot_sha256"]:
            raise ProtocolMismatch("training snapshots differ")
        if m["config"]["provider"] != baseline["config"]["provider"]:
            raise ProtocolMismatch("provider model/snapshot/parameters differ")
    import json

    predictions = [
        {
            r["case_id"]: r
            for r in (
                json.loads(line)
                for line in (Path(p) / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }
        for p in runs
    ]
    common = set.intersection(*(set(p) for p in predictions))
    paired = []
    for cid in sorted(common):
        records = [p[cid] for p in predictions]
        paired.append(
            {
                "case_id": cid,
                "statuses": [r["status"] for r in records],
                "MRR": [r.get("metrics", {}).get("MRR", 0) for r in records],
                "MRR_delta_vs_first": [
                    r.get("metrics", {}).get("MRR", 0) - records[0].get("metrics", {}).get("MRR", 0)
                    for r in records
                ],
            }
        )
    return {
        "protocol_mode": baseline["protocol_mode"],
        "paired_only": paired_only,
        "n_paired": len(paired),
        "runs": [m["run_id"] for m in manifests],
        "metrics": [m["metrics"] for m in manifests],
        "per_case": paired,
    }
