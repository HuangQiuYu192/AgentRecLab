from pathlib import Path

from agentreclab.core.errors import ProtocolMismatch
from agentreclab.state.snapshot import read_json


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
