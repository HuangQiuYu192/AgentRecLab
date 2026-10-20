"""Build four disjoint, frozen 100-user Books subsets from completed baselines. No API calls."""

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.core.config import ROOT
from utils.data.dataset import load_dataset, save_dataset
from utils.evaluation.metrics import aggregate
from utils.experiment.storage import canonical, digest, file_hash, read_json, write_json

NAMES = ("books100_eval", "books100_hard", "books100_disagreement", "books100_retention")


def baseline_rows(directory, source_manifest):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    if manifest["status"] not in ("completed", "completed_with_failures"):
        raise ValueError(f"baseline not complete: {directory}")
    if manifest["dataset_manifest_sha256"] != file_hash(source_manifest):
        raise ValueError("baseline dataset does not match source")
    with closing(
        sqlite3.connect(f"file:{(directory / 'state.sqlite').as_posix()}?mode=ro", uri=True)
    ) as db:
        return {
            cid: {"case_id": cid, "status": status, **(json.loads(result) or {})}
            for cid, status, result in db.execute("SELECT id,status,result FROM cases")
        }


def build(source, iagent_run, i2agent_run, out_root, seed=42):
    source, out_root = Path(source), Path(out_root)
    if any((out_root / name).exists() for name in NAMES):
        raise ValueError("subset directory already exists; frozen subsets must not be overwritten")
    summary_path = out_root / "books100_partition_summary.json"
    if summary_path.exists():
        raise ValueError("partition summary already exists")
    cases, events, catalog, original = load_dataset(source)
    a, b = baseline_rows(iagent_run, source), baseline_rows(i2agent_run, source)
    if set(a) != set(b) or set(a) != {c.case_id for c in cases}:
        raise ValueError("baseline case sets do not match full dataset")
    if len({c.user_id for c in cases}) != len(cases):
        raise ValueError("expected one frozen case per real user")
    eligible = [c for c in cases if a[c.case_id]["status"] == b[c.case_id]["status"] == "succeeded"]
    lengths = sorted(len(c.history_event_ids) for c in eligible)
    q1, q2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]

    def tier(case):
        n = len(case.history_event_ids)
        return "short" if n <= q1 else "medium" if n <= q2 else "long"

    def category(case):
        x, y = (rows[case.case_id]["metrics"]["HR@1"] == 1 for rows in (a, b))
        return (
            "both_hit"
            if x and y
            else "iagent_only"
            if x
            else "i2agent_only"
            if y
            else "neither_hit"
        )

    def sample(pool, size, salt):
        if len(pool) < size:
            raise ValueError(f"not enough users in {salt}")
        groups = {t: [c for c in pool if tier(c) == t] for t in ("short", "medium", "long")}
        quotas = {t: size * len(group) // len(pool) for t, group in groups.items()}
        order = sorted(groups, key=lambda t: (-(size * len(groups[t]) % len(pool)), t))
        for t in order[: size - sum(quotas.values())]:
            quotas[t] += 1
        rng = random.Random(int(digest({"seed": seed, "group": salt})[:16], 16))
        picked = {c.case_id for t, group in groups.items() for c in rng.sample(group, quotas[t])}
        return [c for c in cases if c.case_id in picked]

    selected = {NAMES[0]: sample(eligible, 100, "representative")}
    used = {c.user_id for c in selected[NAMES[0]]}

    def remaining(label):
        return [c for c in eligible if c.user_id not in used and category(c) == label]

    selected[NAMES[1]] = sample(remaining("neither_hit"), 100, "hard")
    used.update(c.user_id for c in selected[NAMES[1]])
    disagreement = sample(remaining("iagent_only"), 50, "disagreement_iagent")
    disagreement += sample(remaining("i2agent_only"), 50, "disagreement_i2agent")
    ids = {c.case_id for c in disagreement}
    selected[NAMES[2]] = [c for c in cases if c.case_id in ids]
    used.update(c.user_id for c in selected[NAMES[2]])
    selected[NAMES[3]] = sample(remaining("both_hit"), 100, "retention")
    assert len({c.user_id for group in selected.values() for c in group}) == 400
    assert all(len(group) == 100 for group in selected.values())
    for rows in (a, b):
        for case in eligible:
            assert rows[case.case_id]["user_id"] == case.user_id
    origins = {
        name: {
            "manifest_sha256": file_hash(Path(path) / "manifest.json"),
            "predictions_sha256": file_hash(Path(path) / "predictions.jsonl"),
        }
        for name, path in (("iagent", iagent_run), ("i2agent", i2agent_run))
    }
    report = {
        "seed": seed,
        "source_manifest_sha256": file_hash(source),
        "source_cases": len(cases),
        "eligible_users": len(eligible),
        "excluded_case_ids": [c.case_id for c in cases if c not in eligible],
        "history_length_cutoffs": {"short_max": q1, "medium_max": q2},
        "sampling": "largest-remainder proportional history strata; eval first; other sets exclude selected users",
        "baseline_sources": origins,
        "subsets": {},
        "interpretation": "Outcome-selected sets are diagnostic; repeated tuning makes these development sets.",
    }
    events_by_id = {e.event_id: e for e in events}
    for name, group in selected.items():
        target = out_root / name
        event_ids = {eid for c in group for eid in c.history_event_ids}
        item_ids = {events_by_id[eid].item_id for eid in event_ids}
        item_ids.update(item for c in group for item in c.candidates)
        item_ids.update(c.metadata["warmup_negative"] for c in group)
        subset_events = [e for e in events if e.event_id in event_ids]
        subset_catalog = {i: value for i, value in catalog.items() if i in item_ids}
        provenance = {
            **original["provenance"],
            "subset_source": {"path": str(source), "sha256": file_hash(source)},
            "seed": seed,
            "max_users": 100,
            "selection_case_ids": [c.case_id for c in group],
            "filtering": name,
            "catalog_scope": "selected histories, original candidates and frozen warmup negatives",
            "baseline_sources": origins,
            "history_length_cutoffs": report["history_length_cutoffs"],
            "purpose": name.removeprefix("books100_"),
            "collaborative_graph": "selected users only; not the full Books collaborative graph",
        }
        manifest = save_dataset(target, group, subset_events, subset_catalog, provenance)
        reread, reread_events, reread_catalog, _ = load_dataset(manifest)
        assert [asdict(c) for c in reread] == [asdict(c) for c in group]
        assert [asdict(e) for e in reread_events] == [asdict(e) for e in subset_events]
        assert reread_catalog == subset_catalog
        baseline_path = target / "baselines.jsonl"
        baseline_path.write_text(
            "".join(
                canonical(
                    {
                        "case_id": c.case_id,
                        "user_id": c.user_id,
                        "group": category(c),
                        "history_length": len(c.history_event_ids),
                        "history_stratum": tier(c),
                        "iagent": a[c.case_id],
                        "i2agent": b[c.case_id],
                    }
                )
                + "\n"
                for c in group
            ),
            encoding="utf-8",
        )
        info = {
            "n_users": 100,
            "n_interactions": len(subset_events),
            "n_items": len(subset_catalog),
            "groups": dict(Counter(category(c) for c in group)),
            "history_strata": dict(Counter(tier(c) for c in group)),
            "case_ids": [c.case_id for c in group],
            "manifest_sha256": file_hash(manifest),
            "baselines_sha256": file_hash(baseline_path),
            "baseline_metrics": {
                label: aggregate([rows[c.case_id] for c in group])
                for label, rows in (("iagent", a), ("i2agent", b))
            },
        }
        write_json(target / "subset_summary.json", info)
        (target / "README.md").write_text(
            f"# {name}\n\nFrozen independent 100-user subset; seed={seed}.\n\n"
            "Run using dataset_manifest.json. No runtime dependency on the full dataset.\n"
            "baselines.jsonl contains evaluation-only labels and existing predictions; never feed it to a model.\n"
            "Original case text, candidate order, target and warmup negative are preserved.\n"
            "Collaborative graph contains these users only. Diagnostic sets do not estimate full-population performance.\n"
            "Keep untouched users for final evaluation after tuning.\n",
            encoding="utf-8",
        )
        report["subsets"][name] = info
    report["overlap_users"] = {
        x: {
            y: len({c.user_id for c in selected[x]} & {c.user_id for c in selected[y]})
            for y in NAMES
        }
        for x in NAMES
    }
    write_json(summary_path, report)
    return {
        "summary": str(summary_path),
        "history_length_cutoffs": report["history_length_cutoffs"],
        "subsets": {
            name: {k: v for k, v in info.items() if k not in ("case_ids", "baseline_metrics")}
            for name, info in report["subsets"].items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "datasets/processed/books/dataset_manifest.json"
    )
    parser.add_argument(
        "--iagent-run",
        type=Path,
        default=ROOT / "outputs/runs/books_deepseek41flash_iagent_upstream",
    )
    parser.add_argument(
        "--i2agent-run",
        type=Path,
        default=ROOT / "outputs/runs/books_deepseek41flash_i2agent_upstream",
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "datasets/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.iagent_run, args.i2agent_run, args.out_root, args.seed),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
