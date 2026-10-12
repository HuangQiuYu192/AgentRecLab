"""Reproducible 10/100-case mock matrix; no API keys and no private data."""

import argparse
import asyncio
from pathlib import Path

from agentreclab.config import ROOT, load_run
from agentreclab.data import make_fixture
from agentreclab.evaluation import compare_runs
from agentreclab.runner import run_experiment
from agentreclab.storage import file_hash, read_json, write_json


async def execute(prefix):
    results = []
    for n in (10, 100):
        fixture = make_fixture(ROOT / "outputs" / "fixtures" / f"books{n}", n)
        runs = []
        for model in ("iagent", "i2agent", "memrec"):
            cfg = load_run(ROOT / "configs/runs/smoke_books.yaml", model=model)
            cfg.update(
                manifest=str(fixture), run_id=f"{prefix}_{n}_{model}", max_calls=2000, concurrency=4
            )
            result = await run_experiment(cfg)
            if result["status"] != "completed" or result["metrics"]["n_evaluated"] != n:
                raise RuntimeError(result)
            directory = Path(result["run"])
            before = file_hash(directory / "traces.jsonl")
            replayed = await run_experiment(cfg, resume=True)
            if replayed["metrics"] != result["metrics"] or before != file_hash(
                directory / "traces.jsonl"
            ):
                raise RuntimeError("completed-case resume dispatched requests or changed metrics")
            manifest = read_json(directory / "manifest.json")
            results.append(
                {
                    "run_id": cfg["run_id"],
                    "n_cases": n,
                    "model": model,
                    "status": result["status"],
                    "metrics": result["metrics"],
                    "source_sha256": manifest["source_sha256"],
                    "manifest_sha256": file_hash(directory / "manifest.json"),
                    "prepared_state_sha256": manifest["state_snapshots"]["prepared"],
                    "resume_no_new_requests": True,
                }
            )
            runs.append(result["run"])
            print(
                f"{n} / {model}: completed, {result['metrics']['total_calls']} mock calls, resume unchanged",
                flush=True,
            )
        paired = compare_runs(runs)
        write_json(ROOT / "outputs" / "reports" / f"{prefix}_{n}_paired.json", paired)
        if paired["n_paired"] != n:
            raise RuntimeError("paired case coverage mismatch")
    output = ROOT / "outputs" / "reports" / f"{prefix}_acceptance.json"
    write_json(output, {"synthetic_only": True, "paid_api_calls": 0, "results": results})
    print(f"ALL_DONE: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="acceptance")
    asyncio.run(execute(parser.parse_args().prefix))
