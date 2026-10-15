import asyncio
import platform
import re
import sqlite3
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from models import create_model
from utils.core.config import ROOT
from utils.core.schema import ExperimentContext
from utils.data.dataset import load_dataset, permutation
from utils.evaluation.metrics import aggregate, evaluate_case
from utils.experiment.batch import BatchClient, BatchPending, validate_batch
from utils.experiment.cache import import_response_cache
from utils.experiment.llm import TracedProvider
from utils.experiment.llm import create_provider as create
from utils.experiment.storage import (
    Recorder,
    StateStore,
    canonical,
    digest,
    file_hash,
    read_json,
    redact,
    write_json,
)


def git_info():
    try:
        command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "-C", str(ROOT)]
        return {
            "commit": subprocess.check_output(command + ["rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(
                subprocess.check_output(command + ["status", "--porcelain"], text=True).strip()
            ),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": True}


def source_version():
    files = [
        ROOT / "main.py",
        *sorted((ROOT / "models").glob("*.py")),
        *sorted((ROOT / "utils").rglob("*.py")),
    ]
    return digest({str(p.relative_to(ROOT)): file_hash(p) for p in files})


def environment_versions():
    from importlib.metadata import version

    return {
        name: version(name)
        for name in (
            "agentreclab",
            "numpy",
            "pandas",
            "pyarrow",
            "httpx",
            "typer",
            "PyYAML",
        )
    }


def readiness(config):
    if config["protocol_mode"] == "paper_reproduction":
        raise ValueError(
            "paper_reproduction not_ready: native models require protocol and real-data validation"
        )
    return (
        "synthetic_mock" if config["provider"]["name"] == "mock" else "native_controlled_comparison"
    )


def planned_calls(model, cases):
    return len(cases) * {"iagent": 2, "i2agent": 5, "memrec": 2}[model] + (
        3 * len({c.user_id for c in cases}) if model == "memrec" else 0
    )


async def run_experiment(config, resume=False, dry_run=False, allow_paid=False):
    validate_batch(config)
    use_batch = config.get("use_batch", False)
    model = create_model(config["model"], use_batch=use_batch)
    cases, interactions, catalog, data_manifest = load_dataset(config["manifest"])
    if not cases:
        raise ValueError("empty case manifest")
    status = readiness(config)
    from utils.experiment.llm import price

    calls = planned_calls(config["model"], cases)
    text_bytes = len(canonical(catalog).encode()) + max(len(c.instruction.encode()) for c in cases)
    output_cap = config["provider"].get("max_tokens")
    if output_cap is None and config.get("budget_cny") is not None:
        raise ValueError("a cost budget requires an explicit output token cap")
    estimated_input = (
        calls * (text_bytes + output_cap * 4 + 16384) if output_cap is not None else None
    )
    estimated_output = calls * output_cap if output_cap is not None else None
    pricing_config = config["provider"]
    if use_batch:
        pricing_config = {**pricing_config, "pricing": config.get("batch", {}).get("pricing")}
    estimated_cost = None
    if output_cap is not None:
        estimated_cost, _ = price(pricing_config, None, estimated_input, estimated_output)
    estimates = {
        "n_cases": len(cases),
        "planned_calls_without_retries": planned_calls(config["model"], cases),
        "max_agent_attempts_with_repairs": (
            3 * (calls + 3 * len(cases)) if config["model"] in ("iagent", "i2agent") else None
        ),
        "max_calls": config.get("max_calls", 1000),
        "budget_cny": config.get("budget_cny"),
        "use_batch": use_batch,
        "provider": config["provider"]["name"],
        "model": config["model"],
        "paid_calls": False,
        "estimated_input_token_bound": estimated_input,
        "estimated_output_token_cap": estimated_output,
        "estimated_cost_cny": estimated_cost,
        "estimate_method": (
            "rough byte bound including catalog/context allowance; before retries/cache, not a bill"
            if output_cap is not None
            else "output limit omitted; provider default unknown; token and cost bounds unavailable"
        ),
    }
    if dry_run:
        return estimates
    if config["provider"]["name"] != "mock" and not allow_paid:
        raise ValueError("real provider requires explicit --allow-paid after reviewing --dry-run")
    rid = config["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", rid):
        raise ValueError("run_id may contain only letters, numbers, underscores and hyphens")
    directory = Path(config.get("outputs", ROOT / "outputs" / "runs")) / rid
    directory.mkdir(parents=True, exist_ok=True)
    # Atomic cross-process lock; released by OS after a crash (SQLite exclusive transaction).
    run_lock = sqlite3.connect(directory / "run_lock.sqlite", timeout=0)
    try:
        run_lock.execute("BEGIN EXCLUSIVE")
    except sqlite3.OperationalError as exc:
        run_lock.close()
        raise RuntimeError("another process owns this run_id") from exc
    safe_config = redact(config)
    fingerprint = digest(
        {
            "config": safe_config,
            "dataset": file_hash(config["manifest"]),
            "source": source_version(),
            "environment": environment_versions(),
            "python": platform.python_version(),
            "lock": file_hash(ROOT / "requirements.lock")
            if (ROOT / "requirements.lock").exists()
            else "unknown",
        }
    )
    manifest_path = directory / "manifest.json"
    store = None
    try:
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if not resume:
                raise ValueError("run_id exists; use --resume or a new run_id")
            if manifest["fingerprint"] != fingerprint:
                raise ValueError("resume fingerprint mismatch: config/data/code changed")
        else:
            manifest = {
                "run_id": rid,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "config": safe_config,
                "git": git_info(),
                "python": platform.python_version(),
                "lock_sha256": file_hash(ROOT / "requirements.lock")
                if (ROOT / "requirements.lock").exists()
                else "unknown",
                "source_sha256": source_version(),
                "package_versions": environment_versions(),
                "implementation": "native-v2",
                "dataset_manifest_sha256": file_hash(config["manifest"]),
                "dataset_provenance": data_manifest["provenance"],
                "train_snapshot_sha256": digest([asdict(e) for e in interactions]),
                "case_ids": [c.case_id for c in cases],
                "cases": [asdict(c) for c in cases],
                "protocol_mode": config["protocol_mode"],
                "reproduction_status": status,
                "prompt_template_sha256": digest(
                    {
                        str(p.relative_to(ROOT)): file_hash(p)
                        for p in sorted((ROOT / "models").glob("*.py"))
                    }
                ),
                "deviations": [
                    "native implementations; no runtime upstream code",
                    "MemRec uses concise native prompts; not a strict paper reproduction",
                    "test labels never passed in LLM prompts",
                    (
                        "upstream compatibility: label-dependent target-presence repair; "
                        "incomplete/duplicate lists accepted when target is present"
                        if config.get("invalid_prediction_policy") == "upstream"
                        else "label-free complete-candidate validation and repair"
                    ),
                    (
                        "provider JSON-object fallback plus local schema validation"
                        if not config["provider"]
                        .get("capabilities", {})
                        .get("supports_json_schema")
                        else "provider strict JSON schema"
                    ),
                    "original candidate order frozen",
                    "no test feedback propagation",
                    "training graph temporal cutoff",
                    "provider parameters explicitly resolved",
                    "deterministic frozen warmup sampling",
                ],
                "status": "running",
                "state_snapshots": {},
            }
            write_json(manifest_path, manifest)
            write_json(directory / "config.json", safe_config)
        store = StateStore(directory / "state.sqlite")
        if config.get("reuse_responses_from") and "cache_import" not in manifest:
            manifest["cache_import"] = import_response_cache(
                config["reuse_responses_from"], store, config
            )
            write_json(manifest_path, manifest)
        recorder = Recorder(directory)
        provider = TracedProvider(
            create(config["provider"]),
            config["provider"],
            store,
            recorder,
            config.get("max_calls", 1000),
            config.get("budget_cny"),
        )
        context = ExperimentContext(
            catalog,
            {e.event_id: e for e in interactions},
            cases,
            provider,
            store,
            recorder,
            config["protocol_mode"],
            config,
        )
        model.prepare(context)
        snapshot = directory / "state_prepared.json"
        if snapshot.exists() and manifest["state_snapshots"].get("prepared"):
            if file_hash(snapshot) != manifest["state_snapshots"]["prepared"]:
                raise ValueError("prepared state snapshot checksum mismatch")
            model.load_state(snapshot)
        else:
            if hasattr(model, "warmup"):
                await model.warmup()
            model.save_state(snapshot)
            manifest["state_snapshots"]["prepared"] = file_hash(snapshot)
            write_json(manifest_path, manifest)
        completed = {r["case_id"] for r in store.rows() if r["status"] == "succeeded"}
        semaphore = asyncio.Semaphore(config.get("concurrency", 1))

        def save_prediction(case, prediction):
            policy = config.get("invalid_prediction_policy", "fail")
            ids = (
                prediction.ordered_item_ids
                if policy == "upstream"
                else permutation(prediction.ordered_item_ids, case.candidates, policy)
            )
            prediction = replace(prediction, ordered_item_ids=ids)
            result = {
                "user_id": case.user_id,
                "prediction": asdict(prediction),
                "metrics": evaluate_case(case, prediction, policy),
                "state_sha256": manifest["state_snapshots"]["prepared"],
            }
            store.put_case(case.case_id, "succeeded", result)

        def save_failure(case, exc):
            store.put_case(
                case.case_id,
                "failed",
                {
                    "user_id": case.user_id,
                    "error_type": type(exc).__name__,
                    "error": redact(str(exc)),
                },
            )

        async def execute(case):
            if case.case_id in completed:
                return
            async with semaphore:
                store.put_case(case.case_id, "running")
                try:
                    prediction = await model.predict(case)
                    policy = config.get("invalid_prediction_policy", "fail")
                    try:
                        ids = (
                            prediction.ordered_item_ids
                            if policy == "upstream"
                            else permutation(prediction.ordered_item_ids, case.candidates, policy)
                        )
                    except ValueError:
                        if policy != "one_retry":
                            raise
                        # Alter request identity explicitly to avoid replaying the same invalid response.
                        repair_provider = RepairProvider(provider)
                        original = context.provider
                        context.provider = repair_provider
                        try:
                            prediction = await model.predict(case)
                        finally:
                            context.provider = original
                        ids = permutation(prediction.ordered_item_ids, case.candidates)
                    save_prediction(case, replace(prediction, ordered_item_ids=ids))
                except Exception as exc:
                    save_failure(case, exc)

        # one_retry mutates provider injection; keep that explicitly serialized.
        if config.get("invalid_prediction_policy") == "one_retry":
            semaphore = asyncio.Semaphore(1)
        if use_batch:
            remaining = [c for c in cases if c.case_id not in completed]
            for case in remaining:
                store.put_case(case.case_id, "running")
            batch = BatchClient(config, directory, store, recorder)
            try:
                predictions = await model.predict_batch(remaining, batch)
            except BatchPending as exc:
                manifest["status"] = "batch_pending"
                write_json(manifest_path, manifest)
                return {
                    "run": str(directory.resolve()),
                    "status": "batch_pending",
                    "message": str(exc),
                }
            for case in remaining:
                try:
                    prediction = predictions[case.case_id]
                    if isinstance(prediction, Exception):
                        raise prediction
                    save_prediction(case, prediction)
                except Exception as exc:
                    save_failure(case, exc)
        else:
            await asyncio.gather(*(execute(c) for c in cases))
        rows = store.rows()
        metrics = aggregate(rows)
        costs = recorder.summary()
        costs.update(store.accounting_summary())
        if use_batch:
            costs["total_calls"] = store.db.execute("SELECT COUNT(*) FROM accounting").fetchone()[0]
        metrics.update(costs)
        metrics["avg_calls_per_case"] = costs["total_calls"] / len(cases)
        write_json(directory / "metrics.json", metrics)
        write_json(directory / "cost_report.json", costs)
        (directory / "predictions.jsonl").write_text(
            "".join(canonical(r) + "\n" for r in rows), encoding="utf-8"
        )
        manifest["status"] = "completed" if metrics["n_failed"] == 0 else "completed_with_failures"
        manifest["metrics"] = metrics
        manifest["execution_mode"] = "batch" if use_batch else "online"
        manifest["effective_concurrency"] = (
            None
            if use_batch
            else (
                1
                if config.get("invalid_prediction_policy") == "one_retry"
                else config.get("concurrency", 1)
            )
        )
        write_json(manifest_path, manifest)
        return {"run": str(directory.resolve()), "status": manifest["status"], "metrics": metrics}
    finally:
        if store:
            store.close()
        run_lock.rollback()
        run_lock.close()


class RepairProvider:
    def __init__(self, provider):
        self.provider = provider

    async def generate(self, messages, **kwargs):
        fields = kwargs.get("schema", {}).get("properties", {})
        if "scores" not in fields and "rerank_list" not in fields:
            return await self.provider.generate(messages, **kwargs)
        return await self.provider.generate(
            messages
            + [
                {
                    "role": "user",
                    "content": "Repair the output: return each candidate exactly once, no other item IDs.",
                }
            ],
            **{**kwargs, "repair_attempt": 1},
        )
