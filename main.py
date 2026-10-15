import asyncio
import json
import os
import sys
from pathlib import Path

import typer

from utils.core.config import ROOT, load_run, load_yaml
from utils.data.dataset import convert as convert_data
from utils.data.dataset import freeze as freeze_data
from utils.data.dataset import load_dataset
from utils.evaluation.metrics import aggregate, compare_runs
from utils.experiment.cache import prepare_response_cache
from utils.experiment.runner import run_experiment
from utils.experiment.storage import file_hash, read_json

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)
data = typer.Typer(no_args_is_help=True)
app.add_typer(data, name="data")


def display(value):
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@app.command()
def doctor():
    import platform

    display(
        {
            "python": platform.python_version(),
            "models": {
                name: "native; no upstream checkout required"
                for name in ("iagent", "i2agent", "memrec")
            },
            "raw_data_present": (ROOT / "datasets" / "raw" / "books").exists(),
            "configured_keys": {
                k: bool(os.environ.get(k)) for k in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY")
            },
            "mock": "available; no network required",
            "real_connectivity": "not probed: use opt-in paid run after dry-run",
        }
    )


@data.command()
def inspect(dataset: str = typer.Option(...), raw_dir: Path = typer.Option(...)):
    display(
        {
            "dataset": dataset,
            "exists": raw_dir.exists(),
            "files": [
                {
                    "name": str(p.relative_to(raw_dir)),
                    "bytes": p.stat().st_size,
                    "sha256": file_hash(p),
                }
                for p in raw_dir.rglob("*")
                if p.is_file()
            ],
        }
    )


@data.command()
def convert(dataset: str = typer.Option(...), config: Path = typer.Option(...)):
    options = load_yaml(config)
    for key in ("pickle", "mapping", "out"):
        if options.get(key) and not Path(options[key]).is_absolute():
            options[key] = str(ROOT / options[key])
    display({"manifest": str(convert_data(dataset, options))})


@data.command()
def freeze(
    dataset: str = typer.Option(...),
    out: Path = typer.Option(...),
    seed: int = 42,
    candidates: int = 10,
    source: Path | None = None,
    limit: int | None = None,
):
    source = source or ROOT / "datasets" / "processed" / dataset / "dataset_manifest.json"
    display({"manifest": str(freeze_data(source, out, seed, limit, candidates))})


@app.command()
def validate(manifest: Path = typer.Option(...)):
    cases, interactions, catalog, _ = load_dataset(manifest)
    display(
        {
            "valid": True,
            "n_cases": len(cases),
            "n_interactions": len(interactions),
            "n_items": len(catalog),
        }
    )


@app.command("prepare-cache")
def prepare_cache(
    config: Path = typer.Option(...),
    source: Path = typer.Option(...),
    out: Path = typer.Option(...),
):
    """Prepare and validate a cache snapshot offline; never dispatch API requests."""
    display(prepare_response_cache(source, out, load_run(config)))


@app.command()
def run(
    config: Path = typer.Option(...),
    provider: str | None = None,
    model: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    allow_paid: bool = False,
    run_id: str | None = None,
    use_batch: bool | None = typer.Option(None, "--use-batch/--no-batch"),
):
    resolved = load_run(config, provider, model)
    if use_batch is not None:
        resolved["use_batch"] = use_batch
    if run_id:
        resolved["run_id"] = run_id
    elif model:
        resolved["run_id"] = resolved["run_id"] + "_" + model
    result = asyncio.run(run_experiment(resolved, resume, dry_run, allow_paid))
    display(result)
    if result.get("status") == "completed_with_failures":
        raise typer.Exit(1)


@app.command()
def evaluate(run: Path = typer.Option(...)):
    rows = [
        json.loads(line)
        for line in (run / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    display(aggregate(rows))


@app.command()
def compare(runs: list[Path] = typer.Option(...), paired_only: bool = False):
    display(compare_runs(runs, paired_only))


@app.command()
def report(run: Path = typer.Option(...), out: Path = typer.Option(...)):
    manifest, metrics = read_json(run / "manifest.json"), read_json(run / "metrics.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = f"# {manifest['run_id']}\n\nProtocol: {manifest['protocol_mode']}\n\nStatus: {manifest['reproduction_status']}\n\nSynthetic mock metrics validate infrastructure only.\n\n"
    text += "| Metric | Exclude failures | Count failures as zero |\n|---|---:|---:|\n"
    for key in metrics["exclude"]:
        text += f"| {key} | {metrics['exclude'][key]} | {metrics['count_as_failure'][key]} |\n"
    text += f"\nSucceeded: {metrics['n_evaluated']}; failed: {metrics['n_failed']}; cost CNY: {metrics['total_cost_cny']}; unknown-cost calls: {metrics['unknown_cost_calls']}\n\nManifest SHA-256: `{file_hash(run / 'manifest.json')}`\n\n"
    text += "Disclosed deviations:\n\n" + "\n".join("- " + d for d in manifest["deviations"]) + "\n"
    out.write_text(text, encoding="utf-8")
    display({"report": str(out)})


def entrypoint():
    args = sys.argv[1:]
    if (
        args
        and args[0].startswith("--")
        and args[0] not in ("--help", "--install-completion", "--show-completion")
    ):
        args = ["run", *args]
    app(args=args)


if __name__ == "__main__":
    entrypoint()
