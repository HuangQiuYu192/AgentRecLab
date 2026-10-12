import json
import subprocess
import sys

from utils.config import ROOT


def test_main_entrypoint_from_another_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--config",
            str(ROOT / "configs/experiments/smoke_books.yaml"),
            "--model",
            "i2agent",
            "--dry-run",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    summary = json.loads(result.stdout)
    assert summary["n_cases"] == 10
    assert summary["planned_calls_without_retries"] == 50
    assert summary["paid_calls"] is False
