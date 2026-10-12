import json
import subprocess
import sys

import yaml

from utils.core.config import ROOT, load_yaml
from utils.data.dataset import make_fixture


def test_main_entrypoint_from_another_directory(tmp_path):
    config = load_yaml(ROOT / "configs/experiments/books100.yaml")
    config["manifest"] = str(make_fixture(tmp_path / "data"))
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "main.py"),
            "--config",
            str(config_path),
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
