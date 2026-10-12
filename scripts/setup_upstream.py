"""Fetch exact audited source without redistributing upstream code or data."""

import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
for name, info in json.loads((root / "third_party" / "lock.json").read_text()).items():
    target = root / "third_party" / name
    if not target.exists():
        subprocess.run(["git", "clone", "--no-checkout", info["url"], str(target)], check=True)
    command = ["git", "-c", f"safe.directory={target.as_posix()}", "-C", str(target)]
    dirty = subprocess.check_output(
        command + ["status", "--porcelain", "--untracked-files=no"], text=True
    )
    # A no-checkout clone has deleted index entries. Only inspect dirty existing checkout.
    head_file = target / ("model.py" if name == "iAgent" else "README.md")
    if head_file.exists() and dirty.strip():
        raise RuntimeError(f"{name} has local changes; refusing checkout")
    subprocess.run(command + ["fetch", "origin", info["commit"]], check=True)
    subprocess.run(command + ["checkout", "--detach", info["commit"]], check=True)
    print(f"{name} pinned at {info['commit']}; license status: {info['license']}")
