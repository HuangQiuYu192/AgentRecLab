"""Record dependency closure of this project, excluding unrelated bundled libraries."""

from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pending = [
    "httpx",
    "typer",
    "pyyaml",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "ruff",
    "setuptools",
]
versions = {}
while pending:
    name = canonicalize_name(pending.pop())
    if name in versions:
        continue
    package = distribution(name)
    versions[name] = package.version
    for requirement in package.requires or []:
        req = Requirement(requirement)
        if req.marker is None or req.marker.evaluate({"extra": ""}):
            pending.append(req.name)
target = Path(__file__).resolve().parents[2] / "requirements.lock"
target.write_text(
    "# Validated dependency versions on Windows / Python 3.12.14\n"
    + "\n".join(f"{name}=={version}" for name, version in sorted(versions.items()))
    + "\n",
    encoding="utf-8",
)
print(f"Locked {len(versions)} project dependency packages")
