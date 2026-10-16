"""Fail if a blueprint top-level folder is missing or an unlisted top-level folder exists (SPEC Section E)."""

from __future__ import annotations

import sys
from pathlib import Path

ALLOWED = {
    "requirements",
    "configs",
    "data",
    "src",
    "pipelines",
    "serving",
    "gateway",
    "simulator",
    "infra",
    "scripts",
    "experiments",
    "tests",
    "docs",
}
IGNORED = {".git", ".github", ".dvc", ".venv", ".pytest_cache", ".ruff_cache", "mlruns", "__pycache__"}


def check(root: Path) -> list[str]:
    present = {p.name for p in root.iterdir() if p.is_dir()} - IGNORED
    errors = [f"missing top-level folder: {m}" for m in sorted(ALLOWED - present)]
    errors += [f"unlisted top-level folder: {u}" for u in sorted(present - ALLOWED)]
    return errors


if __name__ == "__main__":
    errs = check(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
    for e in errs:
        print(e)
    sys.exit(1 if errs else 0)
