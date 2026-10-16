"""Clock with an offset file so the demo can advance time instead of waiting."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path


def _offset_path() -> Path:
    return Path(os.environ.get("FAIRXAI_CLOCK_OFFSET_FILE", ".clock_offset"))


def offset_seconds() -> float:
    try:
        return float(_offset_path().read_text().strip() or 0)
    except (FileNotFoundError, ValueError):
        return 0.0


def now() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=offset_seconds())


def advance(seconds: float) -> None:
    _offset_path().write_text(str(offset_seconds() + seconds))


def reset() -> None:
    _offset_path().unlink(missing_ok=True)
