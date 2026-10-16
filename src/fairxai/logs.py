"""Logging setup that never emits API keys."""

from __future__ import annotations

import logging
import re

_SECRET = re.compile(r"(?i)(x-api-key|api[_-]?key|password|secret)(['\"]?\s*[:=]\s*['\"]?)([^\s,'\"]+)")


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _SECRET.sub(r"\1\2***", record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        h.addFilter(RedactFilter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log
