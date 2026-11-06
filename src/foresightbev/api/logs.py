"""Structured backend logging: JSON lines in a rotating file, also on stderr."""

from __future__ import annotations

import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_NAME = "backend.log"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(record.created)), level=record.levelname, logger=record.name, message=record.getMessage())
        for key in ("request_id", "method", "path", "status", "duration_ms", "job_id"):
            if hasattr(record, key):
                doc[key] = getattr(record, key)
        if record.exc_info:
            doc["exception"] = self.formatException(record.exc_info)
        return json.dumps(doc)


def configure_logging(directory: Path, level: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / LOG_NAME
    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, RotatingFileHandler) and Path(h.baseFilename) == path for h in root.handlers):
        for handler in list(root.handlers):
            if isinstance(handler, RotatingFileHandler):
                root.removeHandler(handler)
                handler.close()
        handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    return path


def tail(path: Path, lines: int, level: str | None) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", errors="replace") as stream:
        for line in stream.readlines()[-max(lines * 5, 1000):]:
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                doc = dict(level="UNPARSED", message=line.strip())
            if level is None or doc.get("level") == level:
                rows.append(doc)
    return rows[-lines:]
