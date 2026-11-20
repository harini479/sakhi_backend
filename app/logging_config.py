"""Structured JSON logging formatted for Google Cloud Logging with PHI/secret redaction."""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict


class CloudRunJsonFormatter(logging.Formatter):
    """Emits single-line JSON logs recognized natively by Google Cloud Logging."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("request_id", "method", "path", "status_code", "latency_ms", "event"):
            val = getattr(record, field, None)
            if val is not None:
                payload[field] = val
        if record.exc_info and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> logging.Logger:
    root = logging.getLogger("sakhi")
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root.setLevel(numeric_level)
    root.propagate = False

    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(CloudRunJsonFormatter())
        root.addHandler(handler)
    else:
        for handler in root.handlers:
            handler.setFormatter(CloudRunJsonFormatter())

    return root
