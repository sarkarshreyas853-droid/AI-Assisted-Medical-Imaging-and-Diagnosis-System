"""Append-only audit log for radiologist decisions (JSON lines)."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

LOG_PATH = os.environ.get("AUDIT_LOG", "logs/audit.jsonl")
_lock = threading.Lock()


def record(entry: dict) -> dict:
    entry = {**entry, "timestamp": datetime.now(timezone.utc).isoformat()}
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_all() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
