#!/usr/bin/env python3
"""Deterministically generate the single oversized-tail fixture."""
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
text = "oversized retained text " + ("0123456789abcdef" * 1024) + " decisive oversized sentinel"
record = {
    "type": "user",
    "uuid": "50000000-0000-4000-8000-000000000001",
    "sessionId": "synthetic-session-oversized",
    "timestamp": "2026-09-05T09:00:00Z",
    "message": {"id": "oversized-1", "role": "user", "content": text},
}
complete = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
pending = '{"type":"assistant","uuid":"50000000-0000-4000-8000-000000000002"'
(HERE / "oversized-tail.jsonl").write_text(complete + pending, encoding="utf-8")
