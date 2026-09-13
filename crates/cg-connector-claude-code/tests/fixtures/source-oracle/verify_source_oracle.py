#!/usr/bin/env python3
"""Verify independent synthetic source facts without importing production code."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW_TEXT_SENTINELS = {
    "compact-rich.jsonl": [
        "Account A-17 needs a transfer review.",
        "café 東京",
        "decisive sentence after the former preview boundary",
        "search_document: literal source prefix",
        "A tab follows:\\tΩ.",
        "search_query: compare the two synthetic transports.",
        "Iroh carries the state over direct peers; NATS routes messages through a broker.",
        "FORBIDDEN_MEDIA_MARKER_6f31",
    ],
    "pending-tail-partial.jsonl": ["still being written"],
}
EXPECTED_RAW_OFFSETS = {
    "compact-rich.jsonl": [190, 300, 864, 1006, 1081, 3960, 3670, 2549],
    "pending-tail-partial.jsonl": [410],
}
CHECKSUMMED = (
    "README.md", "compact-rich.jsonl", "expected-occurrences.json",
    "generate_oversized_tail.py", "malformed-after-valid.jsonl", "oversized-tail.jsonl",
    "pending-tail-complete.jsonl", "pending-tail-partial.jsonl",
    "second-stream-same-session.jsonl", "verify_source_oracle.py",
    "whitespace-first-record.jsonl", "whitespace-only.txt",
)


def raw_offset(data: bytes, needle: str) -> int:
    found = data.find(needle.encode("utf-8"))
    if found < 0:
        raise AssertionError(f"missing raw sentinel: {needle!r}")
    if data.find(needle.encode("utf-8"), found + 1) >= 0:
        raise AssertionError(f"ambiguous raw sentinel: {needle!r}")
    return found


def jsonl_lines(path: Path, allow_pending: bool = False) -> list[dict]:
    data = path.read_bytes()
    parts = data.splitlines(keepends=True)
    decoded = []
    for index, part in enumerate(parts, 1):
        if not part.endswith(b"\n"):
            if allow_pending:
                break
            raise AssertionError(f"{path.name}: unexpected unterminated line {index}")
        if part.strip():
            decoded.append(json.loads(part))
    return decoded


def message_item(record: dict, item: int):
    content = record.get("message", {}).get("content")
    return content[item] if isinstance(content, list) else content


def verify_expected_facts(expected: dict) -> int:
    count = 0
    for filename, stream in expected["streams"].items():
        records = jsonl_lines(HERE / filename)
        for fact in stream["accepted_occurrences"]:
            record = records[fact["record"] - 1]
            assert record["uuid"] == fact["uuid"], fact
            if "message_id" in fact:
                assert record["message"]["id"] == fact["message_id"], fact
            if "text" in fact:
                assert message_item(record, fact["item"]).get("text") == fact["text"] if isinstance(message_item(record, fact["item"]), dict) else message_item(record, fact["item"]) == fact["text"], fact
            count += 1
    return count


def verify_semantic_sentinels() -> None:
    rows = jsonl_lines(HERE / "compact-rich.jsonl")
    assert len(rows) == 12 and {row["sessionId"] for row in rows} == {"synthetic-session-amber"}
    multipart = rows[1]["message"]["content"]
    assert [item["type"] for item in multipart] == ["text", "tool_use", "text"]
    assert multipart[1]["id"] == "tool-edit-1" and multipart[1]["name"] == "Edit"
    assert multipart[1]["input"] == {"file_path": "/workspace/notice.md", "old_string": "seven days", "new_string": "fourteen days", "reason": "retain the negotiated notice period"}
    assert rows[2]["message"]["content"][0]["tool_use_id"] == "tool-edit-1"
    assert rows[3]["message"]["content"][0]["id"] == "tool-orphan-call"
    assert rows[4]["message"]["content"][0]["tool_use_id"] == "tool-without-call"
    attachment, media = rows[5]["message"]["content"]
    assert attachment == {"type": "text_attachment", "text": "Clause 4: payment follows written approval.", "filename": "terms.txt", "media_type": "text/plain", "reference": "memo://terms-4"}
    assert media["source"]["data"] == "FORBIDDEN_MEDIA_MARKER_6f31"
    assert [item["type"] for item in rows[6]["message"]["content"]] == ["text", "future_delta", "text"]
    assert rows[7]["type"] == "telemetry_zeta" and "type" not in rows[8]
    assert rows[9]["subagent"] == {"id": "teammate-river", "name": "River"}
    assert [row["message"]["id"] for row in rows if "message" in row].count("repeat-msg-7") == 4
    assert "timestamp" not in rows[10] and rows[11]["timestamp"] == "not-a-timestamp"
    second = jsonl_lines(HERE / "second-stream-same-session.jsonl")
    assert second[0]["sessionId"] == rows[0]["sessionId"] and second[0]["uuid"] != rows[0]["uuid"]


def main() -> int:
    subprocess.run([sys.executable, str(HERE / "generate_oversized_tail.py")], check=True)
    expected = json.loads((HERE / "expected-occurrences.json").read_text(encoding="utf-8"))
    fact_count = verify_expected_facts(expected)
    verify_semantic_sentinels()
    for filename, sentinels in RAW_TEXT_SENTINELS.items():
        data = (HERE / filename).read_bytes()
        offsets = [raw_offset(data, sentinel) for sentinel in sentinels]
        assert offsets == EXPECTED_RAW_OFFSETS[filename], (filename, offsets)
        for sentinel, offset in zip(sentinels, offsets):
            print(f"offset {filename} {offset} {sentinel!r}")
    malformed = (HERE / "malformed-after-valid.jsonl").read_bytes().splitlines(keepends=True)
    assert len(malformed) == 2 and malformed[1].endswith(b"\n")
    json.loads(malformed[0])
    try:
        json.loads(malformed[1])
    except json.JSONDecodeError as error:
        assert error.pos >= 0
        print(f"malformed line=2 byte={error.pos}")
    else:
        raise AssertionError("malformed fixture unexpectedly parsed")
    partial = (HERE / "pending-tail-partial.jsonl").read_bytes()
    assert partial.count(b"\n") == 1 and not partial.endswith(b"\n")
    assert len(jsonl_lines(HERE / "pending-tail-partial.jsonl", allow_pending=True)) == 1
    completed = jsonl_lines(HERE / "pending-tail-complete.jsonl")
    assert len(completed) == 2 and partial.split(b"\n", 1)[0] == (HERE / "pending-tail-complete.jsonl").read_bytes().split(b"\n", 1)[0]
    whitespace = (HERE / "whitespace-only.txt").read_bytes()
    assert whitespace.strip() == b""
    first = (HERE / "whitespace-first-record.jsonl").read_bytes()
    assert first.startswith(whitespace) and len(jsonl_lines(HERE / "whitespace-first-record.jsonl")) == 1
    oversized = (HERE / "oversized-tail.jsonl").read_bytes()
    assert len(oversized) > 16_384 and oversized.count(b"\n") == 1 and not oversized.endswith(b"\n")
    json.loads(oversized.split(b"\n", 1)[0])
    print(f"accepted_source_facts={fact_count}")
    observed_checksums = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in CHECKSUMMED
    }
    manifest = {
        line.split()[1]: line.split()[0]
        for line in (HERE / "fixture-checksums.sha256").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert observed_checksums == manifest, "fixture checksum manifest differs"
    for name in CHECKSUMMED:
        print(f"sha256 {observed_checksums[name]}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
