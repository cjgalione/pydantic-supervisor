from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tempo_hardmode_otlp import (
    build_export_request_for_entries,
    chunk_payload_text,
    deterministic_payload_text,
    payload_sha256_hex,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_payload_size_and_hash_are_deterministic() -> None:
    payload_a = deterministic_payload_text(8192, "seed-1")
    payload_b = deterministic_payload_text(8192, "seed-1")
    payload_c = deterministic_payload_text(8192, "seed-2")

    assert len(payload_a.encode("utf-8")) == 8192
    assert payload_a == payload_b
    assert payload_a != payload_c
    assert payload_sha256_hex(payload_a) == payload_sha256_hex(payload_b)

    chunks = chunk_payload_text(payload_a, 1024)
    assert len(chunks) == 8
    assert "".join(chunks) == payload_a


def test_generate_dataset_manifest_schema_and_wire_size(tmp_path: Path) -> None:
    out_dir = tmp_path / "dataset"
    manifest = out_dir / "step1_dataset_manifest.json"
    cmd = [
        "python3",
        "scripts/generate_otlp_dataset.py",
        "--output-dir",
        str(out_dir),
        "--stage-name",
        "step1",
        "--run-tag",
        "pytest-run",
        "--trace-count",
        "1",
        "--spans-per-trace",
        "2",
        "--span-payload-bytes",
        "4096",
        "--chunk-size-bytes",
        "1024",
        "--manifest-path",
        str(manifest),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_repo_root())
    subprocess.run(cmd, cwd=_repo_root(), env=env, check=True)
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    entries = doc["entries"]
    assert doc["manifest_type"] == "tempo_hardmode_otlp_dataset"
    assert doc["entry_count"] == 2
    assert len(entries) == 2
    required = {
        "trace_id",
        "span_id",
        "span_payload_target_bytes",
        "wire_request_bytes",
        "payload_sha256",
        "chunk_count",
        "chunk_size_bytes",
        "run_tag",
        "emitted_at_unix_ms",
    }
    for entry in entries:
        assert required.issubset(set(entry.keys()))
        assert int(entry["wire_request_bytes"]) > 0
        assert int(entry["span_payload_target_bytes"]) == 4096
        assert int(entry["chunk_count"]) == 4


def test_request_builder_roundtrip_preserves_payload_markers() -> None:
    payload = deterministic_payload_text(2048, "mock-seed")
    payload_sha = payload_sha256_hex(payload)
    payload_chunks = chunk_payload_text(payload, 512)
    entry = {
        "trace_id": "1" * 32,
        "span_id": "2" * 16,
        "trace_index": 0,
        "span_index": 0,
        "stage_name": "step1",
        "run_tag": "pytest-hardmode",
        "span_name": "hardmode_span_0000",
        "service_name": "pydantic-supervisor-hardmode",
        "span_payload_target_bytes": 2048,
        "payload_sha256": payload_sha,
        "chunk_count": len(payload_chunks),
        "chunk_size_bytes": 512,
    }
    request = build_export_request_for_entries(
        entries=[entry],
        payload_chunks_by_sha={payload_sha: payload_chunks},
        service_name="pydantic-supervisor-hardmode",
    )
    wire = request.SerializeToString()
    parsed = request.__class__()
    parsed.ParseFromString(wire)
    spans = parsed.resource_spans[0].scope_spans[0].spans
    assert len(spans) == 1
    attrs = {attr.key: attr.value.string_value for attr in spans[0].attributes}
    assert attrs["hardmode.payload.sha256"] == payload_sha
    payload_keys = [k for k in attrs.keys() if k.startswith("hardmode.payload.chunk.")]
    assert len(payload_keys) == len(payload_chunks)
