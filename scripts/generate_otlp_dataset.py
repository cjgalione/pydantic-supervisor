#!/usr/bin/env python3
"""Generate deterministic hard-mode OTLP dataset manifests for Tempo stress."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tempo_hardmode_otlp import (
    DEFAULT_CHUNK_SIZE_BYTES,
    DEFAULT_SPAN_SIZE_CLASSES,
    chunk_payload_text,
    compute_wire_request_size_bytes,
    ensure_canonical_blob,
    now_unix_ms,
    payload_sha256_hex,
    read_json,
    span_id_for,
    trace_id_for,
    write_json,
)


def _parse_sizes(raw: str) -> list[int]:
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic hard-mode span dataset manifest"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage-name", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--trace-count", type=int, default=1)
    parser.add_argument("--spans-per-trace", type=int, default=100)
    parser.add_argument("--span-payload-bytes", type=int, required=True)
    parser.add_argument("--chunk-size-bytes", type=int, default=DEFAULT_CHUNK_SIZE_BYTES)
    parser.add_argument(
        "--payload-seed",
        default="tempo-hardmode-v1",
        help="Deterministic seed for canonical payload bytes",
    )
    parser.add_argument(
        "--service-name",
        default="pydantic-supervisor-hardmode",
    )
    parser.add_argument(
        "--span-name-prefix",
        default="hardmode_span",
    )
    parser.add_argument(
        "--canonical-dir",
        default="",
        help="Optional canonical payload blob directory (defaults to <output-dir>/canonical)",
    )
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Optional explicit manifest file path (defaults to <output-dir>/<stage>_dataset_manifest.json)",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=0,
        help="Optional number of shard manifests to emit",
    )
    parser.add_argument(
        "--size-classes",
        default="",
        help="Optional override for canonical size class generation (comma-separated bytes)",
    )
    args = parser.parse_args()

    if args.trace_count <= 0:
        raise SystemExit("--trace-count must be > 0")
    if args.spans_per_trace <= 0:
        raise SystemExit("--spans-per-trace must be > 0")
    if args.span_payload_bytes <= 0:
        raise SystemExit("--span-payload-bytes must be > 0")
    if args.chunk_size_bytes <= 0:
        raise SystemExit("--chunk-size-bytes must be > 0")
    if args.shards < 0:
        raise SystemExit("--shards must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_dir = (
        Path(args.canonical_dir) if args.canonical_dir else (output_dir / "canonical")
    )
    manifest_path = (
        Path(args.manifest_path)
        if args.manifest_path
        else (output_dir / f"{args.stage_name}_dataset_manifest.json")
    )

    classes = (
        _parse_sizes(args.size_classes)
        if args.size_classes.strip()
        else list(DEFAULT_SPAN_SIZE_CLASSES)
    )
    if args.span_payload_bytes not in classes:
        classes.append(args.span_payload_bytes)
    classes = sorted(set(classes))

    canonical_blobs: dict[str, dict[str, object]] = {}
    payload_chunks_for_selected: list[str] = []
    selected_payload_sha = ""
    selected_blob_path = ""
    for size in classes:
        blob_path, payload_sha = ensure_canonical_blob(
            canonical_dir=canonical_dir,
            span_payload_target_bytes=size,
            payload_seed=args.payload_seed,
        )
        text = blob_path.read_text(encoding="utf-8")
        chunks = chunk_payload_text(text, args.chunk_size_bytes)
        canonical_blobs[str(size)] = {
            "canonical_blob_path": str(blob_path),
            "payload_sha256": payload_sha,
            "payload_size_bytes": len(text.encode("utf-8")),
            "chunk_count": len(chunks),
            "chunk_size_bytes": args.chunk_size_bytes,
        }
        if size == args.span_payload_bytes:
            payload_chunks_for_selected = chunks
            selected_payload_sha = payload_sha
            selected_blob_path = str(blob_path)

    if not payload_chunks_for_selected:
        raise SystemExit("failed to initialize selected payload chunk set")

    exemplar_entry = {
        "trace_id": trace_id_for(args.run_tag, args.stage_name, 0),
        "span_id": span_id_for(args.run_tag, args.stage_name, 0, 0),
        "trace_index": 0,
        "span_index": 0,
        "stage_name": args.stage_name,
        "run_tag": args.run_tag,
        "span_name": f"{args.span_name_prefix}_0000",
        "service_name": args.service_name,
        "span_payload_target_bytes": args.span_payload_bytes,
        "payload_sha256": selected_payload_sha,
        "chunk_count": len(payload_chunks_for_selected),
        "chunk_size_bytes": args.chunk_size_bytes,
        "wire_request_bytes": 0,
        "emitted_at_unix_ms": 0,
        "canonical_blob_path": selected_blob_path,
    }
    exemplar_wire_size = compute_wire_request_size_bytes(
        entry=exemplar_entry,
        payload_chunks=payload_chunks_for_selected,
        service_name=args.service_name,
    )

    entries: list[dict[str, object]] = []
    for trace_index in range(args.trace_count):
        trace_id = trace_id_for(args.run_tag, args.stage_name, trace_index)
        for span_index in range(args.spans_per_trace):
            entry = {
                "trace_id": trace_id,
                "span_id": span_id_for(args.run_tag, args.stage_name, trace_index, span_index),
                "trace_index": trace_index,
                "span_index": span_index,
                "stage_name": args.stage_name,
                "run_tag": args.run_tag,
                "span_name": f"{args.span_name_prefix}_{span_index:04d}",
                "service_name": args.service_name,
                "span_payload_target_bytes": args.span_payload_bytes,
                "payload_sha256": selected_payload_sha,
                "chunk_count": len(payload_chunks_for_selected),
                "chunk_size_bytes": args.chunk_size_bytes,
                "wire_request_bytes": exemplar_wire_size,
                "emitted_at_unix_ms": 0,
                "canonical_blob_path": selected_blob_path,
            }
            entries.append(entry)

    manifest = {
        "schema_version": 1,
        "manifest_type": "tempo_hardmode_otlp_dataset",
        "generated_at_unix_ms": now_unix_ms(),
        "stage_name": args.stage_name,
        "run_tag": args.run_tag,
        "trace_count": args.trace_count,
        "spans_per_trace": args.spans_per_trace,
        "span_payload_target_bytes": args.span_payload_bytes,
        "chunk_size_bytes": args.chunk_size_bytes,
        "payload_seed": args.payload_seed,
        "canonical_blobs": canonical_blobs,
        "entry_count": len(entries),
        "entries": entries,
    }
    write_json(manifest_path, manifest)
    print(f"Wrote dataset manifest: {manifest_path}")
    print(
        f"summary traces={args.trace_count} spans_per_trace={args.spans_per_trace} "
        f"entries={len(entries)} span_payload_bytes={args.span_payload_bytes} "
        f"chunk_count={len(payload_chunks_for_selected)} wire_request_bytes={exemplar_wire_size}"
    )

    if args.shards > 0:
        shard_count = args.shards
        for shard in range(shard_count):
            shard_entries = [
                entry
                for idx, entry in enumerate(entries)
                if (idx % shard_count) == shard
            ]
            shard_manifest = dict(manifest)
            shard_manifest["entry_count"] = len(shard_entries)
            shard_manifest["entries"] = shard_entries
            shard_manifest["shard_id"] = shard
            shard_manifest["shard_count"] = shard_count
            shard_path = manifest_path.with_name(
                f"{manifest_path.stem}_shard{shard:02d}-of-{shard_count:02d}{manifest_path.suffix}"
            )
            write_json(shard_path, shard_manifest)
            print(f"Wrote shard manifest: {shard_path} entries={len(shard_entries)}")

    # Quick local integrity sanity check for generated manifest.
    loaded = read_json(manifest_path)
    loaded_entries = loaded.get("entries") or []
    if len(loaded_entries) != len(entries):
        raise SystemExit("manifest verification failed: entry count mismatch")
    first = loaded_entries[0] if loaded_entries else {}
    if first:
        payload_path = Path(str(first["canonical_blob_path"]))
        payload_text = payload_path.read_text(encoding="utf-8")
        if payload_sha256_hex(payload_text) != str(first["payload_sha256"]):
            raise SystemExit("manifest verification failed: payload sha mismatch")


if __name__ == "__main__":
    main()
