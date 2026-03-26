#!/usr/bin/env python3
"""Replay hard-mode OTLP dataset manifests directly to Tempo via gRPC."""

from __future__ import annotations

import argparse
import concurrent.futures
import threading
from pathlib import Path
import sys
from typing import Any

import grpc
from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tempo_hardmode_otlp import (
    build_export_request_for_entries,
    chunk_payload_text,
    load_manifest_from_path_or_s3,
    now_unix_ms,
    resolve_blob_text,
    write_json,
)


def _group_entries_by_trace(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(str(entry["trace_id"]), []).append(entry)
    out: list[list[dict[str, Any]]] = []
    for _trace_id, trace_entries in groups.items():
        trace_entries.sort(key=lambda item: int(item.get("span_index", 0)))
        out.append(trace_entries)
    out.sort(key=lambda trace_entries: str(trace_entries[0]["trace_id"]))
    return out


def _batched(
    trace_entries: list[dict[str, Any]],
    *,
    spans_per_request: int,
) -> list[list[dict[str, Any]]]:
    if spans_per_request <= 0:
        raise ValueError("spans_per_request must be > 0")
    return [
        trace_entries[i : i + spans_per_request]
        for i in range(0, len(trace_entries), spans_per_request)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay OTLP dataset manifest directly to Tempo gRPC endpoint"
    )
    parser.add_argument("--manifest", required=True, help="Local path or s3:// URI")
    parser.add_argument("--endpoint", default="localhost:4317")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-message-bytes", type=int, default=2_147_483_647)
    parser.add_argument(
        "--compression",
        choices=["none", "gzip"],
        default="none",
    )
    parser.add_argument(
        "--spans-per-request",
        type=int,
        default=0,
        help="If set >0, fixed spans per request. If 0, auto by span size.",
    )
    parser.add_argument(
        "--spans-per-request-small",
        type=int,
        default=10,
        help="Auto mode: spans/request for spans smaller than large threshold.",
    )
    parser.add_argument(
        "--large-span-threshold-bytes",
        type=int,
        default=50 * 1024 * 1024,
    )
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--payload-seed", default="tempo-hardmode-v1")
    parser.add_argument("--canonical-cache-dir", default="")
    parser.add_argument("--aws-region", default="")
    parser.add_argument("--results-output", required=True)
    parser.add_argument("--allow-failures", action="store_true")
    args = parser.parse_args()

    if args.concurrency <= 0:
        raise SystemExit("--concurrency must be > 0")
    if args.worker_count <= 0:
        raise SystemExit("--worker-count must be > 0")
    if args.worker_id < 0 or args.worker_id >= args.worker_count:
        raise SystemExit("--worker-id must be in [0, worker-count)")
    if args.spans_per_request < 0:
        raise SystemExit("--spans-per-request must be >= 0")
    if args.spans_per_request_small <= 0:
        raise SystemExit("--spans-per-request-small must be > 0")

    manifest = load_manifest_from_path_or_s3(
        args.manifest,
        aws_region=args.aws_region or None,
    )
    all_entries = list(manifest.get("entries") or [])
    if not all_entries:
        raise SystemExit("manifest has no entries")

    shard_entries = [
        entry
        for idx, entry in enumerate(all_entries)
        if (idx % args.worker_count) == args.worker_id
    ]
    if not shard_entries:
        raise SystemExit("no entries selected for this worker shard")

    fallback_canonical_dir = (
        Path(args.canonical_cache_dir) if args.canonical_cache_dir.strip() else None
    )
    payload_chunks_by_sha: dict[str, list[str]] = {}
    for entry in shard_entries:
        payload_sha = str(entry["payload_sha256"])
        if payload_sha in payload_chunks_by_sha:
            continue
        payload_text = resolve_blob_text(
            entry=entry,
            payload_seed=args.payload_seed,
            fallback_canonical_dir=fallback_canonical_dir,
        )
        payload_chunks_by_sha[payload_sha] = chunk_payload_text(
            payload_text,
            int(entry["chunk_size_bytes"]),
        )

    grouped = _group_entries_by_trace(shard_entries)
    request_batches: list[list[dict[str, Any]]] = []
    for trace_entries in grouped:
        if args.spans_per_request > 0:
            n = args.spans_per_request
        else:
            first_size = int(trace_entries[0]["span_payload_target_bytes"])
            n = (
                1
                if first_size >= args.large_span_threshold_bytes
                else args.spans_per_request_small
            )
        request_batches.extend(_batched(trace_entries, spans_per_request=n))

    if args.compression == "gzip":
        grpc_compression = grpc.Compression.Gzip
    else:
        grpc_compression = grpc.Compression.NoCompression

    channel_options = [
        ("grpc.max_send_message_length", int(args.max_message_bytes)),
        ("grpc.max_receive_message_length", int(args.max_message_bytes)),
    ]

    thread_local = threading.local()
    status_by_span_id: dict[str, dict[str, Any]] = {}
    status_lock = threading.Lock()

    def get_stub() -> trace_service_pb2_grpc.TraceServiceStub:
        if getattr(thread_local, "stub", None) is None:
            channel = grpc.insecure_channel(
                args.endpoint,
                options=channel_options,
                compression=grpc_compression,
            )
            thread_local.channel = channel
            thread_local.stub = trace_service_pb2_grpc.TraceServiceStub(channel)
        return thread_local.stub

    def send_batch(batch: list[dict[str, Any]]) -> None:
        request = build_export_request_for_entries(
            entries=batch,
            payload_chunks_by_sha=payload_chunks_by_sha,
            service_name=str(batch[0]["service_name"]),
        )
        stub = get_stub()
        sent_at_ms = now_unix_ms()
        try:
            stub.Export(request, timeout=args.timeout_seconds)
            ok = True
            error = ""
        except grpc.RpcError as exc:
            ok = False
            error = f"{exc.code().name}: {exc.details() or ''}".strip()
        except Exception as exc:
            ok = False
            error = str(exc)

        with status_lock:
            for entry in batch:
                span_id = str(entry["span_id"])
                status_by_span_id[span_id] = {
                    "replay_ok": ok,
                    "replay_error": error,
                    "emitted_at_unix_ms": sent_at_ms if ok else 0,
                }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(send_batch, batch) for batch in request_batches]
        for future in concurrent.futures.as_completed(futures):
            # Propagate unexpected local exceptions.
            future.result()

    replayed_entries: list[dict[str, Any]] = []
    success_count = 0
    failure_count = 0
    for entry in shard_entries:
        span_id = str(entry["span_id"])
        status = status_by_span_id.get(span_id) or {
            "replay_ok": False,
            "replay_error": "missing_status",
            "emitted_at_unix_ms": 0,
        }
        out_entry = dict(entry)
        out_entry.update(status)
        replayed_entries.append(out_entry)
        if status["replay_ok"]:
            success_count += 1
        else:
            failure_count += 1

    result_payload = {
        "schema_version": 1,
        "manifest_type": "tempo_hardmode_replay_result",
        "source_manifest": args.manifest,
        "replayed_at_unix_ms": now_unix_ms(),
        "endpoint": args.endpoint,
        "compression": args.compression,
        "worker_id": args.worker_id,
        "worker_count": args.worker_count,
        "entry_count": len(replayed_entries),
        "replay_successes": success_count,
        "replay_failures": failure_count,
        "entries": replayed_entries,
    }
    output_path = Path(args.results_output)
    write_json(output_path, result_payload)
    print(
        f"Wrote replay results: {output_path} "
        f"successes={success_count} failures={failure_count}"
    )

    if failure_count > 0 and not args.allow_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
