#!/usr/bin/env python3
"""Validate replayed hard-mode traces against Tempo API source of truth."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tempo_hardmode_otlp import (
    extract_span_objects,
    fetch_json,
    reconstruct_payload_from_attrs,
    span_attributes_map,
    write_json,
)


def _group_by_trace(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        out.setdefault(str(entry["trace_id"]), []).append(entry)
    for trace_entries in out.values():
        trace_entries.sort(key=lambda item: int(item.get("span_index", 0)))
    return out


def _verify_trace(
    *,
    trace_id: str,
    expected_entries: list[dict[str, Any]],
    tempo_base_url: str,
    grafana_base_url: str,
    grafana_auth: str,
    query_timeout_seconds: float,
    poll_interval_seconds: float,
    http_timeout_seconds: float,
) -> dict[str, Any]:
    expected_span_ids = {str(entry["span_id"]) for entry in expected_entries}
    emitted_latest_ms = max(
        int(entry.get("emitted_at_unix_ms", 0) or 0) for entry in expected_entries
    )

    started_ms = int(time.time() * 1000)
    deadline_ms = started_ms + int(query_timeout_seconds * 1000)
    queryable_at_ms = 0
    payload: dict[str, Any] | None = None
    fetch_code = 0
    fetch_error = ""

    while int(time.time() * 1000) <= deadline_ms:
        fetch_code, parsed, raw = fetch_json(
            f"{tempo_base_url.rstrip('/')}/api/traces/{trace_id}",
            timeout_seconds=http_timeout_seconds,
        )
        if fetch_code == 200 and isinstance(parsed, dict):
            payload = parsed
            queryable_at_ms = int(time.time() * 1000)
            break
        fetch_error = raw[:300]
        time.sleep(poll_interval_seconds)

    grafana_code = 0
    grafana_error = ""
    if grafana_base_url:
        grafana_code, _parsed, grafana_raw = fetch_json(
            f"{grafana_base_url.rstrip('/')}/api/traces/{trace_id}",
            timeout_seconds=http_timeout_seconds,
            basic_auth=grafana_auth if grafana_auth else None,
        )
        if grafana_code != 200:
            grafana_error = (grafana_raw or "")[:300]

    if payload is None:
        return {
            "trace_id": trace_id,
            "ok": False,
            "failure_invariant": "trace_not_queryable",
            "tempo_http_code": fetch_code,
            "tempo_error": fetch_error,
            "grafana_http_code": grafana_code,
            "grafana_error": grafana_error,
            "expected_span_count": len(expected_entries),
            "retrieved_span_count": 0,
            "write_to_queryable_ms": -1,
            "span_failures": [
                {
                    "span_id": span_id,
                    "failure_invariant": "trace_not_queryable",
                    "details": "",
                }
                for span_id in sorted(expected_span_ids)
            ],
        }

    span_objects = extract_span_objects(payload)
    found_by_span_id: dict[str, dict[str, Any]] = {}
    for span_obj in span_objects:
        attrs = span_attributes_map(span_obj)
        marker_span_id = attrs.get("hardmode.span_id", "")
        if marker_span_id:
            found_by_span_id[marker_span_id] = span_obj

    span_failures: list[dict[str, str]] = []
    for entry in expected_entries:
        expected_span_id = str(entry["span_id"])
        span_obj = found_by_span_id.get(expected_span_id)
        if span_obj is None:
            span_failures.append(
                {
                    "span_id": expected_span_id,
                    "failure_invariant": "missing_span",
                    "details": "",
                }
            )
            continue

        attrs = span_attributes_map(span_obj)
        payload_text, chunk_count = reconstruct_payload_from_attrs(attrs)
        reconstructed_sha = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
        reconstructed_size = len(payload_text.encode("utf-8"))
        expected_sha = str(entry["payload_sha256"])
        expected_size = int(entry["span_payload_target_bytes"])
        expected_chunk_count = int(entry["chunk_count"])

        if reconstructed_sha != expected_sha:
            span_failures.append(
                {
                    "span_id": expected_span_id,
                    "failure_invariant": "payload_hash_mismatch",
                    "details": f"expected={expected_sha} actual={reconstructed_sha}",
                }
            )
            continue
        if reconstructed_size != expected_size:
            span_failures.append(
                {
                    "span_id": expected_span_id,
                    "failure_invariant": "payload_size_mismatch",
                    "details": f"expected={expected_size} actual={reconstructed_size}",
                }
            )
            continue
        if chunk_count != expected_chunk_count:
            span_failures.append(
                {
                    "span_id": expected_span_id,
                    "failure_invariant": "chunk_count_mismatch",
                    "details": f"expected={expected_chunk_count} actual={chunk_count}",
                }
            )
            continue

    write_to_queryable_ms = -1
    if queryable_at_ms > 0 and emitted_latest_ms > 0:
        write_to_queryable_ms = queryable_at_ms - emitted_latest_ms

    overall_ok = len(span_failures) == 0 and len(found_by_span_id) >= len(expected_span_ids)
    failure_invariant = ""
    if not overall_ok:
        failure_invariant = (
            span_failures[0]["failure_invariant"]
            if span_failures
            else "span_count_mismatch"
        )

    return {
        "trace_id": trace_id,
        "ok": overall_ok,
        "failure_invariant": failure_invariant,
        "tempo_http_code": fetch_code,
        "tempo_error": fetch_error,
        "grafana_http_code": grafana_code,
        "grafana_error": grafana_error,
        "expected_span_count": len(expected_entries),
        "retrieved_span_count": len(found_by_span_id),
        "write_to_queryable_ms": write_to_queryable_ms,
        "span_failures": span_failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate hard-mode replay output against Tempo trace API"
    )
    parser.add_argument("--manifest", required=True, help="Replay results manifest path")
    parser.add_argument("--tempo-base-url", default="http://localhost:3200")
    parser.add_argument(
        "--grafana-base-url",
        default="http://localhost:3000/api/datasources/proxy/uid/tempo",
    )
    parser.add_argument("--grafana-auth", default="admin:admin")
    parser.add_argument("--query-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=2.0)
    parser.add_argument("--http-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sample-traces",
        type=int,
        default=0,
        help="Optional max trace count to validate (0 means all traces)",
    )
    parser.add_argument("--allow-mismatch", action="store_true")
    args = parser.parse_args()

    replay_manifest = Path(args.manifest)
    replay_payload = replay_manifest.read_text(encoding="utf-8")
    import json

    doc = json.loads(replay_payload)
    entries = list(doc.get("entries") or [])
    if not entries:
        raise SystemExit("manifest has no entries")
    failed_replay = [
        entry for entry in entries if not bool(entry.get("replay_ok", False))
    ]
    if failed_replay:
        failure_output = {
            "schema_version": 1,
            "validator_status": "failed",
            "failure_invariant": "replay_transport_failure",
            "replay_failure_count": len(failed_replay),
            "first_replay_failure": failed_replay[0],
            "trace_results": [],
        }
        write_json(Path(args.output), failure_output)
        print(f"Wrote validation output: {args.output} status=failed replay transport failures")
        if args.allow_mismatch:
            return
        raise SystemExit(2)

    grouped = _group_by_trace(entries)
    trace_ids = sorted(grouped.keys())
    if args.sample_traces > 0:
        trace_ids = trace_ids[: args.sample_traces]

    trace_results: list[dict[str, Any]] = []
    first_failure_trace_id = ""
    first_failure_invariant = ""
    for trace_id in trace_ids:
        result = _verify_trace(
            trace_id=trace_id,
            expected_entries=grouped[trace_id],
            tempo_base_url=args.tempo_base_url,
            grafana_base_url=args.grafana_base_url,
            grafana_auth=args.grafana_auth,
            query_timeout_seconds=args.query_timeout_seconds,
            poll_interval_seconds=args.poll_interval_seconds,
            http_timeout_seconds=args.http_timeout_seconds,
        )
        trace_results.append(result)
        if not result["ok"] and not first_failure_trace_id:
            first_failure_trace_id = trace_id
            first_failure_invariant = str(result.get("failure_invariant", ""))

    ok_count = sum(1 for item in trace_results if item["ok"])
    fail_count = len(trace_results) - ok_count
    write_to_queryable_values = [
        int(item.get("write_to_queryable_ms", -1))
        for item in trace_results
        if int(item.get("write_to_queryable_ms", -1)) >= 0
    ]
    median_write_to_queryable_ms = (
        sorted(write_to_queryable_values)[len(write_to_queryable_values) // 2]
        if write_to_queryable_values
        else -1
    )

    status = "passed" if fail_count == 0 else "failed"
    out = {
        "schema_version": 1,
        "validator_status": status,
        "trace_count": len(trace_results),
        "trace_passes": ok_count,
        "trace_failures": fail_count,
        "first_failure_trace_id": first_failure_trace_id,
        "first_failure_invariant": first_failure_invariant,
        "median_write_to_queryable_ms": median_write_to_queryable_ms,
        "trace_results": trace_results,
    }
    write_json(Path(args.output), out)
    print(
        f"Wrote validation output: {args.output} status={status} "
        f"passes={ok_count} failures={fail_count}"
    )

    if fail_count > 0 and not args.allow_mismatch:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
