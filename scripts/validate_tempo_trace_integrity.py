#!/usr/bin/env python3
"""Validate replayed hard-mode traces against Tempo API source of truth."""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path
import sys
from typing import Any
import urllib.parse

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


def _extract_trace_ids(payload: Any) -> set[str]:
    trace_ids: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            trace_id = node.get("traceID") or node.get("traceId")
            if isinstance(trace_id, str):
                trace_ids.add(trace_id)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return trace_ids


def _traceql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _query_traceql_search(
    *,
    base_url: str,
    traceql: str,
    http_timeout_seconds: float,
    basic_auth: str | None = None,
) -> tuple[int, set[str], str]:
    query = urllib.parse.urlencode({"q": traceql, "limit": "20"})
    code, parsed, raw = fetch_json(
        f"{base_url.rstrip('/')}/api/search?{query}",
        timeout_seconds=http_timeout_seconds,
        basic_auth=basic_auth,
    )
    trace_ids = _extract_trace_ids(parsed) if parsed is not None else set()
    return code, trace_ids, (raw or "")[:300]


def _span_traceql_candidates(entry: dict[str, Any]) -> list[str]:
    # Primary query uses underscore marker attrs added for easier TraceQL references.
    # Fallback query uses dotted attrs for compatibility with older datasets.
    span_id = _traceql_escape(str(entry["span_id"]))
    payload_sha = _traceql_escape(str(entry["payload_sha256"]))
    run_tag = _traceql_escape(str(entry["run_tag"]))
    return [
        (
            f'{{.hardmode_span_id="{span_id}" '
            f'&& .hardmode_payload_sha256="{payload_sha}" '
            f'&& .stress_run_tag="{run_tag}"}}'
        ),
        (
            f'{{.hardmode.span_id="{span_id}" '
            f'&& .hardmode.payload.sha256="{payload_sha}" '
            f'&& .stress_run_tag="{run_tag}"}}'
        ),
    ]


def _seed_traceql_checks(entry: dict[str, Any]) -> list[dict[str, Any]]:
    run_tag = _traceql_escape(str(entry.get("run_tag", "")))
    seed_l1 = _traceql_escape(str(entry.get("seed_term_level_1", "")))
    seed_l2 = _traceql_escape(str(entry.get("seed_term_level_2", "")))
    seed_l3 = _traceql_escape(str(entry.get("seed_term_level_3", "")))
    seed_l4 = _traceql_escape(str(entry.get("seed_term_level_4", "")))
    seed_json = _traceql_escape(str(entry.get("seed_term_json", "")))
    checks: list[dict[str, Any]] = []
    if not run_tag:
        return checks
    if seed_l1:
        checks.append(
            {
                "name": "seed_level_1",
                "traceql": (
                    f'{{span."bt.prompt.system.seed_term"="{seed_l1}" '
                    f'&& .stress_run_tag="{run_tag}"}}'
                ),
                "required": True,
            }
        )
    if seed_l2:
        checks.append(
            {
                "name": "seed_level_2",
                "traceql": (
                    f'{{span."bt.context.thread_summary.seed_term"="{seed_l2}" '
                    f'&& .stress_run_tag="{run_tag}"}}'
                ),
                "required": True,
            }
        )
    if seed_l3:
        checks.append(
            {
                "name": "seed_level_3",
                "traceql": (
                    f'{{span."bt.context.tool_args.filters.primary.seed_term"="{seed_l3}" '
                    f'&& .stress_run_tag="{run_tag}"}}'
                ),
                "required": True,
            }
        )
    if seed_l4:
        checks.append(
            {
                "name": "seed_level_4",
                "traceql": (
                    f'{{span."bt.retrieval.evidence.documents.primary.snippet.seed_term"="{seed_l4}" '
                    f'&& .stress_run_tag="{run_tag}"}}'
                ),
                "required": True,
            }
        )
    if seed_json:
        checks.append(
            {
                "name": "seed_nested_json",
                "traceql": (
                    f'{{span."bt.context.serialized"=~".*{seed_json}.*" '
                    f'&& .stress_run_tag="{run_tag}"}}'
                ),
                "required": True,
            }
        )
    checks.append(
        {
            "name": "seed_array_tags_death",
            "traceql": f'{{span.tags="death" && .stress_run_tag="{run_tag}"}}',
            "required": False,
        }
    )
    return checks


def _validate_seed_queries(
    *,
    trace_id: str,
    entry: dict[str, Any],
    tempo_base_url: str,
    query_timeout_seconds: float,
    poll_interval_seconds: float,
    http_timeout_seconds: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    checks = _seed_traceql_checks(entry)
    if not checks:
        return [], []
    out: list[dict[str, Any]] = []
    failed_names: list[str] = []
    deadline_ms = int(time.time() * 1000) + int(query_timeout_seconds * 1000)
    for check in checks:
        required = bool(check.get("required", True))
        last_code = 0
        last_err = ""
        matched = False
        attempts = 0
        check_deadline_ms = deadline_ms if required else int(time.time() * 1000)
        while int(time.time() * 1000) <= check_deadline_ms:
            attempts += 1
            last_code, trace_ids, tempo_err = _query_traceql_search(
                base_url=tempo_base_url,
                traceql=str(check["traceql"]),
                http_timeout_seconds=http_timeout_seconds,
            )
            if last_code == 200 and trace_id in trace_ids:
                matched = True
                last_err = ""
                break
            last_err = tempo_err
            if not required:
                break
            time.sleep(poll_interval_seconds)
        out.append(
            {
                "name": str(check["name"]),
                "matched": matched,
                "http_code": last_code,
                "attempts": attempts,
                "error": last_err,
                "traceql": str(check["traceql"]),
                "required": required,
            }
        )
        if not matched and required:
            failed_names.append(str(check["name"]))
    return out, failed_names


def _attach_seed_query_results(
    *,
    result: dict[str, Any],
    trace_id: str,
    expected_entries: list[dict[str, Any]],
    tempo_base_url: str,
    query_timeout_seconds: float,
    poll_interval_seconds: float,
    http_timeout_seconds: float,
) -> dict[str, Any]:
    checks, failed_names = _validate_seed_queries(
        trace_id=trace_id,
        entry=expected_entries[0],
        tempo_base_url=tempo_base_url,
        query_timeout_seconds=query_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        http_timeout_seconds=http_timeout_seconds,
    )
    result["seed_query_checks"] = checks
    result["seed_query_failure_count"] = len(failed_names)
    if failed_names:
        result["seed_query_failed_names"] = failed_names
        if result.get("ok"):
            result["ok"] = False
            result["failure_invariant"] = "seed_query_not_match"
    else:
        result["seed_query_failed_names"] = []
    return result


def _verify_trace_via_span_search(
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
    expected_by_span = {str(entry["span_id"]): entry for entry in expected_entries}
    emitted_latest_ms = max(
        int(entry.get("emitted_at_unix_ms", 0) or 0) for entry in expected_entries
    )

    started_ms = int(time.time() * 1000)
    deadline_ms = started_ms + int(query_timeout_seconds * 1000)

    missing = set(expected_span_ids)
    found_at_ms: dict[str, int] = {}
    tempo_last_code = 0
    tempo_last_error = ""
    grafana_last_code = 0
    grafana_last_error = ""

    while missing and int(time.time() * 1000) <= deadline_ms:
        for span_id in list(missing):
            entry = expected_by_span[span_id]
            expected_trace_id = str(entry["trace_id"])
            found_in_tempo = False

            for traceql in _span_traceql_candidates(entry):
                tempo_last_code, trace_ids, tempo_err = _query_traceql_search(
                    base_url=tempo_base_url,
                    traceql=traceql,
                    http_timeout_seconds=http_timeout_seconds,
                )
                if tempo_last_code != 200:
                    tempo_last_error = tempo_err
                    continue
                if expected_trace_id in trace_ids:
                    found_in_tempo = True
                    break

            if not found_in_tempo:
                continue

            # Optional secondary check against Grafana proxy path; non-blocking for pass/fail.
            if grafana_base_url:
                traceql = _span_traceql_candidates(entry)[0]
                grafana_last_code, grafana_trace_ids, grafana_err = _query_traceql_search(
                    base_url=grafana_base_url,
                    traceql=traceql,
                    http_timeout_seconds=http_timeout_seconds,
                    basic_auth=grafana_auth if grafana_auth else None,
                )
                if grafana_last_code != 200 or expected_trace_id not in grafana_trace_ids:
                    grafana_last_error = grafana_err

            found_at_ms[span_id] = int(time.time() * 1000)
            missing.remove(span_id)

        if missing and int(time.time() * 1000) <= deadline_ms:
            time.sleep(poll_interval_seconds)

    span_failures = [
        {
            "span_id": span_id,
            "failure_invariant": "span_not_queryable",
            "details": "",
        }
        for span_id in sorted(missing)
    ]
    overall_ok = len(span_failures) == 0
    write_to_queryable_ms = (
        max(found_at_ms.values()) - emitted_latest_ms
        if overall_ok and found_at_ms and emitted_latest_ms > 0
        else -1
    )

    return {
        "trace_id": trace_id,
        "ok": overall_ok,
        "validation_mode": "span_search",
        "failure_invariant": (span_failures[0]["failure_invariant"] if span_failures else ""),
        "tempo_http_code": tempo_last_code,
        "tempo_error": tempo_last_error,
        "grafana_http_code": grafana_last_code,
        "grafana_error": grafana_last_error,
        "expected_span_count": len(expected_entries),
        "retrieved_span_count": len(found_at_ms),
        "write_to_queryable_ms": write_to_queryable_ms,
        "span_failures": span_failures,
    }


def _verify_trace_via_trace_fetch(
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
        "validation_mode": "trace_fetch",
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


def _verify_trace(
    *,
    validation_mode: str,
    trace_id: str,
    expected_entries: list[dict[str, Any]],
    tempo_base_url: str,
    grafana_base_url: str,
    grafana_auth: str,
    query_timeout_seconds: float,
    poll_interval_seconds: float,
    http_timeout_seconds: float,
) -> dict[str, Any]:
    if validation_mode == "span_search":
        result = _verify_trace_via_span_search(
            trace_id=trace_id,
            expected_entries=expected_entries,
            tempo_base_url=tempo_base_url,
            grafana_base_url=grafana_base_url,
            grafana_auth=grafana_auth,
            query_timeout_seconds=query_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            http_timeout_seconds=http_timeout_seconds,
        )
    else:
        result = _verify_trace_via_trace_fetch(
            trace_id=trace_id,
            expected_entries=expected_entries,
            tempo_base_url=tempo_base_url,
            grafana_base_url=grafana_base_url,
            grafana_auth=grafana_auth,
            query_timeout_seconds=query_timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
            http_timeout_seconds=http_timeout_seconds,
        )
    return _attach_seed_query_results(
        result=result,
        trace_id=trace_id,
        expected_entries=expected_entries,
        tempo_base_url=tempo_base_url,
        query_timeout_seconds=query_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        http_timeout_seconds=http_timeout_seconds,
    )


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
    parser.add_argument(
        "--validation-mode",
        choices=("trace_fetch", "span_search"),
        default="trace_fetch",
        help="trace_fetch: full /api/traces/<id> integrity; span_search: each span queryable by marker attrs",
    )
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
            validation_mode=args.validation_mode,
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
        "validation_mode": args.validation_mode,
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
