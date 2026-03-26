"""Hard-mode OTLP helpers for Tempo stress campaigns.

This module intentionally bypasses OpenTelemetry SDK span-limits by building
OTLP protobuf payloads directly.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import grpc
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.proto.common.v1 import common_pb2
from opentelemetry.proto.resource.v1 import resource_pb2
from opentelemetry.proto.trace.v1 import trace_pb2

DEFAULT_SPAN_SIZE_CLASSES = [
    100 * 1024,
    500 * 1024,
    1 * 1024 * 1024,
    10 * 1024 * 1024,
    50 * 1024 * 1024,
    100 * 1024 * 1024,
]

DEFAULT_CHUNK_SIZE_BYTES = 1 * 1024 * 1024
PAYLOAD_CHUNK_KEY_PREFIX = "hardmode.payload.chunk."
PAYLOAD_TEXT_FORMAT_VERSION = "v2"


def now_unix_ms() -> int:
    import time

    return int(time.time() * 1000)


def stable_hex_id(*parts: str, hex_len: int) -> str:
    digest = hashlib.sha256(":".join(parts).encode("utf-8")).hexdigest()
    return digest[:hex_len]


def trace_id_for(run_tag: str, stage_name: str, trace_index: int) -> str:
    return stable_hex_id(run_tag, stage_name, "trace", str(trace_index), hex_len=32)


def span_id_for(run_tag: str, stage_name: str, trace_index: int, span_index: int) -> str:
    return stable_hex_id(
        run_tag, stage_name, "span", str(trace_index), str(span_index), hex_len=16
    )


def deterministic_payload_text(size_bytes: int, seed: str) -> str:
    if size_bytes <= 0:
        return ""
    # ASCII-only deterministic prose so bytes == chars in UTF-8, while still
    # resembling real agent-context content.
    digest = hashlib.sha256(f"{seed}:{size_bytes}".encode("utf-8")).hexdigest()
    template = (
        f"payload_format={PAYLOAD_TEXT_FORMAT_VERSION} "
        f"seed={seed} "
        "This synthetic pydantic supervisor payload captures nested context for "
        "checkout and death benefit triage with retrieval evidence, tool args, "
        "and planning notes. "
        "The user asks for checkout status and death claim guidance while the "
        "assistant validates steps and cites policy docs. "
        "context.thread.summary mentions checkout, death, latency, and replay. "
        "retrieval.evidence.snippet contains policy constraints and citations. "
        f"digest={digest} "
    )
    repeats = (size_bytes + len(template) - 1) // len(template)
    return (template * repeats)[:size_bytes]


def payload_sha256_hex(payload_text: str) -> str:
    return hashlib.sha256(payload_text.encode("utf-8")).hexdigest()


def chunk_payload_text(payload_text: str, chunk_size_bytes: int) -> list[str]:
    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be > 0")
    if not payload_text:
        return []
    out: list[str] = []
    for i in range(0, len(payload_text), chunk_size_bytes):
        out.append(payload_text[i : i + chunk_size_bytes])
    return out


def canonical_blob_path(
    canonical_dir: Path,
    span_payload_target_bytes: int,
    payload_seed: str,
) -> Path:
    token = hashlib.sha256(
        f"{payload_seed}:{PAYLOAD_TEXT_FORMAT_VERSION}".encode("utf-8")
    ).hexdigest()[:12]
    return canonical_dir / f"payload_{span_payload_target_bytes}_{token}.txt"


def ensure_canonical_blob(
    canonical_dir: Path,
    span_payload_target_bytes: int,
    payload_seed: str,
) -> tuple[Path, str]:
    canonical_dir.mkdir(parents=True, exist_ok=True)
    path = canonical_blob_path(canonical_dir, span_payload_target_bytes, payload_seed)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if len(text.encode("utf-8")) == span_payload_target_bytes:
            return path, payload_sha256_hex(text)
    text = deterministic_payload_text(span_payload_target_bytes, payload_seed)
    path.write_text(text, encoding="utf-8")
    return path, payload_sha256_hex(text)


def _kv_string(key: str, value: str) -> common_pb2.KeyValue:
    return common_pb2.KeyValue(
        key=key,
        value=common_pb2.AnyValue(string_value=value),
    )


def _kv_int(key: str, value: int) -> common_pb2.KeyValue:
    return common_pb2.KeyValue(
        key=key,
        value=common_pb2.AnyValue(int_value=int(value)),
    )


def _kv_string_array(key: str, values: list[str]) -> common_pb2.KeyValue:
    return common_pb2.KeyValue(
        key=key,
        value=common_pb2.AnyValue(
            array_value=common_pb2.ArrayValue(
                values=[common_pb2.AnyValue(string_value=v) for v in values]
            )
        ),
    )


def _query_seed_terms(entry: dict[str, Any]) -> dict[str, str]:
    base = hashlib.sha256(
        (
            f"{entry.get('run_tag', '')}:{entry.get('trace_id', '')}:"
            f"{entry.get('span_id', '')}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "lvl1": f"death_l1_{base[:12]}",
        "lvl2": f"death_l2_{base[12:24]}",
        "lvl3": f"death_l3_{base[24:36]}",
        "lvl4": f"death_l4_{base[36:48]}",
        "json": f"death_json_{base[48:60]}",
    }


def seed_terms_for_entry(entry: dict[str, Any]) -> dict[str, str]:
    terms = _query_seed_terms(entry)
    return {
        "seed_term_level_1": terms["lvl1"],
        "seed_term_level_2": terms["lvl2"],
        "seed_term_level_3": terms["lvl3"],
        "seed_term_level_4": terms["lvl4"],
        "seed_term_json": terms["json"],
    }


def build_span_attributes(
    *,
    entry: dict[str, Any],
    payload_chunks: list[str],
) -> list[common_pb2.KeyValue]:
    seeds = _query_seed_terms(entry)
    nested_blob = json.dumps(
        {
            "bt": {
                "prompt": {"system": {"seed_term": seeds["lvl1"]}},
                "context": {
                    "thread_summary": {"seed_term": seeds["lvl2"]},
                    "tool_args": {
                        "filters": {"primary": {"seed_term": seeds["lvl3"]}},
                        "inputs": ["checkout", "death", str(entry["run_tag"])],
                    },
                    "seed_term": seeds["json"],
                },
                "retrieval": {
                    "evidence": {
                        "documents": {
                            "primary": {"snippet": {"seed_term": seeds["lvl4"]}}
                        }
                    }
                },
            }
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
    attrs: list[common_pb2.KeyValue] = [
        _kv_string("stress_run_tag", str(entry["run_tag"])),
        _kv_string("hardmode.stage_name", str(entry["stage_name"])),
        _kv_string("hardmode.trace_id", str(entry["trace_id"])),
        _kv_string("hardmode.span_id", str(entry["span_id"])),
        _kv_string("hardmode.payload.sha256", str(entry["payload_sha256"])),
        _kv_int("hardmode.payload.size_bytes", int(entry["span_payload_target_bytes"])),
        _kv_int("hardmode.payload.chunk_count", int(entry["chunk_count"])),
        _kv_int("hardmode.payload.chunk_size_bytes", int(entry["chunk_size_bytes"])),
        _kv_int("hardmode.trace_index", int(entry["trace_index"])),
        _kv_int("hardmode.span_index", int(entry["span_index"])),
        # Duplicate marker attrs with underscore keys so TraceQL field references are simple.
        _kv_string("hardmode_stage_name", str(entry["stage_name"])),
        _kv_string("hardmode_trace_id", str(entry["trace_id"])),
        _kv_string("hardmode_span_id", str(entry["span_id"])),
        _kv_string("hardmode_payload_sha256", str(entry["payload_sha256"])),
        _kv_int("hardmode_payload_size_bytes", int(entry["span_payload_target_bytes"])),
        _kv_int("hardmode_payload_chunk_count", int(entry["chunk_count"])),
        _kv_int("hardmode_payload_chunk_size_bytes", int(entry["chunk_size_bytes"])),
        _kv_int("hardmode_trace_index", int(entry["trace_index"])),
        _kv_int("hardmode_span_index", int(entry["span_index"])),
        # Mirror pydantic-supervisor style nested keys (dot-delimited) and
        # plant deterministic seed terms for depth-aware query checks.
        _kv_string("bt.prompt.system.seed_term", seeds["lvl1"]),
        _kv_string("bt.context.thread_summary.seed_term", seeds["lvl2"]),
        _kv_string("bt.context.tool_args.filters.primary.seed_term", seeds["lvl3"]),
        _kv_string("bt.retrieval.evidence.documents.primary.snippet.seed_term", seeds["lvl4"]),
        _kv_string("bt.context.serialized", nested_blob),
        _kv_string("hardmode_query_seed_level_1", seeds["lvl1"]),
        _kv_string("hardmode_query_seed_level_2", seeds["lvl2"]),
        _kv_string("hardmode_query_seed_level_3", seeds["lvl3"]),
        _kv_string("hardmode_query_seed_level_4", seeds["lvl4"]),
        _kv_string("hardmode_query_seed_json", seeds["json"]),
        _kv_string_array(
            "span.tags",
            [
                "checkout",
                "death",
                str(entry["stage_name"]),
                str(entry["run_tag"]),
            ],
        ),
    ]
    for idx, chunk in enumerate(payload_chunks):
        attrs.append(_kv_string(f"{PAYLOAD_CHUNK_KEY_PREFIX}{idx:04d}", chunk))
    return attrs


def build_export_request_for_entries(
    *,
    entries: list[dict[str, Any]],
    payload_chunks_by_sha: dict[str, list[str]],
    service_name: str,
    instrumentation_scope_name: str = "tempo-hardmode",
    instrumentation_scope_version: str = "1.0.0",
    start_unix_nano: int | None = None,
    span_duration_nano: int = 1_000_000,
) -> trace_service_pb2.ExportTraceServiceRequest:
    if not entries:
        raise ValueError("entries must not be empty")

    import time

    base_ts = int(start_unix_nano if start_unix_nano is not None else time.time_ns())

    spans: list[trace_pb2.Span] = []
    first_run_tag = str(entries[0]["run_tag"])
    first_stage = str(entries[0]["stage_name"])
    first_trace_id = str(entries[0]["trace_id"])

    for idx, entry in enumerate(entries):
        payload_sha = str(entry["payload_sha256"])
        payload_chunks = payload_chunks_by_sha.get(payload_sha)
        if payload_chunks is None:
            raise KeyError(f"missing payload chunks for sha {payload_sha}")
        start_ts = base_ts + idx * span_duration_nano
        span = trace_pb2.Span(
            trace_id=bytes.fromhex(str(entry["trace_id"])),
            span_id=bytes.fromhex(str(entry["span_id"])),
            name=str(entry["span_name"]),
            kind=trace_pb2.Span.SPAN_KIND_INTERNAL,
            start_time_unix_nano=start_ts,
            end_time_unix_nano=start_ts + span_duration_nano,
            attributes=build_span_attributes(entry=entry, payload_chunks=payload_chunks),
        )
        spans.append(span)

    scope_spans = trace_pb2.ScopeSpans(
        scope=common_pb2.InstrumentationScope(
            name=instrumentation_scope_name,
            version=instrumentation_scope_version,
        ),
        spans=spans,
    )
    resource = resource_pb2.Resource(
        attributes=[
            _kv_string("service.name", service_name),
            _kv_string("service.namespace", "pydantic-supervisor"),
            _kv_string("deployment.environment", "stress-hardmode"),
            _kv_string("stress_run_tag", first_run_tag),
            _kv_string("hardmode.stage_name", first_stage),
            _kv_string("hardmode.trace_id", first_trace_id),
        ]
    )
    resource_spans = trace_pb2.ResourceSpans(resource=resource, scope_spans=[scope_spans])
    return trace_service_pb2.ExportTraceServiceRequest(resource_spans=[resource_spans])


def compute_wire_request_size_bytes(
    *,
    entry: dict[str, Any],
    payload_chunks: list[str],
    service_name: str,
) -> int:
    req = build_export_request_for_entries(
        entries=[entry],
        payload_chunks_by_sha={str(entry["payload_sha256"]): payload_chunks},
        service_name=service_name,
    )
    return int(req.ByteSize())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_from_path_or_s3(
    manifest_path: str,
    *,
    aws_region: str | None = None,
) -> dict[str, Any]:
    if manifest_path.startswith("s3://"):
        from subprocess import run

        with tempfile.NamedTemporaryFile(prefix="tempo-hardmode-manifest-", suffix=".json") as tmp:
            cmd = ["aws", "s3", "cp", manifest_path, tmp.name]
            if aws_region:
                cmd.extend(["--region", aws_region])
            proc = run(cmd, check=False, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"failed to download manifest from s3: {proc.stderr.strip() or proc.stdout.strip()}"
                )
            return json.loads(Path(tmp.name).read_text(encoding="utf-8"))
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))


def resolve_blob_text(
    *,
    entry: dict[str, Any],
    payload_seed: str,
    fallback_canonical_dir: Path | None = None,
) -> str:
    blob_path_raw = str(entry.get("canonical_blob_path", "") or "")
    if blob_path_raw:
        p = Path(blob_path_raw)
        if p.exists():
            return p.read_text(encoding="utf-8")
        if fallback_canonical_dir is not None:
            candidate = fallback_canonical_dir / p.name
            if candidate.exists():
                return candidate.read_text(encoding="utf-8")
    return deterministic_payload_text(int(entry["span_payload_target_bytes"]), payload_seed)


def _decode_any_value(value: dict[str, Any]) -> str:
    if "stringValue" in value and isinstance(value["stringValue"], str):
        return value["stringValue"]
    if "intValue" in value:
        return str(value["intValue"])
    if "boolValue" in value:
        return str(value["boolValue"]).lower()
    if "doubleValue" in value:
        return str(value["doubleValue"])
    if "bytesValue" in value and isinstance(value["bytesValue"], str):
        return value["bytesValue"]
    if "arrayValue" in value and isinstance(value["arrayValue"], dict):
        values = value["arrayValue"].get("values")
        if isinstance(values, list):
            parts: list[str] = []
            for item in values:
                if isinstance(item, dict):
                    parts.append(_decode_any_value(item))
            return ",".join(parts)
    return ""


def span_attributes_map(span_obj: dict[str, Any]) -> dict[str, str]:
    attrs = span_obj.get("attributes")
    out: dict[str, str] = {}
    if not isinstance(attrs, list):
        return out
    for attr in attrs:
        if not isinstance(attr, dict):
            continue
        key = attr.get("key")
        value = attr.get("value")
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        out[key] = _decode_any_value(value)
    return out


def extract_span_objects(payload: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if (
                ("traceId" in node or "trace_id" in node)
                and ("spanId" in node or "span_id" in node)
                and isinstance(node.get("attributes"), list)
            ):
                out.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return out


def reconstruct_payload_from_attrs(attrs: dict[str, str]) -> tuple[str, int]:
    chunk_items: list[tuple[int, str]] = []
    for key, value in attrs.items():
        if not key.startswith(PAYLOAD_CHUNK_KEY_PREFIX):
            continue
        suffix = key[len(PAYLOAD_CHUNK_KEY_PREFIX) :]
        try:
            idx = int(suffix)
        except ValueError:
            continue
        chunk_items.append((idx, value))
    chunk_items.sort(key=lambda item: item[0])
    payload_text = "".join(item[1] for item in chunk_items)
    return payload_text, len(chunk_items)


def fetch_json(
    url: str,
    *,
    timeout_seconds: float = 30.0,
    basic_auth: str | None = None,
) -> tuple[int, dict[str, Any] | None, str]:
    headers = {"Accept": "application/json"}
    if basic_auth:
        import base64

        encoded = base64.b64encode(basic_auth.encode("utf-8")).decode("ascii")
        headers["Authorization"] = f"Basic {encoded}"
    req = urllib.request.Request(url=url, method="GET", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            code = int(resp.getcode() or 0)
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        code = int(exc.code or 0)
        body = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return 0, None, str(exc)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        parsed = None
    return code, parsed, body


def traceql_search_probe(
    *,
    base_url: str,
    traceql: str,
    limit: int = 20,
    timeout_seconds: float = 10.0,
    basic_auth: str | None = None,
) -> dict[str, Any]:
    import time

    query = urllib.parse.urlencode({"q": traceql, "limit": str(limit)})
    url = f"{base_url.rstrip('/')}/api/search?{query}"
    started = time.perf_counter()
    code, parsed, _body = fetch_json(
        url,
        timeout_seconds=timeout_seconds,
        basic_auth=basic_auth,
    )
    elapsed = time.perf_counter() - started

    trace_count = 0
    if parsed is not None:
        spans = extract_span_objects(parsed)
        # We only need a coarse signal of non-empty search result; count trace ids.
        seen: set[str] = set()
        for span_obj in spans:
            trace_id = span_obj.get("traceId") or span_obj.get("trace_id")
            if isinstance(trace_id, str):
                seen.add(trace_id)
        trace_count = len(seen)
        if trace_count == 0:
            # Tempo search responses are not span objects; this path counts traces recursively.
            def walk(node: Any) -> None:
                nonlocal trace_count
                if isinstance(node, dict):
                    trace_id = node.get("traceID") or node.get("traceId")
                    if isinstance(trace_id, str):
                        trace_count += 1
                    for value in node.values():
                        walk(value)
                elif isinstance(node, list):
                    for item in node:
                        walk(item)

            walk(parsed)

    return {
        "http_code": code,
        "time_total_seconds": elapsed,
        "trace_count": trace_count,
    }


def grpc_export_request(
    *,
    endpoint: str,
    request_payload: trace_service_pb2.ExportTraceServiceRequest,
    timeout_seconds: float,
    max_message_bytes: int,
    compression: str = "none",
) -> tuple[bool, str]:
    if compression == "gzip":
        grpc_compression = grpc.Compression.Gzip
    else:
        grpc_compression = grpc.Compression.NoCompression

    options = [
        ("grpc.max_send_message_length", int(max_message_bytes)),
        ("grpc.max_receive_message_length", int(max_message_bytes)),
    ]
    channel = grpc.insecure_channel(
        endpoint,
        options=options,
        compression=grpc_compression,
    )
    stub = trace_service_pb2_grpc.TraceServiceStub(channel)
    try:
        stub.Export(request_payload, timeout=timeout_seconds)
        return True, ""
    except grpc.RpcError as exc:
        return False, f"{exc.code().name}: {exc.details() or ''}".strip()
    except Exception as exc:
        return False, str(exc)
    finally:
        channel.close()
