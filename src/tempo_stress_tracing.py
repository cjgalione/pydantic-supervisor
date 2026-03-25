"""OTEL tracing helpers used by the Tempo stress demonstration harness."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

_VALID_PAYLOAD_PROFILES = {"baseline", "large", "xlarge"}
_payload_profile_config = {
    "baseline": {"extra_spans": 4, "target_bytes": 4_096},
    "large": {"extra_spans": 30, "target_bytes": 60_000},
    "xlarge": {"extra_spans": 120, "target_bytes": 240_000},
}

_provider: Any | None = None
_tracer = None
_config_signature: tuple[str, str, str] | None = None

_stats_lock = threading.Lock()
_export_stats: dict[str, float | int] = {
    "export_batches": 0,
    "export_successes": 0,
    "export_failures": 0,
    "attempted_spans": 0,
    "exported_spans": 0,
    "failed_spans": 0,
    "estimated_retryable_failures": 0,
    "total_export_seconds": 0.0,
}


class _MonitoringExporter(SpanExporter):
    """Exporter wrapper that tracks coarse export outcomes for experiment reporting."""

    def __init__(self, delegate: SpanExporter):
        self._delegate = delegate

    def export(self, spans):
        started = time.perf_counter()
        span_count = len(spans)
        with _stats_lock:
            _export_stats["export_batches"] += 1
            _export_stats["attempted_spans"] += span_count

        try:
            result = self._delegate.export(spans)
        except Exception:
            result = SpanExportResult.FAILURE

        elapsed = time.perf_counter() - started

        with _stats_lock:
            _export_stats["total_export_seconds"] += elapsed
            if result == SpanExportResult.SUCCESS:
                _export_stats["export_successes"] += 1
                _export_stats["exported_spans"] += span_count
            else:
                _export_stats["export_failures"] += 1
                _export_stats["estimated_retryable_failures"] += 1
                _export_stats["failed_spans"] += span_count

        return result

    def shutdown(self):
        return self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        if hasattr(self._delegate, "force_flush"):
            return bool(self._delegate.force_flush(timeout_millis=timeout_millis))
        return True


def _normalize_payload_profile(payload_profile: str | None) -> str:
    profile = (payload_profile or os.environ.get("TRACE_PAYLOAD_PROFILE", "baseline")).strip().lower()
    return profile if profile in _VALID_PAYLOAD_PROFILES else "baseline"


def get_trace_payload_profile() -> str:
    return _normalize_payload_profile(None)


def _http_export_endpoint(base: str) -> str:
    base = base.rstrip("/")
    if base.endswith("/v1/traces"):
        return base
    return f"{base}/v1/traces"


def _build_exporter(protocol: str, endpoint: str) -> SpanExporter:
    normalized_protocol = protocol.strip().lower()

    if normalized_protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcOTLPSpanExporter,
        )

        return GrpcOTLPSpanExporter(endpoint=endpoint, insecure=True)

    if normalized_protocol in {"http/protobuf", "http"}:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HttpOTLPSpanExporter,
        )

        return HttpOTLPSpanExporter(endpoint=_http_export_endpoint(endpoint))

    raise ValueError(
        "Unsupported OTEL_EXPORTER_OTLP_PROTOCOL. Expected one of: grpc, http/protobuf"
    )


def configure_otel_exporter(*, service_name: str, run_tag: str | None = None) -> None:
    global _provider, _tracer, _config_signature

    protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
    default_endpoint = "http://localhost:4318" if protocol != "grpc" else "http://localhost:4317"
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", default_endpoint)

    signature = (protocol, endpoint, service_name)
    if _config_signature == signature and _provider is not None and _tracer is not None:
        return

    exporter = _build_exporter(protocol=protocol, endpoint=endpoint)
    monitored_exporter = _MonitoringExporter(exporter)

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "pydantic-supervisor",
            "deployment.environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "local"),
            "trace.run_tag": run_tag or os.environ.get("TRACE_RUN_TAG", "adhoc"),
        }
    )

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        monitored_exporter,
        max_queue_size=int(os.environ.get("OTEL_BSP_MAX_QUEUE_SIZE", "8192")),
        max_export_batch_size=int(os.environ.get("OTEL_BSP_MAX_EXPORT_BATCH_SIZE", "1024")),
        schedule_delay_millis=int(os.environ.get("OTEL_BSP_SCHEDULE_DELAY", "500")),
        export_timeout_millis=int(os.environ.get("OTEL_BSP_EXPORT_TIMEOUT", "30000")),
    )
    provider.add_span_processor(processor)

    # The provider can only be globally set once per process. The stress harness
    # is single-process, so this is sufficient and deterministic.
    if _provider is None:
        trace.set_tracer_provider(provider)
        _provider = provider
    else:
        _provider = provider

    _tracer = trace.get_tracer("pydantic_supervisor.tempo_stress", "0.1.0")
    _config_signature = signature


def _fragment_for_step(seed: str, *, step: int, target_chars: int) -> str:
    """Build one deterministic fragment without materializing the full payload."""
    if target_chars <= 0:
        return ""
    if not seed:
        seed = "x"
    if len(seed) >= target_chars:
        offset = step % len(seed)
        rotated = seed[offset:] + seed[:offset]
        return rotated[:target_chars]

    out = []
    remaining = target_chars
    offset = step % len(seed)
    head = seed[offset:] + seed[:offset]
    while remaining > 0:
        chunk = head if out else head
        if len(chunk) <= remaining:
            out.append(chunk)
            remaining -= len(chunk)
        else:
            out.append(chunk[:remaining])
            remaining = 0
        head = seed
    return "".join(out)


def _resolve_target_bytes(payload_profile: str, inject_large_attributes: bool) -> int:
    if not inject_large_attributes:
        return 4_096
    raw_override = str(os.environ.get("TRACE_TARGET_BYTES", "")).strip()
    if raw_override:
        try:
            parsed = int(raw_override)
            if parsed > 0:
                return parsed
        except ValueError:
            pass
    return int(_payload_profile_config[payload_profile]["target_bytes"])


def _resolve_fragment_bytes() -> int:
    # Prefer production-style moderate span payload sizing; keep legacy
    # TRACE_FRAGMENT_TARGET_BYTES as a fallback for compatibility.
    raw = str(
        os.environ.get(
            "TRACE_SPAN_TARGET_BYTES",
            os.environ.get("TRACE_FRAGMENT_TARGET_BYTES", "8192"),
        )
    ).strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 8_192
    return max(256, parsed)


def _resolve_pause_config() -> tuple[int, int]:
    raw_pause_ms = str(os.environ.get("TRACE_SPAN_PAUSE_MS", "0")).strip()
    raw_pause_every = str(os.environ.get("TRACE_SPAN_PAUSE_EVERY", "0")).strip()

    try:
        pause_ms = int(raw_pause_ms)
    except ValueError:
        pause_ms = 0
    try:
        pause_every = int(raw_pause_every)
    except ValueError:
        pause_every = 0

    return max(0, pause_ms), max(0, pause_every)


def _iter_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = str(tool_call.get("name", "") or "").strip()
            if name:
                out.append(name)
    return out


def emit_supervisor_trace(
    *,
    query: str,
    final_output: str,
    messages: list[dict[str, Any]],
    payload_profile: str,
    inject_large_attributes: bool,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _tracer is None:
        return {"trace_id": "", "synthetic_spans": 0, "payload_bytes": 0}

    profile = _normalize_payload_profile(payload_profile)
    profile_cfg = _payload_profile_config[profile]
    target_payload_bytes = _resolve_target_bytes(profile, inject_large_attributes)
    fragment_target_bytes = _resolve_fragment_bytes()
    pause_ms, pause_every = _resolve_pause_config()
    payload_seed = json.dumps(
        {
            "query": query,
            "final_output": final_output,
            "messages": messages,
            "metadata": metadata or {},
        },
        ensure_ascii=False,
    )

    tool_names = _iter_tool_names(messages)
    min_spans = int(profile_cfg["extra_spans"])
    synthetic_spans = max(min_spans, (target_payload_bytes + fragment_target_bytes - 1) // fragment_target_bytes)
    root_payload_preview = payload_seed[:4096]

    root_attributes: dict[str, Any] = {
        "span.kind": "agent_root",
        "trace.payload_profile": profile,
        "trace.inject_large_attributes": inject_large_attributes,
        "trace.target_payload_bytes": target_payload_bytes,
        "trace.fragment_target_bytes": fragment_target_bytes,
        "trace.pause_ms": pause_ms,
        "trace.pause_every_spans": pause_every,
        "agent.query": query[:4096],
        "agent.final_output": final_output[:4096],
        "agent.tool_call_count": len(tool_names),
        "agent.payload_preview": root_payload_preview,
    }
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, (str, bool, int, float)):
                root_attributes[f"trace.meta.{key}"] = value

    with _tracer.start_as_current_span(
        "invocation [supervisor_with_critic]", attributes=root_attributes
    ) as root_span:
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        with _tracer.start_as_current_span(
            "llm_response_generation",
            attributes={
                "llm.message_count": len(messages),
                "llm.synthetic_payload_chars": target_payload_bytes,
            },
        ):
            pass

        for idx, tool_name in enumerate(tool_names[:50]):
            with _tracer.start_as_current_span(
                f"tool_routing_decision [{tool_name}]",
                attributes={"tool.name": tool_name, "tool.index": idx},
            ):
                pass

        emitted_payload_bytes = 0
        for idx in range(synthetic_spans):
            if pause_ms > 0 and pause_every > 0 and idx > 0 and idx % pause_every == 0:
                time.sleep(pause_ms / 1000.0)
            remaining_bytes = max(0, target_payload_bytes - emitted_payload_bytes)
            if remaining_bytes <= 0:
                fragment_chars = 0
            else:
                fragment_chars = min(fragment_target_bytes, remaining_bytes)
            fragment = _fragment_for_step(payload_seed, step=idx, target_chars=fragment_chars)
            emitted_payload_bytes += len(fragment.encode("utf-8"))
            with _tracer.start_as_current_span(
                f"synthetic_step_{idx:03d}",
                attributes={
                    "synthetic.step": idx,
                    "synthetic.fragment": fragment,
                    "synthetic.fragment_chars": len(fragment),
                },
            ):
                pass

    return {
        "trace_id": trace_id,
        "synthetic_spans": synthetic_spans,
        "payload_bytes": target_payload_bytes,
    }


def get_export_stats() -> dict[str, float | int]:
    with _stats_lock:
        return dict(_export_stats)


def force_flush() -> None:
    if _provider is not None:
        _provider.force_flush()
