"""OTEL tracing helpers used by the Tempo stress demonstration harness."""

from __future__ import annotations

import json
import os
import re
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
    from opentelemetry.sdk.trace import SpanLimits, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "pydantic-supervisor",
            "deployment.environment": os.environ.get("DEPLOYMENT_ENVIRONMENT", "local"),
            "trace.run_tag": run_tag or os.environ.get("TRACE_RUN_TAG", "adhoc"),
        }
    )

    # Default OTEL limits (especially max_attributes=128) are too low for stress
    # traces with heavy structured attributes; raise safely for this harness.
    span_limits = SpanLimits(
        max_attributes=int(os.environ.get("TRACE_SPAN_ATTRIBUTE_LIMIT", "50000")),
        max_attribute_length=int(os.environ.get("TRACE_SPAN_ATTRIBUTE_VALUE_LIMIT", "131072")),
        max_events=int(os.environ.get("TRACE_SPAN_EVENT_LIMIT", "2048")),
        max_links=int(os.environ.get("TRACE_SPAN_LINK_LIMIT", "2048")),
    )
    provider = TracerProvider(resource=resource, span_limits=span_limits)
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


def _seed_segment(seed: str, *, step: int, slot: int, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    if not seed:
        seed = "x"
    offset = (step * 131 + slot * 37) % len(seed)
    rotated = seed[offset:] + seed[:offset]
    if len(rotated) >= target_chars:
        return rotated[:target_chars]
    repeats = (target_chars + len(rotated) - 1) // len(rotated)
    return (rotated * repeats)[:target_chars]


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
    # Permit large span payload experiments up to 6 MiB per span.
    return min(max(256, parsed), 6 * 1024 * 1024)


def _attrs_size_bytes(attrs: dict[str, Any]) -> int:
    return len(json.dumps(attrs, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _search_terms(text: str, *, limit: int = 24) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in re.findall(r"[a-zA-Z0-9_]+", text.lower()):
        if len(term) < 3:
            continue
        if term in seen:
            continue
        seen.add(term)
        out.append(term)
        if len(out) >= limit:
            break
    return out


def _base_turn_attributes(
    *,
    step: int,
    synthetic_spans: int,
    query: str,
    final_output: str,
    tool_names: list[str],
    seed: str,
    run_tag: str,
) -> dict[str, Any]:
    tool_name = tool_names[step % len(tool_names)] if tool_names else "delegate_to_research_agent"
    has_error = (step % 29) == 0 and step > 0
    attrs = {
        "bt.span.type": "llm",
        "bt.trace.component": "pydantic-supervisor",
        "bt.trace.turn_index": step,
        "bt.trace.turn_count": synthetic_spans,
        "bt.trace.turn_role": "assistant",
        "bt.agent.name": "supervisor_with_critic",
        "bt.agent.phase": "single_turn_conversation",
        "bt.model.provider": "google",
        "bt.model.name": "gemini-2.0-flash-lite",
        "bt.model.temperature": 0.1,
        "bt.model.max_output_tokens": 1024,
        "bt.input.user_message": query[:512],
        "bt.output.assistant_message": final_output[:512],
        "bt.tool.name": tool_name,
        "bt.tool.invocation_index": step % max(1, len(tool_names)),
        "bt.metrics.prompt_tokens": 800 + (step % 120),
        "bt.metrics.completion_tokens": 200 + (step % 90),
        "bt.metrics.total_tokens": 1000 + (step % 210),
        "bt.metrics.latency_ms": 250 + (step % 350),
        "bt.metrics.queue_ms": 5 + (step % 12),
        "bt.error.present": has_error,
        "bt.error.type": "none" if not has_error else "tool_timeout",
        "bt.error.retriable": bool(has_error and step % 2 == 0),
        "bt.turn.synthetic_seed": _seed_segment(seed, step=step, slot=0, target_chars=128),
        "stress_run_tag": run_tag,
    }
    terms = _search_terms(f"{query} {final_output} {tool_name}", limit=24)
    for idx, term in enumerate(terms):
        attrs[f"bt.search.term.{idx:02d}"] = term
    attrs["bt.search.term_count"] = len(terms)
    return attrs


def _expanded_turn_attributes(
    *,
    step: int,
    target_span_bytes: int,
    seed: str,
    base: dict[str, Any],
) -> dict[str, Any]:
    attrs = dict(base)
    if _attrs_size_bytes(attrs) >= target_span_bytes:
        return attrs

    key_specs = [
        ("bt.prompt.system", 1024),
        ("bt.prompt.instructions", 1024),
        ("bt.prompt.guardrails", 1024),
        ("bt.prompt.examples", 1024),
        ("bt.context.thread_summary", 1536),
        ("bt.context.research_notes", 1536),
        ("bt.context.math_notes", 1536),
        ("bt.context.tool_args", 1536),
        ("bt.context.tool_result", 1536),
        ("bt.context.validation", 1024),
        ("bt.context.observations", 1024),
        ("bt.context.candidate_answers", 1024),
        ("bt.retrieval.query_plan", 1024),
        ("bt.retrieval.query_terms", 1024),
        ("bt.retrieval.evidence", 1536),
        ("bt.reasoning.plan", 1024),
        ("bt.reasoning.critique", 1024),
        ("bt.reasoning.revision", 1024),
        ("bt.metrics.token_breakdown", 768),
        ("bt.metrics.latency_breakdown", 768),
        ("bt.output.critic_feedback", 1536),
        ("bt.output.final_answer", 1536),
        ("bt.output.citations", 1024),
        ("bt.output.followups", 1024),
    ]

    slot = 1
    while _attrs_size_bytes(attrs) < target_span_bytes:
        for key, chars in key_specs:
            if _attrs_size_bytes(attrs) >= target_span_bytes:
                break
            numbered_key = f"{key}.{slot:03d}"
            attrs[numbered_key] = _seed_segment(seed, step=step, slot=slot, target_chars=chars)
            slot += 1
    return attrs


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
        return {"trace_id": "", "synthetic_spans": 0, "payload_bytes": 0, "emitted_at_unix_ms": 0}

    profile = _normalize_payload_profile(payload_profile)
    profile_cfg = _payload_profile_config[profile]
    target_payload_bytes = _resolve_target_bytes(profile, inject_large_attributes)
    span_target_bytes = _resolve_fragment_bytes()
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
    run_tag = str((metadata or {}).get("trace_run_tag") or os.environ.get("TRACE_RUN_TAG", "adhoc"))
    min_spans = int(profile_cfg["extra_spans"])
    synthetic_spans = max(min_spans, (target_payload_bytes + span_target_bytes - 1) // span_target_bytes)
    root_payload_preview = payload_seed[:4096]

    root_attributes: dict[str, Any] = {
        "span.kind": "agent_root",
        "trace.payload_profile": profile,
        "trace.inject_large_attributes": inject_large_attributes,
        "trace.target_payload_bytes": target_payload_bytes,
        "trace.span_target_bytes": span_target_bytes,
        "trace.pause_ms": pause_ms,
        "trace.pause_every_spans": pause_every,
        "agent.query": query[:4096],
        "agent.final_output": final_output[:4096],
        "agent.tool_call_count": len(tool_names),
        "agent.payload_preview": root_payload_preview,
        "stress_run_tag": run_tag,
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
            stage_target_bytes = span_target_bytes if remaining_bytes <= 0 else min(
                span_target_bytes, remaining_bytes
            )
            turn_attrs = _base_turn_attributes(
                step=idx,
                synthetic_spans=synthetic_spans,
                query=query,
                final_output=final_output,
                tool_names=tool_names,
                seed=payload_seed,
                run_tag=run_tag,
            )
            turn_attrs = _expanded_turn_attributes(
                step=idx,
                target_span_bytes=stage_target_bytes,
                seed=payload_seed,
                base=turn_attrs,
            )
            emitted_payload_bytes += _attrs_size_bytes(turn_attrs)
            with _tracer.start_as_current_span(
                f"synthetic_step_{idx:03d}",
                attributes=turn_attrs,
            ):
                pass

    return {
        "trace_id": trace_id,
        "synthetic_spans": synthetic_spans,
        "payload_target_bytes": target_payload_bytes,
        "payload_bytes": emitted_payload_bytes,
        "emitted_at_unix_ms": int(time.time() * 1000),
        "trace_run_tag": run_tag,
    }


def get_export_stats() -> dict[str, float | int]:
    with _stats_lock:
        return dict(_export_stats)


def force_flush() -> None:
    if _provider is not None:
        _provider.force_flush()
