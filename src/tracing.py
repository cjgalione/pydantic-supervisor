"""Tracing profile controls for PydanticAI instrumentation backends."""

from __future__ import annotations

import os

from src.modeling import ensure_google_api_keys
from src.tempo_stress_tracing import configure_otel_exporter

DEFAULT_TRACE_PROFILE = "full"
DEFAULT_TRACE_BACKEND = "braintrust"
_last_setup_signature: tuple[str, str, str, str, str] | None = None


def get_trace_profile() -> str:
    profile = os.environ.get("TRACE_PROFILE", DEFAULT_TRACE_PROFILE).strip().lower()
    return profile if profile in {"full", "lean"} else DEFAULT_TRACE_PROFILE


def get_trace_backend() -> str:
    backend = os.environ.get("TRACE_BACKEND", DEFAULT_TRACE_BACKEND).strip().lower()
    return backend if backend in {"braintrust", "otlp"} else DEFAULT_TRACE_BACKEND


def use_pydantic_auto_instrumentation() -> bool:
    return get_trace_profile() != "lean"


def configure_pydantic_tracing(
    *,
    api_key: str | None,
    project_id: str | None,
    project_name: str | None,
) -> None:
    """Configure tracing based on selected backend/profile.

    Supported backends:
    - braintrust: initialize logger and optional setup_pydantic_ai patching
    - otlp: initialize OpenTelemetry exporter for local Tempo stress tests
    """

    global _last_setup_signature

    ensure_google_api_keys()

    backend = get_trace_backend()
    signature = (
        backend,
        get_trace_profile(),
        api_key or "",
        project_id or "",
        project_name or "",
    )
    if _last_setup_signature == signature:
        return

    if backend == "otlp":
        configure_otel_exporter(
            service_name=project_name or "pydantic-supervisor",
            run_tag=os.environ.get("TRACE_RUN_TAG"),
        )
        _last_setup_signature = signature
        return

    if not api_key:
        return

    from braintrust.logger import init_logger

    setup_pydantic_ai = None
    try:
        from braintrust import setup_pydantic_ai as _setup_pydantic_ai

        setup_pydantic_ai = _setup_pydantic_ai
    except ImportError:
        try:
            from braintrust.wrappers.pydantic_ai import setup_pydantic_ai as _setup_pydantic_ai

            setup_pydantic_ai = _setup_pydantic_ai
        except ImportError:
            setup_pydantic_ai = None

    init_logger(
        api_key=api_key,
        project=project_name,
        project_id=project_id,
    )

    if use_pydantic_auto_instrumentation() and setup_pydantic_ai is not None:
        setup_pydantic_ai(
            api_key=api_key,
            project_id=project_id,
            project_name=project_name,
        )

    _last_setup_signature = signature


# Backward-compatible alias to keep existing call sites stable.
def configure_adk_tracing(
    *,
    api_key: str | None,
    project_id: str | None,
    project_name: str | None,
) -> None:
    configure_pydantic_tracing(
        api_key=api_key,
        project_id=project_id,
        project_name=project_name,
    )
