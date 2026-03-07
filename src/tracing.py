"""Tracing profile controls for PydanticAI + Braintrust instrumentation."""

from __future__ import annotations

import os

from src.modeling import ensure_google_api_keys

DEFAULT_TRACE_PROFILE = "full"
_last_setup_signature: tuple[str, str, str, str] | None = None


def get_trace_profile() -> str:
    profile = os.environ.get("TRACE_PROFILE", DEFAULT_TRACE_PROFILE).strip().lower()
    return profile if profile in {"full", "lean"} else DEFAULT_TRACE_PROFILE


def use_pydantic_auto_instrumentation() -> bool:
    return get_trace_profile() != "lean"


def configure_pydantic_tracing(
    *,
    api_key: str | None,
    project_id: str | None,
    project_name: str | None,
) -> None:
    """Configure tracing based on profile.

    - full: initialize logger and patch PydanticAI with Braintrust auto instrumentation
    - lean: initialize logger only and rely on explicit spans in app code
    """
    global _last_setup_signature

    if not api_key:
        return

    ensure_google_api_keys()

    signature = (
        get_trace_profile(),
        api_key,
        project_id or "",
        project_name or "",
    )
    if _last_setup_signature == signature:
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
