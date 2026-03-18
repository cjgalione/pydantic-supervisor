"""Model and API-key compatibility helpers for PydanticAI providers."""

from __future__ import annotations

import os
import warnings
from typing import Any

DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash-lite"
GOOGLE_PROVIDER_PREFIX = "google-gla"
OPENAI_RESPONSES_PROVIDER_PREFIX = "openai-responses"


def ensure_google_api_keys() -> None:
    """Mirror Gemini/Google key env vars for backwards compatibility."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    google_key = os.environ.get("GOOGLE_API_KEY")

    if gemini_key and not google_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key
    elif google_key and not gemini_key:
        os.environ["GEMINI_API_KEY"] = google_key


def get_google_api_key() -> str | None:
    """Return whichever Google-compatible key is configured."""
    ensure_google_api_keys()
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _infer_provider_prefix(raw_model_name: str) -> str | None:
    """Infer a provider prefix for provider-less model names.

    Uses PydanticAI's built-in legacy model prefix parser when available.
    """
    try:
        from pydantic_ai.models import parse_model_id

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            provider_name, _ = parse_model_id(raw_model_name)
    except Exception:
        provider_name = None

    return provider_name


def resolve_model_name(model_name: str | None) -> Any:
    """Normalize model names for PydanticAI provider syntax.

    Accepts legacy names such as `gemini-2.0-flash-lite` and rewrites them to
    `google-gla:gemini-2.0-flash-lite`. Also supports gateway vendor/model IDs
    (for example `moonshotai/Kimi-K2.5`) by routing them to
    `openai-responses:<vendor/model>`.
    """
    raw = (model_name or "").strip()
    if not raw:
        return f"{GOOGLE_PROVIDER_PREFIX}:{DEFAULT_GOOGLE_MODEL}"
    if raw.lower() == "test":
        try:
            from pydantic_ai.models.test import TestModel

            return TestModel(call_tools=[])
        except Exception:
            return "test"
    if ":" in raw:
        return raw

    inferred_provider = _infer_provider_prefix(raw)
    if inferred_provider:
        return f"{inferred_provider}:{raw}"

    # Gateway model IDs usually come in vendor/model format and should not be
    # forced through the Google provider.
    if "/" in raw:
        return f"{OPENAI_RESPONSES_PROVIDER_PREFIX}:{raw}"

    return f"{GOOGLE_PROVIDER_PREFIX}:{raw}"
