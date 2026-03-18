"""Model and API-key compatibility helpers for PydanticAI providers."""

from __future__ import annotations

import os
import warnings
from typing import Any

DEFAULT_GOOGLE_MODEL = "gemini-2.0-flash-lite"
GOOGLE_PROVIDER_PREFIX = "google-gla"
OPENAI_RESPONSES_PROVIDER_PREFIX = "openai-responses"
DEFAULT_BRAINTRUST_GATEWAY_BASE_URL = "https://gateway.braintrust.dev"


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


def _build_braintrust_gateway_responses_model(model_name: str) -> Any | None:
    """Build an OpenAI Responses model configured to call Braintrust gateway."""
    gateway_api_key = (os.environ.get("BRAINTRUST_API_KEY") or "").strip()
    if not gateway_api_key:
        return None

    base_url = (
        os.environ.get("BRAINTRUST_GATEWAY_BASE_URL")
        or DEFAULT_BRAINTRUST_GATEWAY_BASE_URL
    ).strip()
    if not base_url:
        return None

    default_headers: dict[str, str] = {}
    project_id = (os.environ.get("BRAINTRUST_PROJECT_ID") or "").strip()
    org_name = (os.environ.get("BRAINTRUST_ORG_NAME") or "").strip()
    if project_id:
        default_headers["x-bt-project-id"] = project_id
    if org_name:
        default_headers["x-bt-org-name"] = org_name

    try:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIResponsesModel
        from pydantic_ai.providers.openai import OpenAIProvider

        openai_client = AsyncOpenAI(
            api_key=gateway_api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        return OpenAIResponsesModel(
            model_name,
            provider=OpenAIProvider(openai_client=openai_client),
        )
    except Exception:
        return None


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
        provider_name, explicit_model_name = raw.split(":", maxsplit=1)
        if provider_name == OPENAI_RESPONSES_PROVIDER_PREFIX and "/" in explicit_model_name:
            gateway_model = _build_braintrust_gateway_responses_model(explicit_model_name)
            if gateway_model is not None:
                return gateway_model
        return raw

    inferred_provider = _infer_provider_prefix(raw)
    if inferred_provider:
        return f"{inferred_provider}:{raw}"

    # Gateway model IDs usually come in vendor/model format and should not be
    # forced through the Google provider.
    if "/" in raw:
        gateway_model = _build_braintrust_gateway_responses_model(raw)
        if gateway_model is not None:
            return gateway_model
        return f"{OPENAI_RESPONSES_PROVIDER_PREFIX}:{raw}"

    return f"{GOOGLE_PROVIDER_PREFIX}:{raw}"
