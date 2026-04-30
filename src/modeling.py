"""Model and API-key compatibility helpers for PydanticAI providers."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_PROVIDER_PREFIX = "openai"
OPENAI_RESPONSES_PROVIDER_PREFIX = "openai-responses"
DEFAULT_BRAINTRUST_GATEWAY_BASE_URL = "https://gateway.braintrust.dev"

def get_openai_api_key() -> str | None:
    """Return the configured OpenAI-compatible key."""
    return os.environ.get("OPENAI_API_KEY")

def _build_braintrust_gateway_vendor_model(model_name: str) -> Any | None:
    """Build a gateway-backed OpenAI-compatible model for vendor/model ids."""
    gateway_api_key = (os.environ.get("BRAINTRUST_API_KEY") or "").strip()
    if not gateway_api_key:
        return None

    base_url = (
        os.environ.get("BRAINTRUST_GATEWAY_BASE_URL")
        or DEFAULT_BRAINTRUST_GATEWAY_BASE_URL
    ).strip()
    if not base_url:
        return None

    default_headers: dict[str, str] = {
        "x-bt-use-cache": "always",
    }
    project_id = (os.environ.get("BRAINTRUST_PROJECT_ID") or "").strip()
    org_name = (os.environ.get("BRAINTRUST_ORG_NAME") or "").strip()
    if project_id:
        default_headers["x-bt-project-id"] = project_id
    if org_name:
        default_headers["x-bt-org-name"] = org_name

    try:
        from openai import AsyncOpenAI
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        openai_client = AsyncOpenAI(
            api_key=gateway_api_key,
            base_url=base_url,
            default_headers=default_headers or None,
        )
        # Vendor/model IDs on the Braintrust gateway may not support the OpenAI
        # Responses API payload shape for tool declarations; use chat-completions
        # compatibility for these routed models.
        return OpenAIChatModel(
            model_name,
            provider=OpenAIProvider(openai_client=openai_client),
        )
    except Exception:
        return None


def resolve_model_name(model_name: str | None) -> Any:
    """Normalize model names for PydanticAI provider syntax.

    Supports gateway vendor/model IDs (for example `moonshotai/Kimi-K2.5`) by
    routing them to OpenAI-compatible chat models.
    """
    raw = (model_name or "").strip()
    if not raw:
        return f"{OPENAI_PROVIDER_PREFIX}:{DEFAULT_OPENAI_MODEL}"
    if raw.lower() == "test":
        try:
            from pydantic_ai.models.test import TestModel

            return TestModel(call_tools=[])
        except Exception:
            return "test"
    if ":" in raw:
        provider_name, explicit_model_name = raw.split(":", maxsplit=1)
        if provider_name == OPENAI_RESPONSES_PROVIDER_PREFIX and "/" in explicit_model_name:
            gateway_model = _build_braintrust_gateway_vendor_model(explicit_model_name)
            if gateway_model is not None:
                return gateway_model
            return f"{OPENAI_PROVIDER_PREFIX}:{explicit_model_name}"
        return raw

    # Gateway model IDs usually come in vendor/model format.
    if "/" in raw:
        gateway_model = _build_braintrust_gateway_vendor_model(raw)
        if gateway_model is not None:
            return gateway_model
        return f"{OPENAI_PROVIDER_PREFIX}:{raw}"

    return f"{OPENAI_PROVIDER_PREFIX}:{raw}"
