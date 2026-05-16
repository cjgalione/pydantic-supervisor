"""Model and API-key compatibility helpers for PydanticAI providers."""

from __future__ import annotations

import os
from contextvars import ContextVar, Token
from typing import Any

DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
OPENAI_PROVIDER_PREFIX = "openai"
OPENAI_RESPONSES_PROVIDER_PREFIX = "openai-responses"
DEFAULT_BRAINTRUST_GATEWAY_BASE_URL = "https://gateway.braintrust.dev"

_request_braintrust_api_key: ContextVar[str | None] = ContextVar(
    "request_braintrust_api_key",
    default=None,
)
_request_braintrust_project_id: ContextVar[str | None] = ContextVar(
    "request_braintrust_project_id",
    default=None,
)
_request_braintrust_org_name: ContextVar[str | None] = ContextVar(
    "request_braintrust_org_name",
    default=None,
)


def set_braintrust_gateway_context(
    *,
    api_key: str | None = None,
    project_id: str | None = None,
    org_name: str | None = None,
) -> tuple[Token[str | None], Token[str | None], Token[str | None]]:
    """Attach Braintrust request auth to the current async context."""
    return (
        _request_braintrust_api_key.set(api_key),
        _request_braintrust_project_id.set(project_id),
        _request_braintrust_org_name.set(org_name),
    )


def reset_braintrust_gateway_context(
    tokens: tuple[Token[str | None], Token[str | None], Token[str | None]],
) -> None:
    """Reset request-scoped Braintrust auth context."""
    _request_braintrust_api_key.reset(tokens[0])
    _request_braintrust_project_id.reset(tokens[1])
    _request_braintrust_org_name.reset(tokens[2])


def get_openai_api_key() -> str | None:
    """Return the configured OpenAI-compatible key."""
    return os.environ.get("OPENAI_API_KEY")


def _request_or_env(name: str, request_value: str | None) -> str:
    return (os.environ.get(name) or request_value or "").strip()


def _get_braintrust_gateway_api_key() -> str:
    return _request_or_env("BRAINTRUST_API_KEY", _request_braintrust_api_key.get())


def _build_braintrust_gateway_model(model_name: str) -> Any | None:
    """Build a Braintrust gateway-backed OpenAI-compatible model."""
    gateway_api_key = _get_braintrust_gateway_api_key()
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
    project_id = _request_or_env(
        "BRAINTRUST_PROJECT_ID",
        _request_braintrust_project_id.get(),
    )
    org_name = _request_or_env("BRAINTRUST_ORG_NAME", _request_braintrust_org_name.get())
    endpoint_name = (os.environ.get("BRAINTRUST_GATEWAY_ENDPOINT_NAME") or "").strip()
    if project_id:
        default_headers["x-bt-project-id"] = project_id
    if org_name:
        default_headers["x-bt-org-name"] = org_name
    if endpoint_name:
        default_headers["x-bt-endpoint-name"] = endpoint_name

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

    When a Braintrust API key is available, route provider/model selections
    through the Braintrust gateway so remote evals can use organization or
    project AI provider keys configured in Braintrust.
    """
    raw = (model_name or "").strip()
    if not raw:
        gateway_model = _build_braintrust_gateway_model(DEFAULT_OPENAI_MODEL)
        if gateway_model is not None:
            return gateway_model
        return f"{OPENAI_PROVIDER_PREFIX}:{DEFAULT_OPENAI_MODEL}"

    if raw.lower() == "test":
        try:
            from pydantic_ai.models.test import TestModel

            return TestModel(call_tools=[])
        except Exception:
            return "test"

    if ":" in raw:
        provider_name, explicit_model_name = raw.split(":", maxsplit=1)
        if provider_name in {OPENAI_PROVIDER_PREFIX, OPENAI_RESPONSES_PROVIDER_PREFIX}:
            gateway_model = _build_braintrust_gateway_model(explicit_model_name)
            if gateway_model is not None:
                return gateway_model
            if provider_name == OPENAI_RESPONSES_PROVIDER_PREFIX and "/" in explicit_model_name:
                return f"{OPENAI_PROVIDER_PREFIX}:{explicit_model_name}"
            return raw

        gateway_model = _build_braintrust_gateway_model(explicit_model_name)
        if gateway_model is not None:
            return gateway_model
        return raw

    gateway_model = _build_braintrust_gateway_model(raw)
    if gateway_model is not None:
        return gateway_model

    return f"{OPENAI_PROVIDER_PREFIX}:{raw}"
