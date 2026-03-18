"""Patch Braintrust OpenAI tracing to include gateway routing headers in span metadata."""

from __future__ import annotations

from typing import Any


def apply_gateway_header_patch() -> bool:
    """Augment braintrust.oai.log_headers to record gateway endpoint/cache headers.

    Returns True when the patch is successfully applied, otherwise False.
    """
    try:
        import braintrust.oai as oai_module  # pyright: ignore[reportMissingImports]
    except Exception as exc:
        print(f"Warning: could not import braintrust.oai for gateway header patch: {exc}")
        return False

    if getattr(oai_module, "_gateway_header_patch_applied", False):
        return True

    original_log_headers = oai_module.log_headers

    def patched_log_headers(response: Any, span: Any) -> None:
        # Preserve existing behavior (e.g., x-bt-cached metric logging).
        original_log_headers(response, span)

        headers = getattr(response, "headers", None)
        if headers is None:
            return

        gateway_used_endpoint = headers.get("x-bt-used-endpoint")
        gateway_cache_status = headers.get("x-bt-cached") or headers.get("x-cached")

        metadata: dict[str, str] = {}
        if gateway_used_endpoint:
            metadata["gateway_used_endpoint"] = str(gateway_used_endpoint)
        if gateway_cache_status:
            metadata["gateway_cache_status"] = str(gateway_cache_status)

        if metadata:
            span.log(metadata=metadata)

    oai_module.log_headers = patched_log_headers
    oai_module._gateway_header_patch_applied = True
    print("Applied Braintrust gateway header patch")
    return True
