"""Tests for Modal eval server request helpers."""

from __future__ import annotations

from src.eval_server import _normalize_asgi_headers


def test_normalize_asgi_headers_accepts_modal_bytearrays() -> None:
    headers = _normalize_asgi_headers(
        [
            (bytearray(b"Origin"), bytearray(b"https://braintrust.dev")),
            (bytearray(b"X-BT-Auth-Token"), bytearray(b"bt-test-token")),
        ]
    )

    assert headers[b"origin"] == b"https://braintrust.dev"
    assert headers[b"x-bt-auth-token"] == b"bt-test-token"
