"""Tests for Modal eval server request helpers."""

from __future__ import annotations

from src.eval_server import _normalize_asgi_header_pairs, _normalize_asgi_headers


def test_normalize_asgi_headers_accepts_modal_bytearrays() -> None:
    headers = _normalize_asgi_headers(
        [
            (bytearray(b"Origin"), bytearray(b"https://braintrust.dev")),
            (bytearray(b"X-BT-Auth-Token"), bytearray(b"bt-test-token")),
        ]
    )

    assert headers[b"origin"] == b"https://braintrust.dev"
    assert headers[b"x-bt-auth-token"] == b"bt-test-token"


def test_normalize_asgi_header_pairs_returns_hashable_pairs() -> None:
    header_pairs = _normalize_asgi_header_pairs(
        [
            (bytearray(b"Origin"), bytearray(b"https://braintrust.dev")),
        ]
    )

    assert header_pairs == [(b"origin", b"https://braintrust.dev")]
    assert dict(header_pairs) == {b"origin": b"https://braintrust.dev"}
