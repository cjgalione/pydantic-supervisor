"""Tests for model provider normalization helpers."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from pydantic_ai import Agent

from src.modeling import (
    DEFAULT_BRAINTRUST_GATEWAY_BASE_URL,
    DEFAULT_GOOGLE_MODEL,
    GOOGLE_PROVIDER_PREFIX,
    resolve_model_name,
)


class ResolveModelNameTests(unittest.TestCase):
    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_default_model_uses_google_provider(self) -> None:
        resolved = resolve_model_name(None)
        self.assertEqual(resolved, f"{GOOGLE_PROVIDER_PREFIX}:{DEFAULT_GOOGLE_MODEL}")

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_gemini_model_gets_google_prefix(self) -> None:
        self.assertEqual(
            resolve_model_name("gemini-2.0-flash-lite"),
            "google-gla:gemini-2.0-flash-lite",
        )

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_gpt_model_gets_openai_prefix(self) -> None:
        self.assertEqual(
            resolve_model_name("gpt-4.1-mini"),
            "openai:gpt-4.1-mini",
        )

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_gateway_vendor_model_gets_openai_chat_prefix_without_gateway_key(self) -> None:
        self.assertEqual(
            resolve_model_name("moonshotai/Kimi-K2.5"),
            "openai:moonshotai/Kimi-K2.5",
        )

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_openai_responses_vendor_model_downgrades_to_openai_chat_without_gateway_key(self) -> None:
        self.assertEqual(
            resolve_model_name("openai-responses:moonshotai/Kimi-K2.5"),
            "openai:moonshotai/Kimi-K2.5",
        )

    @patch.dict(os.environ, {"BRAINTRUST_API_KEY": ""}, clear=False)
    def test_test_model_resolution(self) -> None:
        resolved = resolve_model_name("test")
        self.assertTrue(resolved == "test" or type(resolved).__name__ == "TestModel")


class AgentInstantiationTests(unittest.TestCase):
    def test_gateway_vendor_model_instantiates_openai_responses_model(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "unit-test-key",
                "GOOGLE_API_KEY": "",
                "GEMINI_API_KEY": "",
                "BRAINTRUST_API_KEY": "",
            },
            clear=False,
        ):
            agent = Agent(model=resolve_model_name("moonshotai/Kimi-K2.5"))
            self.assertEqual(type(agent.model).__name__, "OpenAIChatModel")

    def test_gateway_vendor_model_uses_braintrust_gateway_when_key_present(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BRAINTRUST_API_KEY": "bt-test-key",
                "BRAINTRUST_PROJECT_ID": "proj-test-id",
                "BRAINTRUST_ORG_NAME": "Test Org",
                "OPENAI_API_KEY": "",
            },
            clear=False,
        ):
            resolved = resolve_model_name("moonshotai/Kimi-K2.5")
            self.assertEqual(type(resolved).__name__, "OpenAIChatModel")
            self.assertEqual(str(resolved.client.base_url), DEFAULT_BRAINTRUST_GATEWAY_BASE_URL)
            self.assertEqual(resolved.client.api_key, "bt-test-key")
            self.assertEqual(resolved.client.default_headers.get("x-bt-project-id"), "proj-test-id")
            self.assertEqual(resolved.client.default_headers.get("x-bt-org-name"), "Test Org")


if __name__ == "__main__":
    unittest.main()
