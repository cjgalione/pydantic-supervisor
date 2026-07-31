from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from src.agents import deep_agent


class CapturedSpan:
    def __init__(self) -> None:
        self.logged_outputs: list[dict[str, Any]] = []

    def log(self, **kwargs: Any) -> None:
        self.logged_outputs.append(kwargs)


@pytest.mark.asyncio
async def test_supervisor_root_trace_uses_chat_input_for_topics(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_starts: list[dict[str, Any]] = []
    captured_span = CapturedSpan()

    @contextmanager
    def fake_start_span(**kwargs: Any) -> Iterator[CapturedSpan]:
        captured_starts.append(kwargs)
        yield captured_span

    async def fake_run_pydantic_agent(**kwargs: Any) -> dict[str, Any]:
        return {
            "final_output": "The answer is 42.",
            "messages": [
                {"role": "user", "content": kwargs["query"]},
                {"role": "assistant", "content": "The answer is 42."},
            ],
        }

    monkeypatch.setattr(deep_agent, "start_span", fake_start_span)
    monkeypatch.setattr(deep_agent, "run_pydantic_agent", fake_run_pydantic_agent)

    result = await deep_agent.run_supervisor_with_critic(
        supervisor=object(),
        query="What is 6 * 7?",
        app_name="pydantic-supervisor-batch",
        trace_context={"github_run_id": "123", "question_number": 7},
    )

    assert result["final_output"] == "The answer is 42."
    assert captured_starts[0]["input"] == {
        "new_message": {"role": "user", "parts": [{"text": "What is 6 * 7?"}]},
        "trace_context": {"github_run_id": "123", "question_number": 7},
    }
    assert captured_starts[0]["metadata"] == {"app_name": "pydantic-supervisor-batch"}
    assert "app_name" not in captured_starts[0]["input"]
    assert captured_span.logged_outputs[0]["output"]["final_output"] == "The answer is 42."
