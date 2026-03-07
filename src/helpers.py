"""Runtime helpers for PydanticAI runs and eval serialization."""

from __future__ import annotations

import json
import uuid
from typing import Any

from braintrust import SpanTypeAttribute, start_span

from src.modeling import ensure_google_api_keys
from src.tracing import get_trace_profile


def extract_query_from_input(input_payload: dict[str, Any]) -> str:
    """Extract a user query from eval input payloads."""
    if "query" in input_payload and input_payload["query"]:
        return str(input_payload["query"])

    new_message = input_payload.get("new_message")
    if isinstance(new_message, dict):
        parts = new_message.get("parts", [])
        if isinstance(parts, list):
            text_parts: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        text_parts.append(text.strip())
            if text_parts:
                return "\n".join(text_parts)

        content = new_message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    messages = input_payload.get("messages", [])
    if isinstance(messages, list) and messages:
        first_message = messages[0]
        if isinstance(first_message, dict):
            content = first_message.get("content")
            if isinstance(content, str):
                return content

    raise ValueError("Could not extract user query from input payload")


def _safe_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None), list, dict)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()  # type: ignore[attr-defined]
    try:
        return json.loads(json.dumps(value))
    except Exception:
        return str(value)


def _part_text(part: Any) -> str:
    text = getattr(part, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(part, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        text_parts = [str(item).strip() for item in content if isinstance(item, str) and str(item).strip()]
        if text_parts:
            return "\n".join(text_parts)

    return ""


def _part_tool_call(part: Any) -> tuple[str, Any] | None:
    type_name = type(part).__name__
    if "ToolCallPart" not in type_name and not hasattr(part, "args_as_dict"):
        return None

    tool_name = str(getattr(part, "tool_name", "") or "")
    if not tool_name:
        return None

    if hasattr(part, "args_as_dict"):
        try:
            args = part.args_as_dict()
        except Exception:
            args = getattr(part, "args", {})
    else:
        args = getattr(part, "args", {})

    return tool_name, args


def _part_tool_return(part: Any) -> tuple[str, Any] | None:
    type_name = type(part).__name__
    if "ToolReturnPart" not in type_name:
        return None

    tool_name = str(getattr(part, "tool_name", "") or "")
    content = getattr(part, "content", "")
    return tool_name, content


def _serialize_message(message: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    is_model_response = "ModelResponse" in type(message).__name__

    parts = getattr(message, "parts", None)
    if not isinstance(parts, list) or not parts:
        return out

    assistant_text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for part in parts:
        tool_call = _part_tool_call(part)
        if tool_call is not None:
            tool_name, args = tool_call
            tool_calls.append({"name": tool_name, "args": _safe_json(args)})
            continue

        tool_return = _part_tool_return(part)
        if tool_return is not None:
            _, response = tool_return
            out.append(
                {
                    "role": "tool",
                    "content": response if isinstance(response, str) else str(_safe_json(response)),
                }
            )
            continue

        text = _part_text(part)
        if text and is_model_response:
            assistant_text_parts.append(text)

    if tool_calls:
        out.insert(
            0,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            },
        )

    if assistant_text_parts:
        out.append(
            {
                "role": "assistant",
                "content": "\n".join(assistant_text_parts).strip(),
            }
        )

    return out


def _tool_calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_calls: list[dict[str, Any]] = []
    for message in messages:
        raw = message.get("tool_calls")
        if not isinstance(raw, list):
            continue
        for tc in raw:
            if isinstance(tc, dict):
                tool_calls.append(tc)
    return tool_calls


async def run_pydantic_agent(
    *,
    agent: Any,
    query: str,
    app_name: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Run a PydanticAI agent and return final text plus serialized messages."""
    ensure_google_api_keys()

    uid = user_id or "eval-user"
    sid = session_id or f"session-{uuid.uuid4().hex}"

    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    final_output = ""

    trace_profile = get_trace_profile()

    async def _run_once() -> Any:
        result = await agent.run(query)
        new_messages = result.new_messages()
        for message in new_messages:
            serialized = _serialize_message(message)
            messages.extend(serialized)

            if trace_profile == "lean":
                for tc in _tool_calls_from_messages(serialized):
                    tool_name = str(tc.get("name", "") or "")
                    if not tool_name:
                        continue
                    with start_span(
                        name=f"tool_routing_decision [{tool_name}]",
                        type=SpanTypeAttribute.TOOL,
                        input=_safe_json(tc.get("args", {})),
                        metadata={
                            "tool_name": tool_name,
                            "source": "llm_tool_selection",
                        },
                    ):
                        pass
        return result

    if trace_profile == "lean":
        with start_span(
            name=f"invocation [{app_name}]",
            type=SpanTypeAttribute.TASK,
            input={"new_message": {"role": "user", "parts": [{"text": query}]}},
            metadata={"user_id": uid, "session_id": sid, "app_name": app_name},
        ) as invocation_span:
            with start_span(
                name="llm_response_generation",
                type=SpanTypeAttribute.LLM,
                input={"query": query, "agent_name": str(getattr(agent, "name", "") or "")},
            ) as llm_span:
                result = await _run_once()
                final_output = str(getattr(result, "output", "") or "").strip()
                llm_span.log(output={"final_output": final_output})
            invocation_span.log(output={"final_output": final_output})
    else:
        result = await _run_once()
        final_output = str(getattr(result, "output", "") or "").strip()

    if final_output and not any(
        m.get("role") == "assistant" and m.get("content") for m in messages
    ):
        messages.append({"role": "assistant", "content": final_output})

    return {"final_output": final_output, "messages": messages}


# Backward-compatible alias for unchanged call sites.
async def run_adk_agent(
    *,
    agent: Any,
    query: str,
    app_name: str,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    return await run_pydantic_agent(
        agent=agent,
        query=query,
        app_name=app_name,
        user_id=user_id,
        session_id=session_id,
    )
