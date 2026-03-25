"""Shared helpers for eval modules.

Centralises functions that were previously duplicated across
eval_golden.py and eval_supervisor.py.
"""

from __future__ import annotations

import inspect
import os
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from evals.parameters import extract_prompt_and_model
from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic
from src.config import (
    AgentConfig,
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)
from src.helpers import extract_query_from_input


# ---------------------------------------------------------------------------
# Judge client (shared across eval modules)
# ---------------------------------------------------------------------------

judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={"x-bt-use-cache": "always"},
)


# ---------------------------------------------------------------------------
# Gateway header helpers
# ---------------------------------------------------------------------------

def extract_gateway_headers(headers: Any) -> dict[str, str]:
    """Extract gateway-specific response headers from raw OpenAI responses."""
    gateway_metadata: dict[str, str] = {}
    if headers is None:
        return gateway_metadata
    used_endpoint = headers.get("x-bt-used-endpoint")
    cache_status = headers.get("x-bt-cached") or headers.get("x-cached")
    if used_endpoint:
        gateway_metadata["gateway_used_endpoint"] = str(used_endpoint)
    if cache_status:
        gateway_metadata["gateway_cache_status"] = str(cache_status)
    return gateway_metadata


def parse_with_gateway_metadata(
    *, model: str, input_data: list[dict[str, Any]], text_format: Any
) -> tuple[Any, dict[str, str]]:
    """Parse a Responses API call while preserving gateway response headers."""
    raw_response = judge_client.responses.with_raw_response.parse(
        model=model,
        input=input_data,
        text_format=text_format,
    )
    gateway_metadata = extract_gateway_headers(getattr(raw_response, "headers", None))
    return raw_response.parse(), gateway_metadata


# ---------------------------------------------------------------------------
# Output inspection helpers
# ---------------------------------------------------------------------------

def is_error_output(output: Any) -> bool:
    """Return True when the task produced no usable output."""
    if output is None:
        return True
    if isinstance(output, dict):
        if output.get("error"):
            return True
        final = output.get("final_output")
        if final is None or (isinstance(final, str) and not final.strip()):
            messages = output.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return True
    return False


def latest_assistant_text(output: Any) -> str:
    """Extract the latest assistant response text from the output payload."""
    if not isinstance(output, dict):
        return str(output) if output else ""
    final = output.get("final_output", "")
    if isinstance(final, str) and final.strip():
        return final.strip()
    messages = output.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return str(msg["content"])
    return ""


def output_messages(output: Any) -> list[dict[str, Any]]:
    """Extract message list from output payload."""
    if not isinstance(output, dict):
        return []
    messages = output.get("messages", [])
    if not isinstance(messages, list):
        return []
    return [m for m in messages if isinstance(m, dict)]


# ---------------------------------------------------------------------------
# Number and agent extraction
# ---------------------------------------------------------------------------

def extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from a text string."""
    sci_caret = re.findall(r"(-?\d+(?:\.\d+)?)\s*[x×]\s*10\^(-?\d+)", text, flags=re.IGNORECASE)
    numbers: list[float] = []
    for base_s, exp_s in sci_caret:
        try:
            numbers.append(float(base_s) * (10 ** int(exp_s)))
        except ValueError:
            pass

    raw_numbers = re.findall(r"-?\d+(?:[,_]\d+)*(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    for n in raw_numbers:
        try:
            numbers.append(float(n.replace(",", "").replace("_", "")))
        except ValueError:
            pass
    return numbers


def infer_agents_from_messages(output: Any) -> set[str]:
    """Infer which agents were called from the serialised message trace."""
    found: set[str] = set()
    if not isinstance(output, dict):
        return found
    messages = output.get("messages", [])
    if not isinstance(messages, list):
        return found
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = str(msg.get("content", "") or "").lower()
        if "handoff [researchagent]" in content or "request_research_subtask" in content:
            found.add("ResearchAgent")
        if "handoff [mathagent]" in content or "request_math_subtask" in content:
            found.add("MathAgent")
        tool_calls = msg.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = str(tc.get("name", "") or "").lower()
            if any(k in name for k in ("research", "tavily", "delegate_to_research")):
                found.add("ResearchAgent")
            if any(k in name for k in ("math", "add", "subtract", "multiply", "divide", "delegate_to_math")):
                found.add("MathAgent")
    return found


# ---------------------------------------------------------------------------
# Parameter unwrapping
# ---------------------------------------------------------------------------

def unwrap_parameters(params: dict) -> dict:
    """Extract raw parameter values from Braintrust parameter objects."""
    system_prompt, supervisor_model = extract_prompt_and_model(
        params.get("system_prompt"),
        default_prompt=DEFAULT_SYSTEM_PROMPT,
        default_model=DEFAULT_SUPERVISOR_MODEL,
    )
    research_agent_prompt, research_model = extract_prompt_and_model(
        params.get("research_agent_prompt"),
        default_prompt=DEFAULT_RESEARCH_AGENT_PROMPT,
        default_model=DEFAULT_RESEARCH_MODEL,
    )
    math_agent_prompt, math_model = extract_prompt_and_model(
        params.get("math_agent_prompt"),
        default_prompt=DEFAULT_MATH_AGENT_PROMPT,
        default_model=DEFAULT_MATH_MODEL,
    )

    prompt_modification = params.get("prompt_modification")
    if inspect.isclass(prompt_modification) and issubclass(prompt_modification, BaseModel):
        prompt_modification = getattr(prompt_modification(), "value", "")
    elif isinstance(prompt_modification, BaseModel):
        prompt_modification = getattr(prompt_modification, "value", "")
    elif prompt_modification is None:
        prompt_modification = ""

    return {
        "system_prompt": system_prompt,
        "prompt_modification": prompt_modification,
        "research_agent_prompt": research_agent_prompt,
        "math_agent_prompt": math_agent_prompt,
        "supervisor_model": supervisor_model,
        "research_model": research_model,
        "math_model": math_model,
    }


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

async def run_supervisor_task(
    input: dict,
    hooks: Any = None,
    *,
    app_name: str = "pydantic-supervisor-eval",
    extra_metadata_keys: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run a single task through the supervisor + critic pipeline.

    Args:
        input: The eval input payload.
        hooks: Braintrust hooks object (provides parameters and metadata).
        app_name: Tracing app name for this eval run.
        extra_metadata_keys: Additional keys from run_result to copy into hooks.metadata.
    """
    try:
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}
        config_params = unwrap_parameters(params)
        config = AgentConfig(**config_params) if config_params else None

        supervisor = get_supervisor(config=config, force_rebuild=True)
        query = extract_query_from_input(input)

        run_result = await run_supervisor_with_critic(
            supervisor=supervisor,
            query=query,
            app_name=app_name,
        )
        serialized_messages = run_result["messages"]

        if hooks and hasattr(hooks, "metadata"):
            meta = {
                "final_output": run_result.get("final_output", ""),
                "num_messages": len(serialized_messages),
            }
            for key in extra_metadata_keys:
                if key in run_result:
                    meta[key] = run_result[key]
            hooks.metadata.update(meta)

        return {
            "final_output": run_result.get("final_output", ""),
            "messages": serialized_messages,
        }
    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"final_output": "", "messages": [{"error": str(e)}]}
