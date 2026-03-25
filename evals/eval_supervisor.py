"""Supervisor evaluation for the PydanticAI implementation."""

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

# Ensure project root is on sys.path so `src` package can be imported
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, init_dataset, init_function  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.braintrust_parameter_patch import apply_parameter_patch  # noqa: E402
from evals.braintrust_gateway_header_patch import apply_gateway_header_patch  # noqa: E402
from evals.parameters import (  # noqa: E402
    MathAgentPromptParam,
    PromptModificationParam,
    ResearchAgentPromptParam,
    SystemPromptParam,
)
from evals.shared import (  # noqa: E402
    is_error_output,
    latest_assistant_text,
    output_messages,
    parse_with_gateway_metadata,
    run_supervisor_task as _run_supervisor_task,
)
from src.helpers import extract_query_from_input  # noqa: E402
from src.tracing import configure_adk_tracing  # noqa: E402

load_dotenv()
apply_parameter_patch()
apply_gateway_header_patch()

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"
DEFAULT_BRAINTRUST_DATASET = "Pydantic Supervisor Dataset"

configure_adk_tracing(
    api_key=os.environ.get("BRAINTRUST_API_KEY"),
    project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
    project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
)


# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, Any]:
    return await _run_supervisor_task(
        input,
        hooks,
        app_name="pydantic-supervisor-eval-supervisor",
    )


# ---------------------------------------------------------------------------
# Supervisor-specific query heuristics
# ---------------------------------------------------------------------------

def _query_requires_math_handoff(query: str) -> bool:
    q = query.lower()
    if re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\b", q):
        return True
    if re.search(r"\d", q) and any(
        token in q
        for token in (
            "calculate",
            "add",
            "subtract",
            "multiply",
            "divide",
            "sum",
            "difference",
            "product",
            "square root",
            "percent",
            "minus",
            "plus",
            "area",
        )
    ):
        return True
    return False


def _query_requires_research_handoff(query: str) -> bool:
    q = query.lower()
    if any(
        token in q
        for token in (
            "latest",
            "current",
            "who is",
            "what is the capital",
            "population",
            "president",
            "ceo",
            "mayor",
            "won",
            "source",
            "sources",
        )
    ):
        return True
    return bool(re.search(r"\b(who|what|when|where)\b", q)) and (not _query_requires_math_handoff(query))


def _has_message_marker(messages: list[dict[str, Any]], markers: tuple[str, ...]) -> bool:
    lowered = tuple(m.lower() for m in markers)
    for message in messages:
        content = str(message.get("content", "") or "").lower()
        if any(marker in content for marker in lowered):
            return True
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_name = str(tc.get("name", "") or "").lower()
            if any(marker in tool_name for marker in lowered):
                return True
    return False


# ---------------------------------------------------------------------------
# Scorers
# ---------------------------------------------------------------------------

async def delegation_compliance_scorer(input, output, expected, metadata, trace):
    """Deterministic policy compliance check for required delegation markers."""
    del expected, metadata, trace

    if is_error_output(output):
        return {"name": "Delegation Compliance", "score": None}

    if isinstance(input, dict):
        try:
            query = extract_query_from_input(input)
        except Exception:
            query = str(input)
    else:
        query = str(input)

    messages = output_messages(output)

    requires_math = _query_requires_math_handoff(query)
    requires_research = _query_requires_research_handoff(query)

    has_math_handoff = _has_message_marker(
        messages,
        ("delegate_to_math_agent", "request_math_subtask", "handoff [mathagent]"),
    )
    has_research_handoff = _has_message_marker(
        messages,
        ("delegate_to_research_agent", "request_research_subtask", "handoff [researchagent]"),
    )
    has_web_marker = _has_message_marker(
        messages,
        ("tavily_search", "http://", "https://", "url:"),
    )

    math_ok = (not requires_math) or has_math_handoff
    research_ok = (not requires_research) or (has_research_handoff and has_web_marker)
    compliant = math_ok and research_ok

    return {
        "name": "Delegation Compliance",
        "score": 1.0 if compliant else 0.0,
        "metadata": {
            "query": query,
            "requires_math_handoff": requires_math,
            "requires_research_handoff": requires_research,
            "has_math_handoff": has_math_handoff,
            "has_research_handoff": has_research_handoff,
            "has_web_marker": has_web_marker,
        },
    }


class RoutingAccuracyOutput(BaseModel):
    """Structured output for routing accuracy evaluation."""

    choice: Literal["A", "B", "C", "D"]
    reasoning: str


ROUTING_ACCURACY_PROMPT = """
You are an expert evaluator of AI agent routing systems. Your task is to determine whether a user question was correctly routed to the appropriate agents.

The system has the following specialized agents:
1. **MathAgent**: Should handle mathematical calculations, arithmetic, equations, numerical problems, and any query requiring computation with specific numbers.
2. **ResearchAgent**: Should handle factual questions, information lookup, current events, geography, history, statistics, and any query requiring external knowledge or web search.

The supervisor can:
- Route to a single agent
- Route to multiple agents (if the query requires both research and math)
- Answer directly without routing (for simple greetings, conversational queries, or ambiguous questions)

**User Question**: {input}

**Agents Called**: {agents_called}

**Task**: Evaluate the routing decision and respond with your reasoning, then select ONE of these options:

(A) CORRECT
(B) MOSTLY_CORRECT
(C) PARTIALLY_WRONG
(D) INCORRECT
"""


def _infer_agents_from_tool_name(tool_name: str) -> set[str]:
    lowered = tool_name.lower()
    agents: set[str] = set()

    if any(
        key in lowered
        for key in (
            "research",
            "tavily",
            "delegate_to_research_agent",
            "request_research_subtask",
        )
    ):
        agents.add("ResearchAgent")

    if any(
        key in lowered
        for key in (
            "math",
            "delegate_to_math_agent",
            "request_math_subtask",
            "add",
            "subtract",
            "multiply",
            "divide",
        )
    ):
        agents.add("MathAgent")

    return agents


def _infer_agents_from_span_name(span_name: str) -> set[str]:
    lowered = span_name.lower()
    agents: set[str] = set()
    if "researchagent" in lowered:
        agents.add("ResearchAgent")
    if "mathagent" in lowered:
        agents.add("MathAgent")
    return agents


async def _collect_agents_called(trace: Any, output: Any) -> list[str]:
    """Infer called agents from trace spans and serialized tool call messages."""
    found: set[str] = set()

    spans: list[Any] = []
    try:
        spans = await trace.get_spans(span_type=["task", "llm"])
    except Exception:
        spans = []

    for span in spans:
        span_name = str(getattr(span, "span_attributes", {}).get("name", "") or "")
        lowered = span_name.lower()
        if span_name in {"MathAgent", "ResearchAgent"}:
            found.add(span_name)
        else:
            found.update(_infer_agents_from_span_name(span_name))
            found.update(_infer_agents_from_tool_name(lowered))

    if isinstance(output, dict):
        messages = output.get("messages", [])
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    continue
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tool_name = str(tc.get("name", "") or "")
                    found.update(_infer_agents_from_tool_name(tool_name))

    ordered = [name for name in ["ResearchAgent", "MathAgent"] if name in found]
    return ordered


async def routing_accuracy_scorer(input, output, expected, metadata, trace):
    if is_error_output(output):
        return {"name": "Routing Accuracy", "score": None}

    choice_map = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.0}
    agents_called = await _collect_agents_called(trace, output)

    agents_called_str = (
        ", ".join(agents_called)
        if agents_called
        else "None (supervisor answered directly)"
    )

    prompt = ROUTING_ACCURACY_PROMPT.format(
        input=input,
        agents_called=agents_called_str,
    )
    response, gateway_metadata = parse_with_gateway_metadata(
        model="gpt-4o-mini",
        input_data=[{"role": "user", "content": prompt}],
        text_format=RoutingAccuracyOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        metadata = {
            "agents_called": agents_called_str,
            "reasoning": "No output",
            "choice": "D",
        }
        metadata.update(gateway_metadata)
        return {
            "name": "Routing Accuracy",
            "score": 0.0,
            "metadata": metadata,
        }

    metadata = {
        "agents_called": agents_called_str,
        "reasoning": parsed.reasoning,
        "choice": parsed.choice,
    }
    metadata.update(gateway_metadata)
    return {
        "name": "Routing Accuracy",
        "score": choice_map.get(parsed.choice, 0.0),
        "metadata": metadata,
    }


response_quality_prompt = """
You are an expert evaluator of AI assistant responses.

User Question: {{input}}
AI Response: {{output}}

Evaluate the response based on:
1. ACCURACY
2. COMPLETENESS
3. CLARITY
4. RELEVANCE

Scoring guidance:
- For pure arithmetic questions, a concise correct numeric answer is acceptable.
- For compound questions that ask for both a factual lookup and a calculation,
  the response must include both the factual answer and the computed result.
- Do not mark a response incorrect merely for brevity if it fully answers the question.

Respond with:
EXCELLENT
GOOD
FAIR
POOR
"""


def _is_self_contained_math_query(query: str) -> bool:
    q = query.lower()

    # Heuristic: these query forms usually contain all required operands/expressions.
    if "derivative" in q and re.search(r"derivative\s+of\s+.+", q):
        return True
    if "integral" in q and re.search(r"integral\s+of\s+.+", q):
        return True
    if "limit" in q and re.search(r"limit\s+of\s+.+", q):
        return True
    if "solve for" in q and "=" in q:
        return True
    if "quadratic equation" in q and re.search(r"[a-z0-9\)\]]\s*=\s*[a-z0-9\(\[]", q):
        return True
    if re.search(r"\b(x\^2|x²)\b", q) and "=" in q:
        return True
    return False


def _looks_like_clarification_request(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"\b(i need|need more information|could you provide|please provide)\b",
        r"\bwhat (is|are) .*(looking for|value|values)\b",
        r"\bi need the value of\b",
        r"\bcould you clarify\b",
    ]
    return any(re.search(p, lowered) for p in patterns)


async def no_unnecessary_clarification_scorer(input, output, expected, metadata, trace):
    """Penalize asking for clarification when the math prompt is already self-contained."""
    del expected, metadata, trace

    if is_error_output(output):
        return {"name": "No Unnecessary Clarification", "score": None}

    if isinstance(input, dict):
        try:
            query = extract_query_from_input(input)
        except Exception:
            query = str(input)
    else:
        query = str(input)

    assistant_text = latest_assistant_text(output)
    self_contained = _is_self_contained_math_query(query)
    asked_for_clarification = _looks_like_clarification_request(assistant_text)

    bad_case = self_contained and asked_for_clarification
    return {
        "name": "No Unnecessary Clarification",
        "score": 0.0 if bad_case else 1.0,
        "metadata": {
            "self_contained_math_query": self_contained,
            "asked_for_clarification": asked_for_clarification,
            "query": query,
            "assistant_response": assistant_text,
        },
    }


class ResponseQualityOutput(BaseModel):
    """Structured output for response quality scoring."""

    choice: Literal["EXCELLENT", "GOOD", "FAIR", "POOR"]
    reasoning: str


async def response_quality_scorer(input, output, expected, metadata, trace):
    """Score response quality with structured output parsing."""
    del expected, metadata, trace

    if is_error_output(output):
        return {"name": "Response Quality", "score": None}

    assistant_response = latest_assistant_text(output)

    if isinstance(input, dict):
        try:
            normalized_input = extract_query_from_input(input)
        except Exception:
            normalized_input = str(input)
    else:
        normalized_input = str(input)

    prompt = response_quality_prompt.replace("{{input}}", normalized_input).replace(
        "{{output}}",
        assistant_response or str(output),
    )

    response, gateway_metadata = parse_with_gateway_metadata(
        model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
        input_data=[{"role": "user", "content": prompt}],
        text_format=ResponseQualityOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        metadata = {"choice": "POOR", "reasoning": "No parsed output"}
        metadata.update(gateway_metadata)
        return {
            "name": "Response Quality",
            "score": 0.0,
            "metadata": metadata,
        }

    score_map = {"EXCELLENT": 1.0, "GOOD": 0.75, "FAIR": 0.5, "POOR": 0.0}
    metadata = {"choice": parsed.choice, "reasoning": parsed.reasoning}
    metadata.update(gateway_metadata)
    return {
        "name": "Response Quality",
        "score": score_map.get(parsed.choice, 0.0),
        "metadata": metadata,
    }


async def step_efficiency_scorer(output):
    """Score based on number of serialized messages."""
    max_steps = 8
    if isinstance(output, dict):
        num_steps = len(output.get("messages", []))
    elif isinstance(output, str):
        num_steps = 1 if output.strip() else 0
    else:
        num_steps = 0

    if num_steps <= max_steps:
        return 1.0
    return max(0.0, 1.0 - (num_steps - max_steps) / max_steps)


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_local_dataset() -> list[dict[str, Any]]:
    """Load eval data from local dataset.jsonl for deterministic local execution."""
    dataset_path = project_root / "dataset.jsonl"
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    # Keep at least one explicit case that should never be answered directly.
    rows.append(
        {
            "input": {
                "messages": [
                    {
                        "content": "Calculate 341 * 29. Do not answer directly without delegating to MathAgent.",
                        "type": "human",
                        "additional_kwargs": {},
                        "example": False,
                        "id": None,
                        "name": None,
                        "response_metadata": {},
                    }
                ]
            }
        }
    )
    return rows


def get_eval_data(project_name: str):
    """Choose local dataset by default, with optional remote dataset override."""
    use_remote = os.environ.get("BRAINTRUST_USE_REMOTE_DATASET", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    if use_remote:
        return init_dataset(
            project=project_name,
            name=os.environ.get("BRAINTRUST_DATASET", DEFAULT_BRAINTRUST_DATASET),
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            org_name=os.environ.get("BRAINTRUST_ORG_NAME", "Braintrust Demos"),
        )
    return load_local_dataset()


# ---------------------------------------------------------------------------
# Eval registration
# ---------------------------------------------------------------------------

def _register_eval():
    project_name = os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT)

    use_published_step_scorer = (
        os.environ.get("USE_PUBLISHED_STEP_SCORER", "1").lower() in {"1", "true", "yes"}
    )
    published_step_efficiency_score = (
        init_function(project_name=project_name, slug="step-efficiency")
        if use_published_step_scorer
        else None
    )

    async def step_efficiency_score(output: Any):
        """Use the published scorer when available, with local fallback on invoke failures."""
        if published_step_efficiency_score is None:
            return await step_efficiency_scorer(output)

        try:
            return published_step_efficiency_score(output=output)
        except Exception as exc:
            fallback_score = await step_efficiency_scorer(output)
            return {
                "name": "Step Efficiency",
                "score": fallback_score,
                "metadata": {
                    "fallback": "local_step_efficiency_scorer",
                    "published_invoke_error": f"{type(exc).__name__}: {exc}",
                    "project_name": project_name,
                    "scorer_slug": "step-efficiency",
                },
            }

    Eval(
        project_name,
        data=get_eval_data(project_name),
        task=run_supervisor_task,
        scores=[
            response_quality_scorer,
            no_unnecessary_clarification_scorer,
            routing_accuracy_scorer,
            delegation_compliance_scorer,
            step_efficiency_score,
        ],  # type: ignore
        parameters={
            "system_prompt": SystemPromptParam,
            "prompt_modification": PromptModificationParam,
            "research_agent_prompt": ResearchAgentPromptParam,
            "math_agent_prompt": MathAgentPromptParam,
        },
    )


# Register the eval at import time (required by braintrust eval's lazy-load discovery).
# Set SKIP_EVAL_REGISTRATION=1 to import scorers without triggering the eval.
if not os.environ.get("SKIP_EVAL_REGISTRATION"):
    _register_eval()
