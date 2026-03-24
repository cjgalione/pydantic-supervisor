"""Reusable Braintrust scorer definitions for `braintrust push`."""

from __future__ import annotations

import os
import re
from typing import Any

import braintrust
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

PROJECT_NAME = os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor")
JUDGE_MODEL = os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o-mini")


class StepEfficiencyParams(BaseModel):
    output: list[dict[str, Any]]
    max_steps: int = 8


class NoUnnecessaryClarificationParams(BaseModel):
    input: Any
    output: Any


async def step_efficiency_scorer(output):
    """Score based on total number of output messages."""
    max_steps = 8
    if isinstance(output, dict):
        num_steps = len(output.get("messages", []))
    elif isinstance(output, list):
        # Online scoring root spans often store output as a list of message-like items.
        num_steps = len(output)
    elif isinstance(output, str):
        num_steps = 1 if output.strip() else 0
    else:
        num_steps = 0

    if num_steps <= max_steps:
        return 1.0
    return max(0.0, 1.0 - (num_steps - max_steps) / max_steps)


def _extract_query_from_payload(input_payload: Any) -> str:
    if isinstance(input_payload, str):
        return input_payload

    if not isinstance(input_payload, dict):
        return str(input_payload)

    if "query" in input_payload and input_payload["query"]:
        return str(input_payload["query"])

    new_message = input_payload.get("new_message")
    if isinstance(new_message, dict):
        parts = new_message.get("parts", [])
        if isinstance(parts, list):
            texts: list[str] = []
            for part in parts:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
            if texts:
                return "\n".join(texts)
        content = new_message.get("content")
        if isinstance(content, str) and content.strip():
            return content

    messages = input_payload.get("messages", [])
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            role = str(message.get("role", "")).lower()
            if role == "user" and isinstance(content, str) and content.strip():
                return content

    return str(input_payload)


def _latest_assistant_text(output_payload: Any) -> str:
    if isinstance(output_payload, str):
        return output_payload
    if not isinstance(output_payload, dict):
        return str(output_payload)

    messages = output_payload.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return str(msg.get("content"))

    content = output_payload.get("content")
    if isinstance(content, str):
        return content
    return str(output_payload)


def _is_self_contained_math_query(query: str) -> bool:
    q = query.lower()
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


async def no_unnecessary_clarification_scorer(input: Any, output: Any):
    """Penalize asking for clarification when a math query is already self-contained."""
    query = _extract_query_from_payload(input)
    assistant_text = _latest_assistant_text(output)

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


RESPONSE_QUALITY_PROMPT = """
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
""".strip()


project = braintrust.projects.create(name=PROJECT_NAME)

project.scorers.create(
    name="Step Efficiency",
    slug="step-efficiency",
    description="Penalizes excessive step counts in the final message trace.",
    parameters=StepEfficiencyParams,
    handler=step_efficiency_scorer,
)

project.scorers.create(
    name="Response Quality (LLM Judge)",
    slug="response-quality-llm-judge",
    description=(
        "LLM-as-a-judge scorer for overall response quality, with guidance for "
        "concise math answers and compound research+math questions."
    ),
    # Use chat-style messages to avoid OpenAI errors about missing `messages`.
    messages=[
        {
            "role": "user",
            "content": RESPONSE_QUALITY_PROMPT,
        }
    ],
    model=JUDGE_MODEL,
    use_cot=True,
    choice_scores={
        "EXCELLENT": 1.0,
        "GOOD": 0.75,
        "FAIR": 0.5,
        "POOR": 0.0,
    },
)

project.scorers.create(
    name="No Unnecessary Clarification",
    slug="no-unnecessary-clarification",
    description=(
        "Penalizes assistant responses that ask for clarification when a math query "
        "already contains sufficient information to proceed."
    ),
    parameters=NoUnnecessaryClarificationParams,
    handler=no_unnecessary_clarification_scorer,
)


# ---------------------------------------------------------------------------
# Golden eval scorers (ground-truth accuracy + trajectory fidelity)
# ---------------------------------------------------------------------------


class AnswerAccuracyParams(BaseModel):
    output: Any
    expected: Any


class TrajectoryFidelityParams(BaseModel):
    output: Any
    expected: Any


def _extract_numbers_from_text(text: str) -> list[float]:
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


def _get_response_text(output_payload: Any) -> str:
    if isinstance(output_payload, str):
        return output_payload
    if not isinstance(output_payload, dict):
        return str(output_payload)
    final = output_payload.get("final_output", "")
    if isinstance(final, str) and final.strip():
        return final.strip()
    messages = output_payload.get("messages", [])
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("content"):
                return str(msg["content"])
    return ""


def _infer_agents_from_output(output_payload: Any) -> set[str]:
    found: set[str] = set()
    if not isinstance(output_payload, dict):
        return found
    messages = output_payload.get("messages", [])
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


async def answer_accuracy_scorer(output: Any, expected: Any) -> dict[str, Any]:
    """Deterministic ground-truth accuracy check.

    answer_type options:
      contains_numeric – response contains a number within tolerance of expected.answer
      contains         – response contains expected.answer as a substring (case-insensitive)
      contains_all     – response contains every string in expected.answer_all
    """
    if not isinstance(expected, dict):
        return {"name": "Answer Accuracy", "score": None}

    answer_type = expected.get("answer_type", "contains")
    response_text = _get_response_text(output)

    if answer_type == "contains_numeric":
        raw_expected = expected.get("answer")
        if raw_expected is None:
            return {"name": "Answer Accuracy", "score": None}
        try:
            expected_value = float(str(raw_expected).replace(",", ""))
        except ValueError:
            return {"name": "Answer Accuracy", "score": None}
        tolerance = float(expected.get("tolerance") or 0.01)
        for num in _extract_numbers_from_text(response_text):
            if expected_value == 0:
                if abs(num - expected_value) < 1e-9:
                    return {"name": "Answer Accuracy", "score": 1.0}
            else:
                if abs(num - expected_value) / abs(expected_value) <= tolerance:
                    return {"name": "Answer Accuracy", "score": 1.0}
        return {"name": "Answer Accuracy", "score": 0.0}

    if answer_type == "contains":
        expected_str = str(expected.get("answer", "")).strip()
        found = expected_str.lower() in response_text.lower()
        return {"name": "Answer Accuracy", "score": 1.0 if found else 0.0}

    if answer_type == "contains_all":
        required = expected.get("answer_all", [])
        if not isinstance(required, list) or not required:
            return {"name": "Answer Accuracy", "score": None}
        hits = sum(1 for item in required if str(item).lower() in response_text.lower())
        return {"name": "Answer Accuracy", "score": hits / len(required)}

    return {"name": "Answer Accuracy", "score": None}


async def trajectory_fidelity_scorer(output: Any, expected: Any) -> dict[str, Any]:
    """Verify the system called the agents required by the golden test case.

    1.0 → all expected agents were called
    0–1 → partial: some expected agents were called (fraction)
    0.0 → none of the expected agents were called
    1.0 → always, when expected_agents is empty (direct answer permitted)
    """
    if not isinstance(expected, dict):
        return {"name": "Trajectory Fidelity", "score": None}

    expected_agents: list[str] = expected.get("expected_agents") or []
    if not expected_agents:
        return {"name": "Trajectory Fidelity", "score": 1.0}

    agents_called = _infer_agents_from_output(output)
    expected_set = set(expected_agents)
    hit = expected_set & agents_called
    missed = expected_set - agents_called

    if not missed:
        score = 1.0
    elif not hit:
        score = 0.0
    else:
        score = len(hit) / len(expected_set)
    return {"name": "Trajectory Fidelity", "score": score}


project.scorers.create(
    name="Answer Accuracy",
    slug="answer-accuracy",
    description=(
        "Deterministic ground-truth accuracy check for golden eval cases. "
        "Supports contains_numeric (within tolerance), contains (substring), "
        "and contains_all (all required substrings present) matching strategies."
    ),
    parameters=AnswerAccuracyParams,
    handler=answer_accuracy_scorer,
)

project.scorers.create(
    name="Trajectory Fidelity",
    slug="trajectory-fidelity",
    description=(
        "Verifies that the multi-agent system called the expected specialist agents "
        "(MathAgent, ResearchAgent) for each golden test case. "
        "Catches routing regressions caused by prompt drift or model upgrades."
    ),
    parameters=TrajectoryFidelityParams,
    handler=trajectory_fidelity_scorer,
)
