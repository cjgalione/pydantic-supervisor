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
