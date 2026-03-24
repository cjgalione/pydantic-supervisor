"""
Golden dataset eval: ground-truth accuracy + trajectory fidelity.

This eval answers a question the other eval suites cannot: "Is the system
actually producing *correct* answers?"  Every test case in the golden dataset
carries an expected answer and an expected agent trajectory.  Three new scorers
measure the dimensions that matter most for production reliability:

  answer_accuracy_scorer  – deterministic match against the expected answer
                            (numeric within tolerance, substring, or all-of check)
  trajectory_fidelity_scorer – did the right agents get called?
  answer_grounding_scorer – LLM judge anchored to the known-correct answer,
                            so the judge is calibrated rather than free-form

Together these complement the behavioural scorers in eval_supervisor.py and
complete the eval trinity:
  ✅ Behavioural  (routing, delegation compliance, efficiency)
  ✅ Quality      (response quality LLM judge)
  ✅ Accuracy     (ground-truth matching) ← this file
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Literal

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, init_dataset, init_function  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.braintrust_gateway_header_patch import apply_gateway_header_patch  # noqa: E402
from evals.braintrust_parameter_patch import apply_parameter_patch  # noqa: E402
from evals.parameters import (  # noqa: E402
    MathAgentPromptParam,
    ResearchAgentPromptParam,
    SystemPromptParam,
    extract_prompt_and_model,
)
from src.agents.deep_agent import get_supervisor, run_supervisor_with_critic  # noqa: E402
from src.config import (  # noqa: E402
    AgentConfig,
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)
from src.helpers import extract_query_from_input  # noqa: E402
from src.tracing import configure_adk_tracing  # noqa: E402

load_dotenv()
apply_parameter_patch()
apply_gateway_header_patch()

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"
DEFAULT_BRAINTRUST_DATASET = "Golden Dataset"

configure_adk_tracing(
    api_key=os.environ.get("BRAINTRUST_API_KEY"),
    project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
    project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
)

judge_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    default_headers={"x-bt-use-cache": "always"},
)


# ---------------------------------------------------------------------------
# Shared helpers (mirrors eval_supervisor.py conventions)
# ---------------------------------------------------------------------------

def _extract_gateway_headers(headers: Any) -> dict[str, str]:
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


def _parse_with_gateway_metadata(
    *, model: str, input_data: list[dict[str, Any]], text_format: Any
) -> tuple[Any, dict[str, str]]:
    raw_response = judge_client.responses.with_raw_response.parse(
        model=model,
        input=input_data,
        text_format=text_format,
    )
    gateway_metadata = _extract_gateway_headers(getattr(raw_response, "headers", None))
    return raw_response.parse(), gateway_metadata


def _is_error_output(output: Any) -> bool:
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


def _latest_assistant_text(output: Any) -> str:
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


def _extract_numbers_from_text(text: str) -> list[float]:
    """Extract all numeric values from a text string."""
    # Handle scientific notation with caret (e.g., "3 x 10^5")
    sci_caret = re.findall(r"(-?\d+(?:\.\d+)?)\s*[x×]\s*10\^(-?\d+)", text, flags=re.IGNORECASE)
    numbers: list[float] = []
    for base_s, exp_s in sci_caret:
        try:
            numbers.append(float(base_s) * (10 ** int(exp_s)))
        except ValueError:
            pass

    # Standard numbers including scientific notation
    raw_numbers = re.findall(r"-?\d+(?:[,_]\d+)*(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    for n in raw_numbers:
        try:
            numbers.append(float(n.replace(",", "").replace("_", "")))
        except ValueError:
            pass
    return numbers


def _infer_agents_from_messages(output: Any) -> set[str]:
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
# Task function
# ---------------------------------------------------------------------------

def _unwrap_parameters(params: dict) -> dict:
    import inspect

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
    return {
        "system_prompt": system_prompt,
        "prompt_modification": "",
        "research_agent_prompt": research_agent_prompt,
        "math_agent_prompt": math_agent_prompt,
        "supervisor_model": supervisor_model,
        "research_model": research_model,
        "math_model": math_model,
    }


async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, Any]:
    """Run a single golden test case through the full supervisor + critic pipeline."""
    try:
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}
        config_params = _unwrap_parameters(params)
        config = AgentConfig(**config_params) if config_params else None

        supervisor = get_supervisor(config=config, force_rebuild=True)
        query = extract_query_from_input(input)

        run_result = await run_supervisor_with_critic(
            supervisor=supervisor,
            query=query,
            app_name="pydantic-supervisor-eval-golden",
        )
        serialized_messages = run_result["messages"]

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "final_output": run_result.get("final_output", ""),
                    "num_messages": len(serialized_messages),
                    "critic_corrected": run_result.get("critic_corrected", False),
                }
            )

        return {
            "final_output": run_result.get("final_output", ""),
            "messages": serialized_messages,
        }
    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"final_output": "", "messages": [{"error": str(e)}]}


# ---------------------------------------------------------------------------
# Scorer 1: Answer Accuracy (deterministic)
# ---------------------------------------------------------------------------

async def answer_accuracy_scorer(
    input: Any, output: Any, expected: Any, metadata: Any, trace: Any
) -> dict[str, Any]:
    """Deterministic accuracy check against the known-correct expected answer.

    Supports three matching strategies set by ``expected.answer_type``:

    * ``contains_numeric`` – the response must contain a number within
      ``expected.tolerance`` (relative) of ``expected.answer``.
    * ``contains`` – the response must contain ``expected.answer`` as a
      case-insensitive substring.
    * ``contains_all`` – the response must contain every string in
      ``expected.answer_all`` (case-insensitive).

    This is the fastest and cheapest scorer: no LLM call, pure text matching.
    It catches regressions that LLM judges routinely miss.
    """
    del metadata, trace

    if _is_error_output(output):
        return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "error output"}}

    if not isinstance(expected, dict):
        return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "no expected field"}}

    answer_type = expected.get("answer_type", "contains")
    response_text = _latest_assistant_text(output)

    # ── contains_numeric ────────────────────────────────────────────────────
    if answer_type == "contains_numeric":
        raw_expected = expected.get("answer")
        if raw_expected is None:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "missing answer"}}
        try:
            expected_value = float(str(raw_expected).replace(",", ""))
        except ValueError:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "non-numeric expected answer"}}

        tolerance = float(expected.get("tolerance") or 0.01)
        numbers_found = _extract_numbers_from_text(response_text)

        for num in numbers_found:
            if expected_value == 0:
                if abs(num - expected_value) < 1e-9:
                    return {
                        "name": "Answer Accuracy",
                        "score": 1.0,
                        "metadata": {"expected": expected_value, "found": num, "tolerance_pct": tolerance},
                    }
            else:
                relative_error = abs(num - expected_value) / abs(expected_value)
                if relative_error <= tolerance:
                    return {
                        "name": "Answer Accuracy",
                        "score": 1.0,
                        "metadata": {
                            "expected": expected_value,
                            "found": num,
                            "relative_error": relative_error,
                            "tolerance_pct": tolerance,
                        },
                    }

        return {
            "name": "Answer Accuracy",
            "score": 0.0,
            "metadata": {
                "expected": expected_value,
                "numbers_found_in_response": numbers_found[:10],
                "response_snippet": response_text[:300],
            },
        }

    # ── contains ────────────────────────────────────────────────────────────
    if answer_type == "contains":
        expected_str = str(expected.get("answer", "")).strip()
        if not expected_str:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "empty answer"}}
        found = expected_str.lower() in response_text.lower()
        return {
            "name": "Answer Accuracy",
            "score": 1.0 if found else 0.0,
            "metadata": {
                "expected_substring": expected_str,
                "found": found,
                "response_snippet": response_text[:300],
            },
        }

    # ── contains_all ────────────────────────────────────────────────────────
    if answer_type == "contains_all":
        required = expected.get("answer_all", [])
        if not isinstance(required, list) or not required:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "empty answer_all"}}
        results = {item: item.lower() in response_text.lower() for item in required}
        all_found = all(results.values())
        fraction = sum(1 for v in results.values() if v) / len(results)
        return {
            "name": "Answer Accuracy",
            "score": 1.0 if all_found else fraction,
            "metadata": {
                "required": required,
                "found_map": results,
                "response_snippet": response_text[:300],
            },
        }

    return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": f"unknown answer_type: {answer_type}"}}


# ---------------------------------------------------------------------------
# Scorer 2: Trajectory Fidelity (deterministic)
# ---------------------------------------------------------------------------

async def trajectory_fidelity_scorer(
    input: Any, output: Any, expected: Any, metadata: Any, trace: Any
) -> dict[str, Any]:
    """Verify that the system called the agents expected for this test case.

    A score of 1.0 means all expected agents were invoked.
    A partial score (0.5) means some but not all expected agents were invoked.
    A score of 0.0 means none of the required agents were invoked.

    When ``expected_agents`` is empty the supervisor is allowed to answer
    directly, so the score is always 1.0 regardless of what was called.

    This scorer makes it possible to catch prompt-drift or model-update
    regressions where a capable model starts short-cutting compound queries
    by answering directly without the required research or math delegation.
    """
    del metadata, trace

    if _is_error_output(output):
        return {"name": "Trajectory Fidelity", "score": None, "metadata": {"reason": "error output"}}

    if not isinstance(expected, dict):
        return {"name": "Trajectory Fidelity", "score": None, "metadata": {"reason": "no expected field"}}

    expected_agents: list[str] = expected.get("expected_agents") or []

    # No required agents → any trajectory is acceptable
    if not expected_agents:
        return {
            "name": "Trajectory Fidelity",
            "score": 1.0,
            "metadata": {"expected_agents": [], "note": "direct answer permitted"},
        }

    agents_called = _infer_agents_from_messages(output)
    expected_set = set(expected_agents)
    hit = expected_set & agents_called
    missed = expected_set - agents_called

    if not missed:
        score = 1.0
    elif not hit:
        score = 0.0
    else:
        score = len(hit) / len(expected_set)

    return {
        "name": "Trajectory Fidelity",
        "score": score,
        "metadata": {
            "expected_agents": sorted(expected_set),
            "agents_called": sorted(agents_called),
            "hit": sorted(hit),
            "missed": sorted(missed),
        },
    }


# ---------------------------------------------------------------------------
# Scorer 3: Answer Grounding (LLM judge anchored to expected answer)
# ---------------------------------------------------------------------------

ANSWER_GROUNDING_PROMPT = """
You are a precise evaluator checking whether an AI assistant's response correctly answers a question.

You have access to the KNOWN CORRECT ANSWER — use it as the ground truth.

User Question: {question}

Known Correct Answer: {expected_answer}

AI Response: {ai_response}

Evaluate whether the AI response:
1. Contains or correctly conveys the known correct answer
2. Does not contradict the known correct answer
3. Is appropriately specific (not just vague or evasive)

Scoring guidance:
- CORRECT: The AI response contains the known correct answer or an equivalent form of it
- PARTIALLY_CORRECT: The AI response is on the right track but missing the key value, or gives a close but inexact answer
- INCORRECT: The AI response contradicts the known correct answer or gives a clearly wrong value

Always provide your reasoning before selecting a choice.
""".strip()


class AnswerGroundingOutput(BaseModel):
    """Structured output for answer grounding evaluation."""

    choice: Literal["CORRECT", "PARTIALLY_CORRECT", "INCORRECT"]
    reasoning: str


async def answer_grounding_scorer(
    input: Any, output: Any, expected: Any, metadata: Any, trace: Any
) -> dict[str, Any]:
    """LLM judge that scores whether the response contains the expected answer.

    Unlike a free-form LLM judge, this scorer is *anchored* to the known
    correct answer.  This dramatically reduces hallucinated scores: the judge
    does not need to independently know the right answer — it only needs to
    verify that the AI's response matches the provided ground truth.

    This produces a well-calibrated signal even for obscure facts or
    calculations where a free-form judge would be unreliable.
    """
    del metadata, trace

    if _is_error_output(output):
        return {"name": "Answer Grounding", "score": None, "metadata": {"reason": "error output"}}

    if not isinstance(expected, dict):
        return {"name": "Answer Grounding", "score": None, "metadata": {"reason": "no expected field"}}

    answer_type = expected.get("answer_type", "contains")
    if answer_type == "contains_all":
        expected_answer_str = " and ".join(str(a) for a in (expected.get("answer_all") or []))
    else:
        expected_answer_str = str(expected.get("answer", "")).strip()

    if not expected_answer_str:
        return {"name": "Answer Grounding", "score": None, "metadata": {"reason": "empty expected answer"}}

    if isinstance(input, dict):
        try:
            question = extract_query_from_input(input)
        except Exception:
            question = str(input)
    else:
        question = str(input)

    ai_response = _latest_assistant_text(output)

    prompt = ANSWER_GROUNDING_PROMPT.format(
        question=question,
        expected_answer=expected_answer_str,
        ai_response=ai_response or "(no response)",
    )

    response, gateway_metadata = _parse_with_gateway_metadata(
        model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
        input_data=[{"role": "user", "content": prompt}],
        text_format=AnswerGroundingOutput,
    )
    parsed = response.output_parsed
    if parsed is None:
        meta = {"choice": "INCORRECT", "reasoning": "No parsed output"}
        meta.update(gateway_metadata)
        return {"name": "Answer Grounding", "score": 0.0, "metadata": meta}

    score_map = {"CORRECT": 1.0, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0.0}
    meta = {
        "choice": parsed.choice,
        "reasoning": parsed.reasoning,
        "expected_answer": expected_answer_str,
    }
    meta.update(gateway_metadata)
    return {
        "name": "Answer Grounding",
        "score": score_map[parsed.choice],
        "metadata": meta,
    }


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_golden_dataset() -> list[dict[str, Any]]:
    """Load the curated golden dataset from datasets/golden_dataset.jsonl."""
    dataset_path = project_root / "datasets" / "golden_dataset.jsonl"
    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def get_eval_data(project_name: str):
    """Choose local golden dataset by default, with optional remote override."""
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
    return load_golden_dataset()


# ---------------------------------------------------------------------------
# Eval registration
# ---------------------------------------------------------------------------

project_name = os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT)

use_published_step_scorer = (
    os.environ.get("USE_PUBLISHED_STEP_SCORER", "1").lower() in {"1", "true", "yes"}
)
step_efficiency_score = (
    init_function(project_name=project_name, slug="step-efficiency")
    if use_published_step_scorer
    else None
)

scorers: list[Any] = [
    answer_accuracy_scorer,
    trajectory_fidelity_scorer,
    answer_grounding_scorer,
]
if step_efficiency_score is not None:
    scorers.append(step_efficiency_score)

Eval(
    project_name,
    experiment_name="golden-ground-truth",
    data=get_eval_data(project_name),
    task=run_supervisor_task,
    scores=scorers,  # type: ignore
    parameters={
        "system_prompt": SystemPromptParam,
        "research_agent_prompt": ResearchAgentPromptParam,
        "math_agent_prompt": MathAgentPromptParam,
    },
)
