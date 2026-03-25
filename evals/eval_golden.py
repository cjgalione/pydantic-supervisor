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
  Behavioural  (routing, delegation compliance, efficiency)
  Quality      (response quality LLM judge)
  Accuracy     (ground-truth matching) <- this file
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from braintrust import Eval, init_dataset, init_function  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from evals.braintrust_gateway_header_patch import apply_gateway_header_patch  # noqa: E402
from evals.braintrust_parameter_patch import apply_parameter_patch  # noqa: E402
from evals.parameters import (  # noqa: E402
    MathAgentPromptParam,
    ResearchAgentPromptParam,
    SystemPromptParam,
)
from evals.shared import (  # noqa: E402
    extract_numbers_from_text,
    infer_agents_from_messages,
    is_error_output,
    latest_assistant_text,
    parse_with_gateway_metadata,
    run_supervisor_task as _run_supervisor_task,
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


# ---------------------------------------------------------------------------
# Task function
# ---------------------------------------------------------------------------

async def run_supervisor_task(input: dict, hooks: Any = None) -> dict[str, Any]:
    return await _run_supervisor_task(
        input,
        hooks,
        app_name="pydantic-supervisor-eval-golden",
        extra_metadata_keys=("critic_corrected",),
    )


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
    """
    del metadata, trace

    if is_error_output(output):
        return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "error output"}}

    if not isinstance(expected, dict):
        return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "no expected field"}}

    answer_type = expected.get("answer_type", "contains")
    response_text = latest_assistant_text(output)

    # -- contains_numeric --
    if answer_type == "contains_numeric":
        raw_expected = expected.get("answer")
        if raw_expected is None:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "missing answer"}}
        try:
            expected_value = float(str(raw_expected).replace(",", ""))
        except ValueError:
            return {"name": "Answer Accuracy", "score": None, "metadata": {"reason": "non-numeric expected answer"}}

        tolerance = float(expected.get("tolerance") or 0.01)
        numbers_found = extract_numbers_from_text(response_text)

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

    # -- contains --
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

    # -- contains_all --
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
    """Verify that the system called the agents expected for this test case."""
    del metadata, trace

    if is_error_output(output):
        return {"name": "Trajectory Fidelity", "score": None, "metadata": {"reason": "error output"}}

    if not isinstance(expected, dict):
        return {"name": "Trajectory Fidelity", "score": None, "metadata": {"reason": "no expected field"}}

    expected_agents: list[str] = expected.get("expected_agents") or []

    if not expected_agents:
        return {
            "name": "Trajectory Fidelity",
            "score": 1.0,
            "metadata": {"expected_agents": [], "note": "direct answer permitted"},
        }

    agents_called = infer_agents_from_messages(output)
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
    """LLM judge anchored to the known correct answer."""
    del metadata, trace

    if is_error_output(output):
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

    ai_response = latest_assistant_text(output)

    prompt = ANSWER_GROUNDING_PROMPT.format(
        question=question,
        expected_answer=expected_answer_str,
        ai_response=ai_response or "(no response)",
    )

    response, gateway_metadata = parse_with_gateway_metadata(
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

def _register_eval():
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


# Register the eval at import time (required by braintrust eval's lazy-load discovery).
# Set SKIP_EVAL_REGISTRATION=1 to import scorers without triggering the eval.
if not os.environ.get("SKIP_EVAL_REGISTRATION"):
    _register_eval()
