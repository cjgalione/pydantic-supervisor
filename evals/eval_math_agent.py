"""Math Agent evaluation focused on calculation accuracy and tool usage."""

import os
import re
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from autoevals import LLMClassifier  # noqa: E402
from braintrust import Eval  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from evals.parameters import MathAgentPromptParam, MathModelParam  # noqa: E402
from src.agents.math_agent import get_math_agent  # noqa: E402
from src.helpers import run_adk_agent  # noqa: E402
from src.tracing import configure_adk_tracing  # noqa: E402

load_dotenv()
configure_adk_tracing(
    api_key=os.environ.get("BRAINTRUST_API_KEY"),
    project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
    project_name=os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor"),
)


def _param_value(param: Any, default: Any) -> Any:
    if param is None:
        return default
    if hasattr(param, "value"):
        return getattr(param, "value")
    if isinstance(param, type):
        try:
            instance = param()
            if hasattr(instance, "value"):
                return getattr(instance, "value")
        except Exception:
            return default
    return param


async def run_math_task(input: dict, hooks: Any = None) -> dict:
    """Run a math calculation through the math agent."""
    try:
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}
        math_agent_prompt = _param_value(params.get("math_agent_prompt"), None)
        math_model = _param_value(params.get("math_model"), "gemini-2.0-flash-lite")

        agent = get_math_agent(system_prompt=math_agent_prompt, model=math_model)
        query = str(input.get("query", ""))

        run_result = await run_adk_agent(
            agent=agent,
            query=query,
            app_name="pydantic-supervisor-eval-math",
        )
        serialized = run_result["messages"]

        tool_calls: list[dict[str, Any]] = []
        for msg in serialized:
            for tc in msg.get("tool_calls", []):
                tool_calls.append(
                    {
                        "name": tc.get("name", ""),
                        "args": tc.get("args", {}),
                    }
                )

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "tool_calls": tool_calls,
                    "total_messages": len(serialized),
                }
            )

        return {"messages": serialized}
    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


MATH_TEST_DATA = [
    {"input": {"query": "What is 25 + 17?", "expected_answer": 42}},
    {"input": {"query": "Calculate 100 - 37", "expected_answer": 63}},
    {"input": {"query": "What is 12 * 8?", "expected_answer": 96}},
    {"input": {"query": "Divide 144 by 12", "expected_answer": 12}},
    {"input": {"query": "What's 15 * 7 + 3?", "expected_answer": 108}},
    {"input": {"query": "Calculate (50 + 30) / 4", "expected_answer": 20}},
]


async def calculation_accuracy_scorer(input, output, expected):
    """Check if the calculated answer matches the expected value."""
    if not expected or "expected_answer" not in expected:
        return 0.5

    expected_answer = expected["expected_answer"]
    messages = output.get("messages", [])

    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if content and role == "assistant":
            if str(expected_answer) in str(content):
                return 1.0
            break

    return 0.0


async def tool_usage_scorer(output, metadata=None):
    """Check if the agent used math tools appropriately."""
    if not metadata:
        return 0.5

    tool_calls = metadata.get("tool_calls", [])
    if not tool_calls:
        return 0.0

    tool_names = [tc["name"] for tc in tool_calls]
    valid_tools = {"add", "subtract", "multiply", "divide"}
    used_valid_tools = any(name in valid_tools for name in tool_names)
    return 1.0 if used_valid_tools else 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal unnecessary tool calls."""
    if not metadata:
        return 0.5

    num_calls = len(metadata.get("tool_calls", []))
    if num_calls <= 2:
        return 1.0
    if num_calls <= 4:
        return 0.8
    if num_calls <= 6:
        return 0.6
    return 0.4


async def response_format_scorer(output):
    """Check if the response is clear and includes the final answer."""
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""

        if content and role == "assistant":
            if re.search(r"\d+", content):
                return 1.0
            break
    return 0.0


calculation_correctness_prompt = """
You are evaluating a math agent's calculation.

Question: {{input}}
Agent's Response: {{output}}
Expected Answer: {{expected}}

Evaluate whether:
1. The calculation is mathematically correct
2. The final answer matches the expected result
3. The reasoning (if shown) is sound

Respond with:
CORRECT - Calculation and answer are correct
INCORRECT - Calculation or answer is wrong
"""

calculation_correctness_scorer = LLMClassifier(
    name="Calculation Correctness",
    prompt_template=calculation_correctness_prompt,
    choice_scores={"CORRECT": 1.0, "INCORRECT": 0.0},
    use_cot=True,
    model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
)


Eval(
    os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor"),
    experiment_name="math-agent",
    data=MATH_TEST_DATA,  # type: ignore
    task=run_math_task,
    scores=[
        calculation_accuracy_scorer,
        tool_usage_scorer,
        efficiency_scorer,
        response_format_scorer,
        calculation_correctness_scorer,
    ],  # type: ignore
    parameters={
        "math_agent_prompt": MathAgentPromptParam,
        "math_model": MathModelParam,
    },
)
