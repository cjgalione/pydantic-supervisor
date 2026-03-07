"""Research Agent evaluation focused on web search and information retrieval."""

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

from evals.parameters import ResearchAgentPromptParam, ResearchModelParam  # noqa: E402
from src.agents.research_agent import get_research_agent  # noqa: E402
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


async def run_research_task(input: dict, hooks: Any = None) -> dict:
    """Run a research query through the research agent."""
    try:
        params = hooks.parameters if hooks and hasattr(hooks, "parameters") else {}
        research_agent_prompt = _param_value(params.get("research_agent_prompt"), None)
        research_model = _param_value(params.get("research_model"), "gemini-2.0-flash-lite")

        agent = get_research_agent(
            system_prompt=research_agent_prompt,
            model=research_model,
        )
        query = str(input.get("query", ""))

        run_result = await run_adk_agent(
            agent=agent,
            query=query,
            app_name="pydantic-supervisor-eval-research",
        )
        serialized = run_result["messages"]

        tool_calls: list[str] = []
        for msg in serialized:
            for tc in msg.get("tool_calls", []):
                tool_calls.append(str(tc.get("name", "")))

        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update(
                {
                    "tool_calls": tool_calls,
                    "used_web_search": "tavily_search" in tool_calls,
                    "total_messages": len(serialized),
                }
            )

        return {"messages": serialized}
    except Exception as e:
        if hooks and hasattr(hooks, "metadata"):
            hooks.metadata.update({"error": str(e)})
        return {"messages": [{"error": str(e)}]}


RESEARCH_TEST_DATA = [
    {
        "input": {"query": "Who is the current president of France?"},
        "expected": {"should_use_search": True, "should_have_url": True},
    },
    {
        "input": {"query": "What is the capital of Japan?"},
        "expected": {"should_use_search": True, "should_have_url": True},
    },
    {
        "input": {"query": "When was the Eiffel Tower built?"},
        "expected": {"should_use_search": True, "should_have_url": True},
    },
    {
        "input": {"query": "What are the main causes of climate change?"},
        "expected": {"should_use_search": True, "should_have_url": True},
    },
]


async def web_search_usage_scorer(output, metadata=None):
    """Check if the agent used web search when appropriate."""
    if metadata and metadata.get("used_web_search"):
        return 1.0
    return 0.0


async def source_attribution_scorer(output):
    """Check if the response includes URL citations."""
    messages = output.get("messages", [])
    for msg in reversed(messages):
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        role = msg.get("role", "") if isinstance(msg, dict) else ""
        if content and role == "assistant":
            if re.search(r"https?://", content):
                return 1.0
            break
    return 0.0


async def efficiency_scorer(output, metadata=None):
    """Score based on minimal tool calls."""
    if not metadata:
        return 0.5

    num_searches = metadata.get("tool_calls", []).count("tavily_search")
    if num_searches == 1:
        return 1.0
    if num_searches == 2:
        return 0.9
    if num_searches <= 4:
        return 0.7
    return 0.5


answer_quality_prompt = """
You are evaluating a research agent's response to a factual question.

Question: {{input}}
Response: {{output}}

Evaluate the response on:
1. ACCURACY: Is the information correct and factual?
2. COMPLETENESS: Does it answer the question fully?
3. CLARITY: Is it well-structured and clear?
4. RELEVANCE: Does it address what was asked?

Respond with:
EXCELLENT - Accurate, complete, clear, and highly relevant
GOOD - Mostly accurate and complete with minor issues
FAIR - Some accuracy or completeness issues
POOR - Inaccurate, incomplete, or irrelevant
"""

answer_quality_scorer = LLMClassifier(
    name="Answer Quality",
    prompt_template=answer_quality_prompt,
    choice_scores={"EXCELLENT": 1.0, "GOOD": 0.75, "FAIR": 0.5, "POOR": 0.0},
    use_cot=True,
    model=os.environ.get("EVAL_JUDGE_MODEL", "gpt-4o"),
)


Eval(
    os.environ.get("BRAINTRUST_PROJECT", "pydantic-supervisor"),
    experiment_name="research-agent",
    data=RESEARCH_TEST_DATA,  # type: ignore
    task=run_research_task,
    scores=[
        web_search_usage_scorer,
        source_attribution_scorer,
        efficiency_scorer,
        answer_quality_scorer,
    ],  # type: ignore
    parameters={
        "research_agent_prompt": ResearchAgentPromptParam,
        "research_model": ResearchModelParam,
    },
)
