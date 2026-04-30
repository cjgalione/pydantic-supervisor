"""Research agent with Tavily web search capabilities."""

import os
from typing import Any, Callable

from pydantic_ai import Agent
from tavily import TavilyClient

from src.config import DEFAULT_RESEARCH_AGENT_PROMPT, DEFAULT_RESEARCH_MODEL
from src.modeling import resolve_model_name


def _get_tavily_client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def tavily_search(query: str, max_results: int = 3) -> str:
    """Search the web with Tavily and return summarized results with links."""
    limited_max_results = max(1, min(max_results, 5))
    try:
        response: dict[str, Any] = _get_tavily_client().search(
            query=query,
            max_results=limited_max_results,
            include_answer=True,
            include_raw_content=False,
        )
    except Exception as exc:
        error_text = str(exc).strip()
        lowered = error_text.lower()
        if "usage limit" in lowered or "forbidden" in lowered:
            return (
                "Web search is temporarily unavailable because the Tavily quota is exhausted. "
                "Proceed with a best-effort response and note that live sources could not be fetched."
            )
        return f"Web search failed: {error_text or type(exc).__name__}"

    lines: list[str] = []
    answer = response.get("answer")
    if answer:
        lines.append(f"Answer: {answer}")

    results = response.get("results", []) or []
    if not results:
        if lines:
            return "\n\n".join(lines)
        return "No search results found."

    for i, item in enumerate(results, start=1):
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()
        content = str(item.get("content", "")).strip()
        block = (
            f"{i}. {title or 'Untitled'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Summary: {content or 'N/A'}"
        )
        lines.append(block)

    return "\n\n".join(lines)


def _register_tools(agent: Agent, tools: list[Callable[..., Any]]) -> None:
    for tool in tools:
        agent.tool_plain(name=tool.__name__)(tool)


def get_research_agent(
    system_prompt: str | None = None,
    model: str = DEFAULT_RESEARCH_MODEL,
    extra_tools: list[Callable[..., Any]] | None = None,
) -> Agent:
    """Create the research agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_RESEARCH_AGENT_PROMPT

    tools: list[Callable[..., Any]] = [tavily_search]
    if extra_tools:
        tools.extend(extra_tools)

    agent = Agent(
        name="ResearchAgent",
        model=resolve_model_name(model),
        system_prompt=prompt,
    )
    _register_tools(agent, tools)
    return agent
