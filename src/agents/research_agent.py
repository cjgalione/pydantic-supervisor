"""Research agent with web search capabilities."""

import json
import os
from typing import Any, Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic_ai import Agent
from tavily import TavilyClient

from src.config import DEFAULT_RESEARCH_AGENT_PROMPT, DEFAULT_RESEARCH_MODEL
from src.modeling import resolve_model_name


def _get_tavily_client() -> TavilyClient:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)


def _get_exa_client() -> Any:
    api_key = os.environ.get("EXA_API_KEY")
    if not api_key:
        raise RuntimeError("EXA_API_KEY is not set")

    from exa_py import Exa

    return Exa(api_key=api_key)


def _get_you_api_key() -> str:
    for env_name in ("YDC_API_KEY", "YOU_API_KEY", "YOUCOM_API_KEY"):
        api_key = os.environ.get(env_name)
        if api_key:
            return api_key
    raise RuntimeError("YDC_API_KEY is not set")


def _provider_has_key(provider: str) -> bool:
    if provider == "exa":
        return bool(os.environ.get("EXA_API_KEY"))
    if provider == "tavily":
        return bool(os.environ.get("TAVILY_API_KEY"))
    if provider == "you":
        return any(os.environ.get(name) for name in ("YDC_API_KEY", "YOU_API_KEY", "YOUCOM_API_KEY"))
    return False


def _normalize_search_provider(provider: str | None) -> str:
    normalized = (provider or "exa").strip().lower().replace("_", "-")
    aliases = {
        "you.com": "you",
        "youcom": "you",
        "ydc": "you",
    }
    return aliases.get(normalized, normalized)


def _search_provider_order() -> list[str]:
    preferred = _normalize_search_provider(
        os.environ.get("SEARCH_PROVIDER") or os.environ.get("WEB_SEARCH_PROVIDER")
    )
    providers = ["exa", "tavily", "you"]
    ordered = [preferred] if preferred in providers else ["exa"]
    ordered.extend(
        provider
        for provider in providers
        if provider not in ordered and _provider_has_key(provider)
    )
    return ordered


def _build_tavily_output(response: dict[str, Any]) -> str:
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


def _result_value(result: Any, name: str, default: Any = "") -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _build_exa_output(response: Any) -> str:
    lines: list[str] = []
    results = _result_value(response, "results", []) or []

    if not results:
        return "No search results found."

    for i, result in enumerate(results, start=1):
        title = str(_result_value(result, "title", "") or "").strip()
        url = str(_result_value(result, "url", "") or "").strip()
        highlights = _result_value(result, "highlights", []) or []
        summary = str(_result_value(result, "summary", "") or "").strip()
        text = str(_result_value(result, "text", "") or "").strip()

        if isinstance(highlights, str):
            content = highlights.strip()
        else:
            content = " ".join(str(item).strip() for item in highlights if str(item).strip())

        if not content:
            content = summary or text[:800]

        block = (
            f"{i}. {title or 'Untitled'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Summary: {content or 'N/A'}"
        )
        lines.append(block)
    return "\n\n".join(lines)


def _build_you_output(response: dict[str, Any], max_results: int) -> str:
    lines: list[str] = []
    grouped_results = response.get("results", {}) or {}
    results: list[tuple[str, dict[str, Any]]] = []

    if isinstance(grouped_results, dict):
        for result_type in ("web", "news"):
            for item in grouped_results.get(result_type, []) or []:
                if isinstance(item, dict):
                    results.append((result_type, item))

    if not results:
        return "No search results found."

    for i, (result_type, item) in enumerate(results[:max_results], start=1):
        title = str(item.get("title", "") or "").strip()
        url = str(item.get("url", "") or "").strip()
        description = str(item.get("description", "") or "").strip()
        snippets = item.get("snippets", []) or []
        if isinstance(snippets, str):
            content = snippets.strip()
        else:
            content = " ".join(
                str(snippet).strip() for snippet in snippets[:2] if str(snippet).strip()
            )
        content = content or description
        block = (
            f"{i}. {title or 'Untitled'}\n"
            f"URL: {url or 'N/A'}\n"
            f"Source: You.com {result_type}\n"
            f"Summary: {content or 'N/A'}"
        )
        lines.append(block)
    return "\n\n".join(lines)


def _search_tavily(query: str, max_results: int) -> str:
    response: dict[str, Any] = _get_tavily_client().search(
        query=query,
        max_results=max_results,
        include_answer=True,
        include_raw_content=False,
    )
    return _build_tavily_output(response)


def _search_exa(query: str, max_results: int) -> str:
    response = _get_exa_client().search(
        query,
        type="auto",
        num_results=max_results,
        contents={"highlights": True},
    )
    return _build_exa_output(response)


def _search_you(query: str, max_results: int) -> str:
    params = urlencode({"query": query, "count": max_results})
    request = Request(
        f"https://ydc-index.io/v1/search?{params}",
        headers={
            "X-API-Key": _get_you_api_key(),
            "User-Agent": "supervisor-demos/1.0",
        },
    )
    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return _build_you_output(payload, max_results=max_results)


def _search_with_provider(provider: str, query: str, max_results: int) -> str:
    if provider == "exa":
        return _search_exa(query=query, max_results=max_results)
    if provider == "tavily":
        return _search_tavily(query=query, max_results=max_results)
    if provider == "you":
        return _search_you(query=query, max_results=max_results)
    raise RuntimeError(f"Unsupported search provider: {provider}")


def tavily_search(query: str, max_results: int = 3) -> str:
    """Search the web and return summarized results with links."""
    limited_max_results = max(1, min(max_results, 5))
    errors: list[str] = []

    for provider in _search_provider_order():
        try:
            return _search_with_provider(
                provider=provider,
                query=query,
                max_results=limited_max_results,
            )
        except Exception as exc:
            errors.append(f"{provider}: {exc}")

    error_text = "; ".join(errors).strip()
    lowered = error_text.lower()
    if "usage limit" in lowered or "forbidden" in lowered:
        return (
            "Web search is temporarily unavailable because the search provider quota is exhausted. "
            "Proceed with a best-effort response and note that live sources could not be fetched."
        )
    return f"Web search failed: {error_text or 'no configured search provider succeeded'}"


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
