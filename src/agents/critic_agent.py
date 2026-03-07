"""Critic agent that validates delegation/tool-use policy compliance."""

from pydantic_ai import Agent

from src.modeling import resolve_model_name

DEFAULT_CRITIC_AGENT_PROMPT = (
    "You are CriticAgent. Validate whether the candidate answer follows delegation policy.\n"
    "Policy:\n"
    "- Math-like queries MUST involve MathAgent handoff/tool usage evidence.\n"
    "- Factual/latest/source-seeking queries MUST involve ResearchAgent handoff and web-search evidence.\n"
    "- If policy-triggering query was answered directly without required delegation, reject.\n"
    "Return JSON ONLY with schema:\n"
    '{"compliant": true|false, "required_action": "accept"|"delegate_research"|"delegate_math"|"retry_with_instruction", "rationale": "short explanation"}\n'
    "No markdown or extra keys."
)


def get_critic_agent(
    system_prompt: str | None = None,
    model: str = "gemini-2.0-flash-lite",
) -> Agent:
    """Create the critic agent."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_CRITIC_AGENT_PROMPT
    return Agent(
        name="CriticAgent",
        model=resolve_model_name(model),
        system_prompt=prompt,
    )
