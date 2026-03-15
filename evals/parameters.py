"""
Parameter definitions for Braintrust evals.

Prompt-bearing parameters are represented as Braintrust prompt objects so the
playground can render prompt editors with model selection instead of plain text
inputs. Plain scalar parameters continue to use single-field Pydantic models.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.config import (
    DEFAULT_MATH_AGENT_PROMPT,
    DEFAULT_MATH_MODEL,
    DEFAULT_RESEARCH_AGENT_PROMPT,
    DEFAULT_RESEARCH_MODEL,
    DEFAULT_SUPERVISOR_MODEL,
    DEFAULT_SYSTEM_PROMPT,
)

def _prompt_parameter(*, prompt: str, model: str, description: str) -> dict[str, Any]:
    """Build a Braintrust prompt parameter with a default system prompt and model."""
    return {
        "type": "prompt",
        "description": description,
        "default": {
            "prompt": {
                "type": "chat",
                "messages": [
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
            },
            "options": {
                "model": model,
            },
        },
    }


def extract_prompt_and_model(
    param: Any,
    *,
    default_prompt: str,
    default_model: str,
) -> tuple[str, str]:
    """Extract a system prompt string and model name from a Braintrust prompt object."""
    if param is None:
        return default_prompt, default_model

    prompt_text = default_prompt
    model_name = default_model

    prompt_block = getattr(param, "prompt", None)
    if prompt_block is not None:
        messages = getattr(prompt_block, "messages", None) or []
        for message in messages:
            if getattr(message, "role", None) != "system":
                continue

            content = getattr(message, "content", None)
            if isinstance(content, str):
                prompt_text = content
                break

            if isinstance(content, list):
                text_parts: list[str] = []
                for part in content:
                    text = getattr(part, "text", None)
                    if isinstance(text, str) and text:
                        text_parts.append(text)
                if text_parts:
                    prompt_text = "\n".join(text_parts)
                    break

    options = getattr(param, "options", None) or {}
    if isinstance(options, dict):
        maybe_model = options.get("model")
        if isinstance(maybe_model, str) and maybe_model.strip():
            model_name = maybe_model

    return prompt_text, model_name


# Define scalar parameters as single-field Pydantic models.
# The patched SDK will extract the 'value' field's schema and default.


SystemPromptParam = _prompt_parameter(
    prompt=DEFAULT_SYSTEM_PROMPT,
    model=DEFAULT_SUPERVISOR_MODEL,
    description="Prompt object for the supervisor agent, including its model.",
)


class PromptModificationParam(BaseModel):
    """Append-only supervisor prompt modification parameter."""

    value: str = Field(
        default="",
        description=(
            "Optional append-only modification for the supervisor prompt. "
            "Use this to tune routing criteria without replacing the full base prompt."
        ),
    )


ResearchAgentPromptParam = _prompt_parameter(
    prompt=DEFAULT_RESEARCH_AGENT_PROMPT,
    model=DEFAULT_RESEARCH_MODEL,
    description="Prompt object for the research agent, including its model.",
)


MathAgentPromptParam = _prompt_parameter(
    prompt=DEFAULT_MATH_AGENT_PROMPT,
    model=DEFAULT_MATH_MODEL,
    description="Prompt object for the math agent, including its model.",
)
