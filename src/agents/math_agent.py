"""Math agent with arithmetic capabilities."""

import re
from typing import Any, Callable

from pydantic_ai import Agent
from pint import UnitRegistry

from src.config import DEFAULT_MATH_AGENT_PROMPT, DEFAULT_MATH_MODEL
from src.modeling import resolve_model_name

_UREG = UnitRegistry()
_UNIT_ALIASES = {
    "j": "joule",
    "joules": "joule",
    "hp": "horsepower",
    "horsepower-seconds": "horsepower * second",
    "horsepower seconds": "horsepower * second",
    "horsepower-seconds(s)": "horsepower * second",
    "horsepower-hours": "horsepower * hour",
    "horsepower hours": "horsepower * hour",
    "hp-s": "horsepower * second",
    "hp*s": "horsepower * second",
    "hp-hr": "horsepower * hour",
    "hp*h": "horsepower * hour",
}
_LIGHTBULB_HOURS_PATTERN = re.compile(
    r"(?P<watts>\d+(?:\.\d+)?)\s*w(?:att)?\s*lightbulb[-\s]*hours?",
    flags=re.IGNORECASE,
)


def _normalize_unit(unit: str) -> str:
    lowered = unit.strip().lower()
    return _UNIT_ALIASES.get(lowered, lowered)


def _trim_unit_expression(unit: str) -> str:
    # LLM tool arguments occasionally include explanatory text after punctuation.
    cleaned = unit.strip().strip("'\"")
    return re.split(r"[.\n]", cleaned, maxsplit=1)[0].strip()


def _convert_to_lightbulb_hours(value: float, from_unit: str, to_unit: str) -> float | None:
    match = _LIGHTBULB_HOURS_PATTERN.search(to_unit)
    if not match:
        return None

    watts = float(match.group("watts"))
    if watts <= 0:
        return None

    source_quantity = float(value) * _UREG.parse_units(_normalize_unit(from_unit))
    joules = source_quantity.to(_UREG.joule).magnitude
    joules_per_lightbulb_hour = watts * 3600.0
    return float(joules / joules_per_lightbulb_hour)


def add(a: float, b: float) -> float:
    """Add two numbers and return their sum."""
    return a + b


def subtract(a: float, b: float) -> float:
    """Subtract b from a and return the result."""
    return a - b


def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


def divide(a: float, b: float) -> float:
    """Divide a by b and return the quotient."""
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def convert_units(value: float, from_unit: str, to_unit: str) -> float:
    """Convert numeric values between compatible units."""
    source = _trim_unit_expression(from_unit)
    target = _trim_unit_expression(to_unit)

    lightbulb_hours = _convert_to_lightbulb_hours(value=value, from_unit=source, to_unit=target)
    if lightbulb_hours is not None:
        return lightbulb_hours

    normalized_source = _normalize_unit(source)
    normalized_target = _normalize_unit(target)
    quantity = float(value) * _UREG.parse_units(normalized_source)
    converted = quantity.to(_UREG.parse_units(normalized_target))
    return float(converted.magnitude)


def _register_tools(agent: Agent, tools: list[Callable[..., Any]]) -> None:
    for tool in tools:
        agent.tool_plain(name=tool.__name__)(tool)


def get_math_agent(
    system_prompt: str | None = None,
    model: str = DEFAULT_MATH_MODEL,
    extra_tools: list[Callable[..., Any]] | None = None,
) -> Agent:
    """Create the math agent with optional custom prompt and model."""
    prompt = system_prompt if system_prompt is not None else DEFAULT_MATH_AGENT_PROMPT

    tools: list[Callable[..., Any]] = [add, subtract, multiply, divide, convert_units]
    if extra_tools:
        tools.extend(extra_tools)

    agent = Agent(
        name="MathAgent",
        model=resolve_model_name(model),
        system_prompt=prompt,
    )
    _register_tools(agent, tools)
    return agent
