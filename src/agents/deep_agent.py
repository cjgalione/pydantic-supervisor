"""PydanticAI supervisor with delegation tools for specialized subtasks."""

from __future__ import annotations

import json
import math
import re
import ast
from typing import Any

from braintrust import SpanTypeAttribute, start_span
from pydantic_ai import Agent

from src.agents.critic_agent import get_critic_agent
from src.agents.math_agent import add, divide, get_math_agent, multiply, subtract
from src.agents.research_agent import get_research_agent
from src.config import AgentConfig
from src.helpers import run_pydantic_agent
from src.modeling import resolve_model_name


_MATH_OPS = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}

_CRITIC_ACTIONS = {"accept", "delegate_research", "delegate_math", "retry_with_instruction"}


def _register_tools(agent: Agent, tools: list[Any]) -> None:
    for tool in tools:
        agent.tool_plain(name=tool.__name__)(tool)


def _parse_number_token(token: str) -> float | None:
    cleaned = token.strip().lower().replace(",", "")
    if "^" in cleaned and "e" not in cleaned:
        base, exp = cleaned.split("^", 1)
        try:
            return float(base) ** float(exp)
        except ValueError:
            return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_conversion_operation(operation: str) -> tuple[float, str, str] | None:
    text = operation.strip()
    m = re.match(r"(?i)^convert\s+([^\s]+)\s+(.+?)\s+to\s+(.+)$", text)
    if not m:
        return None
    value_token, from_unit, to_unit = m.groups()
    value = _parse_number_token(value_token)
    if value is None:
        return None
    return value, from_unit.strip(), to_unit.strip()


def _is_basic_math_operation(operation: str) -> bool:
    return operation.strip().lower() in _MATH_OPS


def _classify_math_operation(operation: str) -> str:
    if _is_basic_math_operation(operation):
        return "arithmetic"
    if _parse_conversion_operation(operation) is not None:
        return "unit_conversion"
    return "other"


def _build_math_query(
    operation: str,
    a: float | None = None,
    b: float | None = None,
    result_mode: str = "numeric",
) -> str:
    if _is_basic_math_operation(operation):
        if a is None or b is None:
            raise ValueError("Basic arithmetic operations require both a and b operands.")
        if result_mode == "numeric":
            return (
                f"Use operation '{operation}' on the values a={a} and b={b}. "
                "Return the final numeric result."
            )
        return (
            f"Use operation '{operation}' on the values a={a} and b={b}. "
            "Explain the steps and include the final answer."
        )

    conversion = _parse_conversion_operation(operation)
    if conversion is not None:
        value, from_unit, to_unit = conversion
        if result_mode == "numeric":
            return (
                "Use unit conversion for this task and return only the final numeric result. "
                f"Convert value={value} from_unit='{from_unit}' to_unit='{to_unit}'."
            )
        return (
            "Use unit conversion for this task and provide a concise explanation with the final answer. "
            f"Convert value={value} from_unit='{from_unit}' to_unit='{to_unit}'."
        )

    if result_mode == "numeric":
        context = f"Context values (if useful): a={a}, b={b}." if a is not None and b is not None else ""
        return (
            "Solve the following quantitative task and return the final numeric result. "
            f"Task: {operation}. "
            f"{context}"
        )

    context = f"Context values (if useful): a={a}, b={b}." if a is not None and b is not None else ""
    return (
        "Solve the following math task and provide a concise explanation with the final answer. "
        f"Task: {operation}. "
        f"{context}"
    )


def _run_math(operation: str, a: float, b: float) -> float:
    op = operation.strip().lower()
    if op not in _MATH_OPS:
        raise ValueError(f"Unsupported math operation: {operation}")
    return float(_MATH_OPS[op](a, b))


def _extract_float_from_text(text: str) -> float | None:
    sci_caret_matches = re.findall(r"(-?\d+(?:\.\d+)?)\s*[x×]\s*10\^(-?\d+)", text, flags=re.IGNORECASE)
    if sci_caret_matches:
        base_s, exp_s = sci_caret_matches[-1]
        try:
            return float(base_s) * (10 ** int(exp_s))
        except ValueError:
            pass

    matches = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _fallback_numeric_from_operation_text(operation: str) -> float | None:
    lowered = operation.lower()

    sphere = re.search(r"volume of a sphere.*radius\s*([0-9]+(?:\.[0-9]+)?)", lowered)
    if sphere:
        r = float(sphere.group(1))
        return (4.0 / 3.0) * math.pi * (r**3)

    circle = re.search(r"area of a circle.*radius\s*([0-9]+(?:\.[0-9]+)?)", lowered)
    if circle:
        r = float(circle.group(1))
        return math.pi * (r**2)

    return None


def _safe_eval_numeric_expression(expression: str) -> float | None:
    expr = expression.strip().replace("^", "**")
    if not expr:
        return None

    allowed_names: dict[str, float] = {
        "pi": math.pi,
        "e": math.e,
    }
    allowed_bin_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    allowed_unary_ops = (ast.UAdd, ast.USub)

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in allowed_names:
                return float(allowed_names[node.id])
            raise ValueError("Unsupported name")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, allowed_unary_ops):
            value = _eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, allowed_bin_ops):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            if isinstance(node.op, ast.Mod):
                return left % right
        raise ValueError("Unsupported expression")

    try:
        parsed = ast.parse(expr, mode="eval")
        return float(_eval(parsed))
    except Exception:
        return None


def _handoff_span_metadata(*, target: str, input_data: dict[str, object]) -> dict[str, object]:
    return {
        "target_agent": target,
        "handoff_input": input_data,
    }


def _has_marker(messages: list[dict[str, Any]], markers: tuple[str, ...]) -> bool:
    lowered_markers = tuple(m.lower() for m in markers)
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = str(message.get("content", "") or "").lower()
        if any(marker in content for marker in lowered_markers):
            return True
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            tool_name = str(tc.get("name", "") or "").lower()
            if any(marker in tool_name for marker in lowered_markers):
                return True
    return False


def _query_needs_math_handoff(query: str) -> bool:
    q = query.lower()
    if re.search(r"\d", q) and any(
        token in q
        for token in (
            "calculate",
            "sum",
            "add",
            "subtract",
            "multiply",
            "divide",
            "minus",
            "plus",
            "product",
            "difference",
            "percent",
            "square root",
            "area",
            "solve",
            "equation",
        )
    ):
        return True
    return bool(re.search(r"\b\d+\s*[\+\-\*/]\s*\d+\b", q))


def _query_needs_research_handoff(query: str) -> bool:
    q = query.lower()
    keyword_match = any(
        token in q
        for token in (
            "latest",
            "current",
            "who is",
            "what is the capital",
            "population",
            "president",
            "ceo",
            "mayor",
            "won",
            "sources",
            "source",
            "according to",
        )
    )
    has_math_like = _query_needs_math_handoff(query)
    wh_match = bool(re.search(r"\b(who|what|when|where)\b", q)) and not has_math_like
    return keyword_match or wh_match


def _fallback_critic_decision(query: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    needs_math = _query_needs_math_handoff(query)
    needs_research = _query_needs_research_handoff(query)

    has_math_handoff = _has_marker(
        messages,
        ("delegate_to_math_agent", "request_math_subtask", "handoff [mathagent]"),
    )
    has_research_handoff = _has_marker(
        messages,
        ("delegate_to_research_agent", "request_research_subtask", "handoff [researchagent]"),
    )
    has_web_search = _has_marker(messages, ("tavily_search", "http://", "https://", "url:"))

    if needs_math and not has_math_handoff:
        return {
            "compliant": False,
            "required_action": "delegate_math",
            "rationale": "Math-style query requires MathAgent delegation evidence.",
        }
    if needs_research and (not has_research_handoff or not has_web_search):
        return {
            "compliant": False,
            "required_action": "delegate_research",
            "rationale": "Research-style query requires ResearchAgent handoff and web-search evidence.",
        }
    if (needs_math or needs_research) and not (has_math_handoff or has_research_handoff):
        return {
            "compliant": False,
            "required_action": "retry_with_instruction",
            "rationale": "Policy-triggering query was answered directly without delegation.",
        }
    return {
        "compliant": True,
        "required_action": "accept",
        "rationale": "Delegation/tool-use policy appears satisfied.",
    }


def _messages_summary(messages: list[dict[str, Any]], limit: int = 14) -> list[dict[str, Any]]:
    if len(messages) <= limit:
        return messages
    return messages[-limit:]


def _normalize_critic_decision(raw: dict[str, Any], query: str, messages: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = _fallback_critic_decision(query, messages)

    compliant_raw = raw.get("compliant")
    compliant_is_valid = isinstance(compliant_raw, bool)
    if not compliant_is_valid:
        return fallback

    compliant = bool(compliant_raw)
    required_action = str(raw.get("required_action", "") or "").strip()
    rationale = str(raw.get("rationale", "") or "").strip()

    if required_action not in _CRITIC_ACTIONS:
        return fallback
    if compliant and required_action != "accept":
        required_action = "accept"
    if (not compliant) and required_action == "accept":
        required_action = "retry_with_instruction"
    if not rationale:
        rationale = "No rationale provided by critic."

    normalized = {
        "compliant": compliant,
        "required_action": required_action,
        "rationale": rationale,
    }

    # Enforce deterministic policy if critic output disagrees.
    if (
        fallback["compliant"] != normalized["compliant"]
        or fallback["required_action"] != normalized["required_action"]
    ):
        return fallback

    if normalized["rationale"] == "No rationale provided by critic.":
        normalized["rationale"] = str(fallback.get("rationale", normalized["rationale"]))
    return normalized


def get_deep_agent(config: AgentConfig | None = None) -> Agent:
    """Create the supervisor agent and wire delegation tools."""
    resolved_config = config or AgentConfig()
    supervisor_prompt = resolved_config.render_supervisor_prompt()

    research_agent: Agent | None = None
    math_agent: Agent | None = None
    critic_agent: Agent | None = None

    async def _run_research_handoff(
        *,
        query: str,
        mode: str,
        app_name: str,
    ) -> dict[str, Any]:
        if research_agent is None:
            raise RuntimeError("ResearchAgent is not initialized")
        with start_span(
            name="handoff [ResearchAgent]",
            type=SpanTypeAttribute.TASK,
            input={"query": query},
            metadata=_handoff_span_metadata(
                target="ResearchAgent",
                input_data={"query": query, "mode": mode},
            ),
        ) as handoff_span:
            result = await run_pydantic_agent(
                agent=research_agent,
                query=query,
                app_name=app_name,
            )
            final_output = str(result.get("final_output", "")).strip()
            messages = result.get("messages", [])
            handoff_span.log(output={"final_output": final_output, "messages": messages})
            return {"final_output": final_output, "messages": messages}

    async def _run_math_handoff(
        *,
        math_task: str | None = None,
        operation: str | None = None,
        a: float | None = None,
        b: float | None = None,
        result_mode: str = "explanatory",
        mode: str = "delegate",
        app_name: str = "pydantic-supervisor-delegate-math",
    ) -> dict[str, Any]:
        if math_agent is None:
            raise RuntimeError("MathAgent is not initialized")
        if result_mode not in {"numeric", "explanatory"}:
            raise ValueError("result_mode must be either 'numeric' or 'explanatory'")

        resolved_operation = (math_task or operation or "").strip()
        if not resolved_operation:
            if mode == "delegate":
                return {
                    "final_output": "",
                    "parsed_result": None,
                    "returned_response": "Math task is missing required parameters.",
                    "messages": [],
                }
            raise ValueError("Provide a non-empty math_task (or operation alias).")

        if _is_basic_math_operation(resolved_operation) and (a is None or b is None):
            if mode == "delegate":
                return {
                    "final_output": "",
                    "parsed_result": None,
                    "returned_response": "Math task is missing operands for the requested operation.",
                    "messages": [],
                }
            raise ValueError("Basic arithmetic operations require both a and b operands.")

        query = _build_math_query(operation=resolved_operation, a=a, b=b, result_mode=result_mode)
        input_payload: dict[str, object] = {"math_task": resolved_operation}
        metadata_input: dict[str, object] = {
            "math_task": resolved_operation,
            "operation_type": _classify_math_operation(resolved_operation),
            "result_mode": result_mode,
            "mode": mode,
        }
        if a is not None:
            input_payload["a"] = a
            metadata_input["a"] = a
        if b is not None:
            input_payload["b"] = b
            metadata_input["b"] = b

        with start_span(
            name="handoff [MathAgent]",
            type=SpanTypeAttribute.TASK,
            input=input_payload,
            metadata=_handoff_span_metadata(
                target="MathAgent",
                input_data=metadata_input,
            ),
        ) as handoff_span:
            result = await run_pydantic_agent(
                agent=math_agent,
                query=query,
                app_name=app_name,
            )
            final_text = str(result.get("final_output", "")).strip()
            messages = result.get("messages", [])
            parsed = _extract_float_from_text(final_text)
            parsed_result: float | None = parsed

            if result_mode == "numeric":
                if parsed_result is None and _is_basic_math_operation(resolved_operation):
                    if a is None or b is None:
                        raise ValueError("Basic arithmetic operations require both a and b operands.")
                    parsed_result = _run_math(operation=resolved_operation, a=a, b=b)
                if parsed_result is None:
                    # Retry once with stricter numeric-only instructions before failing.
                    numeric_retry_query = (
                        "Return only the final numeric value for this task with no words or units. "
                        f"Task: {resolved_operation}"
                    )
                    retry_result = await run_pydantic_agent(
                        agent=math_agent,
                        query=numeric_retry_query,
                        app_name=f"{app_name}-numeric-retry",
                    )
                    retry_final = str(retry_result.get("final_output", "")).strip()
                    retry_parsed = _extract_float_from_text(retry_final)
                    messages.extend(retry_result.get("messages", []))
                    if retry_parsed is not None:
                        parsed_result = retry_parsed
                        final_text = retry_final or final_text
                if parsed_result is None:
                    heuristic = _fallback_numeric_from_operation_text(resolved_operation)
                    if heuristic is not None:
                        parsed_result = heuristic
                if parsed_result is None:
                    expr_value = _safe_eval_numeric_expression(resolved_operation)
                    if expr_value is not None:
                        parsed_result = expr_value
                if parsed_result is None and mode == "subtask":
                    raise ValueError(
                        f"MathAgent did not return a numeric result for operation '{resolved_operation}'. "
                        f"Model output: {final_text}"
                    )
                if parsed_result is not None:
                    response_text = str(parsed_result)
                else:
                    # In delegate mode, avoid failing the entire run for advanced symbolic tasks.
                    response_text = final_text or "MathAgent returned no output."
            else:
                if final_text:
                    response_text = final_text
                elif parsed_result is not None:
                    response_text = str(parsed_result)
                elif _is_basic_math_operation(resolved_operation):
                    if a is None or b is None:
                        raise ValueError("Basic arithmetic operations require both a and b operands.")
                    response_text = str(_run_math(operation=resolved_operation, a=a, b=b))
                    parsed_result = _extract_float_from_text(response_text)
                else:
                    response_text = "MathAgent returned no output."

            if result_mode == "numeric" and parsed_result is None and mode == "subtask":
                raise ValueError(
                    f"MathAgent did not return a numeric result for operation '{resolved_operation}'. "
                    f"Model output: {final_text}"
                )
            handoff_span.log(
                output={
                    "final_output": final_text,
                    "parsed_result": parsed_result,
                    "result_mode": result_mode,
                    "returned_response": response_text,
                    "messages": messages,
                }
            )
            return {
                "final_output": final_text,
                "parsed_result": parsed_result,
                "returned_response": response_text,
                "messages": messages,
            }

    async def request_research_subtask(query: str, max_results: int = 3) -> str:
        """Request research before completing a downstream math subtask."""
        del max_results
        result = await _run_research_handoff(
            query=query,
            mode="subtask",
            app_name="pydantic-supervisor-research-subtask",
        )
        return str(result.get("final_output", ""))

    async def request_math_subtask(operation: str, a: float, b: float) -> float:
        """Request a math subtask during compound research + calculation workflows."""
        result = await _run_math_handoff(
            operation=operation,
            a=a,
            b=b,
            result_mode="numeric",
            mode="subtask",
            app_name="pydantic-supervisor-math-subtask",
        )
        parsed_result = result.get("parsed_result")
        if parsed_result is None:
            raise ValueError(f"MathAgent did not return a numeric result for operation '{operation}'.")
        return float(parsed_result)

    research_agent = get_research_agent(
        system_prompt=resolved_config.research_agent_prompt,
        model=resolved_config.research_model,
        extra_tools=[request_math_subtask],
    )
    math_agent = get_math_agent(
        system_prompt=resolved_config.math_agent_prompt,
        model=resolved_config.math_model,
        extra_tools=[request_research_subtask],
    )
    critic_agent = get_critic_agent(model=resolved_config.supervisor_model)

    async def delegate_to_research_agent(query: str, max_results: int = 3) -> str:
        """Delegate a factual lookup or web-research task to ResearchAgent."""
        del max_results
        result = await _run_research_handoff(
            query=query,
            mode="delegate",
            app_name="pydantic-supervisor-delegate-research",
        )
        return str(result.get("final_output", ""))

    async def delegate_to_math_agent(
        math_task: str | None = None,
        operation: str | None = None,
        a: float | None = None,
        b: float | None = None,
        result_mode: str = "explanatory",
    ) -> str:
        """Delegate a math task to MathAgent.

        `result_mode`:
        - "numeric": enforce numeric output
        - "explanatory": allow textual or symbolic responses

        IMPORTANT:
        - For symbolic math, pass the full expression/question in `math_task`
          (e.g., "derivative of x^2"), not only an operator word.
        - `operation` is retained as a backward-compatible alias.
        """
        result = await _run_math_handoff(
            math_task=math_task,
            operation=operation,
            a=a,
            b=b,
            result_mode=result_mode,
            mode="delegate",
            app_name="pydantic-supervisor-delegate-math",
        )
        return str(result.get("returned_response", ""))

    async def _run_critic_decision(
        *,
        query: str,
        candidate_final_output: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if critic_agent is None:
            return _fallback_critic_decision(query, messages)

        messages_summary = _messages_summary(messages)
        critic_input = {
            "query": query,
            "candidate_final_output": candidate_final_output,
            "messages": messages,
        }
        critic_query = (
            "Evaluate this candidate against delegation policy and return JSON only.\n"
            f"{json.dumps(critic_input, ensure_ascii=False)}"
        )
        with start_span(
            name="critic [CriticAgent]",
            type=SpanTypeAttribute.TASK,
            input={
                "query": query,
                "candidate_final_output": candidate_final_output,
                "messages_summary": messages_summary,
            },
        ) as critic_span:
            critic_result = await run_pydantic_agent(
                agent=critic_agent,
                query=critic_query,
                app_name="pydantic-supervisor-critic",
            )
            raw_text = str(critic_result.get("final_output", "")).strip()
            raw_decision: dict[str, Any] = {}
            try:
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    raw_decision = parsed
            except Exception:
                raw_decision = {}

            if not raw_decision:
                for message in reversed(critic_result.get("messages", [])):
                    if not isinstance(message, dict) or message.get("role") != "assistant":
                        continue
                    content = str(message.get("content", "")).strip()
                    if not content:
                        continue
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict):
                            raw_decision = parsed
                            break
                    except Exception:
                        continue

            decision = _normalize_critic_decision(raw_decision, query, messages)
            critic_span.log(output={"decision": decision})
            return decision

    async def validate_and_correct(
        query: str,
        candidate_final_output: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        decision = await _run_critic_decision(
            query=query,
            candidate_final_output=candidate_final_output,
            messages=messages,
        )
        enriched_messages: list[dict[str, Any]] = list(messages)
        enriched_messages.append(
            {
                "role": "system",
                "content": f"critic_decision: {json.dumps(decision, ensure_ascii=False)}",
                "critic_decision": decision,
            }
        )

        corrected = False
        corrected_output = candidate_final_output

        if not decision.get("compliant", False):
            corrected = True
            action = str(decision.get("required_action", "retry_with_instruction"))

            if action == "delegate_research":
                handoff = await _run_research_handoff(
                    query=query,
                    mode="critic_correction",
                    app_name="pydantic-supervisor-critic-delegate-research",
                )
                corrected_output = str(handoff.get("final_output", "")).strip()
                enriched_messages.extend(handoff.get("messages", []))
                enriched_messages.append(
                    {
                        "role": "system",
                        "content": "handoff marker: handoff [ResearchAgent]",
                    }
                )
            elif action == "delegate_math":
                handoff = await _run_math_handoff(
                    math_task=query,
                    result_mode="explanatory",
                    mode="critic_correction",
                    app_name="pydantic-supervisor-critic-delegate-math",
                )
                corrected_output = str(handoff.get("returned_response", "")).strip()
                enriched_messages.extend(handoff.get("messages", []))
                enriched_messages.append(
                    {
                        "role": "system",
                        "content": "handoff marker: handoff [MathAgent]",
                    }
                )
            else:
                strict_query = (
                    "POLICY ENFORCEMENT: You MUST delegate to the correct specialist agent(s) for this query "
                    "and must not answer directly when delegation rules apply.\n"
                    f"Original query: {query}"
                )
                rerun = await run_pydantic_agent(
                    agent=supervisor_agent,
                    query=strict_query,
                    app_name="pydantic-supervisor-critic-retry",
                )
                corrected_output = str(rerun.get("final_output", "")).strip()
                enriched_messages.extend(rerun.get("messages", []))

            decision = await _run_critic_decision(
                query=query,
                candidate_final_output=corrected_output,
                messages=enriched_messages,
            )
            enriched_messages.append(
                {
                    "role": "system",
                    "content": f"critic_decision_retry: {json.dumps(decision, ensure_ascii=False)}",
                    "critic_decision": decision,
                }
            )

        if corrected_output and not any(
            isinstance(m, dict)
            and m.get("role") == "assistant"
            and str(m.get("content", "")).strip() == corrected_output
            for m in enriched_messages
        ):
            enriched_messages.append({"role": "assistant", "content": corrected_output})

        return {
            "final_output": corrected_output,
            "messages": enriched_messages,
            "critic_decision": decision,
            "corrected": corrected,
        }

    supervisor_agent = Agent(
        name="SupervisorAgent",
        model=resolve_model_name(resolved_config.supervisor_model),
        system_prompt=supervisor_prompt,
    )
    _register_tools(
        supervisor_agent,
        [
            delegate_to_research_agent,
            request_research_subtask,
            delegate_to_math_agent,
            request_math_subtask,
        ],
    )
    setattr(supervisor_agent, "_validate_and_correct", validate_and_correct)
    return supervisor_agent


_cached_deep_agent: Agent | None = None


def get_supervisor(config: AgentConfig | None = None, force_rebuild: bool = False) -> Agent:
    """Get a cached or newly built supervisor agent."""
    global _cached_deep_agent

    if config is not None:
        return get_deep_agent(config)

    if force_rebuild or _cached_deep_agent is None:
        _cached_deep_agent = get_deep_agent()
    return _cached_deep_agent


async def run_supervisor_with_critic(
    *,
    supervisor: Agent,
    query: str,
    app_name: str,
) -> dict[str, Any]:
    """Run supervisor, then enforce delegation policy with critic validation."""
    with start_span(
        name="invocation [supervisor_with_critic]",
        type=SpanTypeAttribute.TASK,
        input={"query": query, "app_name": app_name},
    ) as root_span:
        candidate = await run_pydantic_agent(
            agent=supervisor,
            query=query,
            app_name=app_name,
        )
        candidate_output = str(candidate.get("final_output", "")).strip()
        candidate_messages = candidate.get("messages", [])

        validator = getattr(supervisor, "_validate_and_correct", None)
        if callable(validator):
            validated = await validator(query, candidate_output, candidate_messages)
        else:
            fallback_decision = _fallback_critic_decision(query, candidate_messages)
            validated = {
                "final_output": candidate_output,
                "messages": candidate_messages,
                "critic_decision": fallback_decision,
                "corrected": False,
            }

        root_span.log(
            output={
                "final_output": validated.get("final_output", ""),
                "messages": validated.get("messages", []),
                "critic_decision": validated.get("critic_decision", {}),
                "critic_corrected": bool(validated.get("corrected", False)),
            }
        )
        return {
            "final_output": str(validated.get("final_output", "")).strip(),
            "messages": validated.get("messages", []),
            "critic_decision": validated.get("critic_decision", {}),
            "critic_corrected": bool(validated.get("corrected", False)),
        }
