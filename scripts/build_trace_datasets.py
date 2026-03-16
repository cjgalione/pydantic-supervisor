"""Build curated agent datasets from exported Braintrust trace JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT_EXPORT = Path("/tmp/pydantic_supervisor_roots.json")
CANDIDATE_EXPORT = Path("/tmp/pydantic_supervisor_candidate_spans.json")

DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"

SUPERVISOR_QUERIES = [
    "What is 12*9?",
    "What is the derivative of sin(x^2)?",
    "Briefly explain the history of the Renaissance.",
    "Research the pros and cons of renewable energy.",
    "What are the main functions of the United Nations?",
    "Integrate cos(x).",
    "How is electricity generated?",
    "What is the difference between a virus and a bacteria?",
    "Solve this: 4 + 2 * (8 / 2) - 1.",
    "Explain 'confirmation bias'.",
    "How does a solar panel work?",
    "Find the average of the numbers 10, 20, and 30.",
    "What is the current global population?",
    "Write me a simple HTML page.",
    "I don't understand this math problem! Explain it again, please!",
]

RESEARCH_QUERIES = [
    "Briefly explain the history of the Renaissance.",
    "Research the pros and cons of renewable energy.",
    "What are the main functions of the United Nations?",
    "What is the difference between a virus and a bacteria?",
    "How is electricity generated?",
    "What is the formula for the area of a triangle?",
    "What is the history of the internet?",
    "Explain 'confirmation bias'.",
    "How does a solar panel work?",
    "What is the binomial theorem?",
    "What are the main types of rocks?",
    "What is a black hole?",
    "What is the speed of light?",
    "What is the current global population?",
    "How do vaccines work?",
]

MATH_QUERIES = [
    "What is 12*9?",
    "What is the derivative of sin(x^2)?",
    "Integrate cos(x).",
    "Solve this: 4 + 2 * (8 / 2) - 1.",
    "Find the average of the numbers 10, 20, and 30.",
    "Find the eigenvalues of the matrix [[1, 2], [3, 4]].",
    "Solve the system of equations: x + y = 5, x - y = 1.",
    "Calculate the area of a circle with radius 7.",
    "Solve the quadratic equation: x^2 + 5x + 6 = 0.",
    "Calculate the surface area of a cube with side length 5.",
    "Solve: 10 + 5 - 2 * 3",
    "Calculate 5! (5 factorial).",
    "Calculate the volume of a sphere with radius 3.",
    "Solve for x: 2(x+3) - 7 = 9.",
    "Differentiate y = x^3 - 2x + 1.",
]

MATH_REFERENCE_ANSWERS = {
    "What is 12*9?": "108",
    "What is the derivative of sin(x^2)?": "2x cos(x^2)",
    "Integrate cos(x).": "sin(x) + C",
    "Solve this: 4 + 2 * (8 / 2) - 1.": "11",
    "Find the average of the numbers 10, 20, and 30.": "20",
    "Find the eigenvalues of the matrix [[1, 2], [3, 4]].": "(5 ± sqrt(33)) / 2",
    "Solve the system of equations: x + y = 5, x - y = 1.": "x = 3, y = 2",
    "Calculate the area of a circle with radius 7.": "49*pi",
    "Solve the quadratic equation: x^2 + 5x + 6 = 0.": "x = -2 and x = -3",
    "Calculate the surface area of a cube with side length 5.": "150",
    "Solve: 10 + 5 - 2 * 3": "9",
    "Calculate 5! (5 factorial).": "120",
    "Calculate the volume of a sphere with radius 3.": "36*pi",
    "Solve for x: 2(x+3) - 7 = 9.": "x = 5",
    "Differentiate y = x^3 - 2x + 1.": "3x^2 - 2",
}


def _load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return list(payload["data"])


def _has_url(messages: list[dict[str, Any]]) -> bool:
    return any(
        "http://" in str(message.get("content", "")) or "https://" in str(message.get("content", ""))
        for message in messages
        if isinstance(message, dict)
    )


def _tool_route(messages: list[dict[str, Any]]) -> str:
    tool_names: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        for tool_call in message.get("tool_calls", []):
            if isinstance(tool_call, dict):
                tool_names.add(str(tool_call.get("name", "")))

    uses_research = any("research" in name for name in tool_names)
    uses_math = any("math" in name for name in tool_names)
    if uses_research and uses_math:
        return "mixed"
    if uses_research:
        return "research"
    if uses_math:
        return "math"
    return "direct"


def _require_compliant(root_row: dict[str, Any]) -> None:
    decision = ((root_row.get("output") or {}).get("critic_decision") or {})
    if decision.get("compliant") is not True:
        query = (root_row.get("input") or {}).get("query", "<unknown>")
        raise ValueError(f"Expected compliant supervisor trace for {query!r}")


def _pick_best_research_span(rows: list[dict[str, Any]]) -> dict[str, Any]:
    with_url = [row for row in rows if _has_url((row.get("output") or {}).get("messages", []))]
    return with_url[0] if with_url else rows[0]


def _pick_best_math_span(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def _score(row: dict[str, Any]) -> tuple[int, int]:
        output = row.get("output") or {}
        response = str(output.get("returned_response") or output.get("final_output") or "")
        return (1 if "Final Answer" in response else 0, len(response))

    return sorted(rows, key=_score, reverse=True)[0]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))


def main() -> None:
    if not ROOT_EXPORT.exists() or not CANDIDATE_EXPORT.exists():
        missing = [str(path) for path in (ROOT_EXPORT, CANDIDATE_EXPORT) if not path.exists()]
        raise FileNotFoundError(f"Missing exported Braintrust data: {', '.join(missing)}")

    root_rows = _load_rows(ROOT_EXPORT)
    candidate_rows = _load_rows(CANDIDATE_EXPORT)

    roots_by_query = {row["input"]["query"]: row for row in root_rows if isinstance(row.get("input"), dict)}
    research_by_root: dict[str, list[dict[str, Any]]] = {}
    math_by_root: dict[str, list[dict[str, Any]]] = {}

    for row in candidate_rows:
        name = ((row.get("span_attributes") or {}).get("name")) or ""
        root_span_id = row.get("root_span_id")
        if name == "handoff [ResearchAgent]":
            research_by_root.setdefault(root_span_id, []).append(row)
        elif name == "handoff [MathAgent]":
            math_by_root.setdefault(root_span_id, []).append(row)

    supervisor_rows: list[dict[str, Any]] = []
    for query in SUPERVISOR_QUERIES:
        root = roots_by_query[query]
        _require_compliant(root)
        output = root["output"]
        supervisor_rows.append(
            {
                "input": {"query": query},
                "expected": {
                    "final_output": output["final_output"],
                    "route": _tool_route(output.get("messages", [])),
                    "critic_compliant": True,
                },
                "metadata": {
                    "source_root_span_id": root["root_span_id"],
                    "source_span_id": root["id"],
                    "source_span_name": "invocation [supervisor_with_critic]",
                    "created": root["created"],
                },
            }
        )

    research_rows: list[dict[str, Any]] = []
    for query in RESEARCH_QUERIES:
        root = roots_by_query[query]
        _require_compliant(root)
        span = _pick_best_research_span(research_by_root[root["root_span_id"]])
        output = span["output"]
        research_rows.append(
            {
                "input": {"query": query},
                "expected": {
                    "final_output": output["final_output"],
                    "should_use_search": True,
                    "should_include_url": _has_url(output.get("messages", [])),
                },
                "metadata": {
                    "source_root_span_id": root["root_span_id"],
                    "source_span_id": span["id"],
                    "source_span_name": "handoff [ResearchAgent]",
                    "created": span["created"],
                },
            }
        )

    math_rows: list[dict[str, Any]] = []
    for query in MATH_QUERIES:
        root = roots_by_query[query]
        _require_compliant(root)
        span = _pick_best_math_span(math_by_root[root["root_span_id"]])
        output = span["output"]
        math_rows.append(
            {
                "input": {"query": query},
                "expected": {
                    "final_output": output.get("returned_response") or output.get("final_output", ""),
                    "expected_answer": MATH_REFERENCE_ANSWERS[query],
                },
                "metadata": {
                    "source_root_span_id": root["root_span_id"],
                    "source_span_id": span["id"],
                    "source_span_name": "handoff [MathAgent]",
                    "created": span["created"],
                },
            }
        )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    _write_jsonl(DATASET_DIR / "supervisor_trace_dataset.jsonl", supervisor_rows)
    _write_jsonl(DATASET_DIR / "research_trace_dataset.jsonl", research_rows)
    _write_jsonl(DATASET_DIR / "math_trace_dataset.jsonl", math_rows)

    print("wrote", DATASET_DIR / "supervisor_trace_dataset.jsonl")
    print("wrote", DATASET_DIR / "research_trace_dataset.jsonl")
    print("wrote", DATASET_DIR / "math_trace_dataset.jsonl")


if __name__ == "__main__":
    main()
