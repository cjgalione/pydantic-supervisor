#!/usr/bin/env python3
"""Run a single supervisor query with configurable routing/eval tracing options."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Add project root to path so local imports work when run from scripts/.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_graph import get_supervisor, run_supervisor_with_critic
from src.config import AgentConfig
from src.tracing import configure_adk_tracing

DEFAULT_PROJECT = "pydantic-supervisor"
DEFAULT_SUPERVISOR_MODEL = "gemini-2.0-flash-lite"


def _coerce_value(value: str) -> Any:
    """Best-effort parse for metadata values (JSON primitives/objects)."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_metadata(
    metadata_json: str | None,
    metadata_kv: list[str] | None,
) -> dict[str, Any]:
    """Parse metadata from JSON blob + repeated key=value pairs."""
    metadata: dict[str, Any] = {}

    if metadata_json:
        parsed = json.loads(metadata_json)
        if not isinstance(parsed, dict):
            raise ValueError("--trace-metadata-json must be a JSON object")
        metadata.update(parsed)

    for pair in metadata_kv or []:
        if "=" not in pair:
            raise ValueError(f"Invalid --trace-metadata entry (expected key=value): {pair}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --trace-metadata entry (empty key): {pair}")
        metadata[key] = _coerce_value(raw_value)

    return metadata


async def _run(args: argparse.Namespace) -> None:
    query = args.query

    supervisor_model = args.supervisor_model or DEFAULT_SUPERVISOR_MODEL
    research_model = args.research_model or supervisor_model
    math_model = args.math_model or supervisor_model

    metadata = _parse_metadata(args.trace_metadata_json, args.trace_metadata)
    metadata.setdefault("selected_model", supervisor_model)

    if not args.no_braintrust:
        api_key = os.environ.get("BRAINTRUST_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "BRAINTRUST_API_KEY is missing. Set it in environment/.env or use --no-braintrust."
            )
        configure_adk_tracing(
            api_key=api_key,
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=args.project,
        )

    config = AgentConfig(
        supervisor_model=supervisor_model,
        research_model=research_model,
        math_model=math_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    result = await run_supervisor_with_critic(
        supervisor=supervisor,
        query=query,
        app_name=args.workflow_name,
    )

    messages = result["messages"]
    print(f"FINAL: {result.get('final_output', '')}")
    print("MESSAGES:")
    print(json.dumps(messages, indent=2, ensure_ascii=False))
    if metadata:
        print("TRACE METADATA:")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run one retest query through the supervisor.")
    parser.add_argument(
        "--query",
        required=True,
        help="User query to run through the supervisor.",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_PROJECT),
        help="Braintrust project name.",
    )
    parser.add_argument(
        "--supervisor-model",
        default=None,
        help="Override supervisor model.",
    )
    parser.add_argument(
        "--research-model",
        default=None,
        help="Override research model.",
    )
    parser.add_argument(
        "--math-model",
        default=None,
        help="Override math model.",
    )
    parser.add_argument(
        "--workflow-name",
        default="pydantic-supervisor-retest",
        help="Logical app/workflow name for this run.",
    )
    parser.add_argument(
        "--trace-metadata-json",
        default=None,
        help="Optional JSON object merged into trace metadata.",
    )
    parser.add_argument(
        "--trace-metadata",
        action="append",
        default=[],
        help="Repeatable key=value metadata entries.",
    )
    parser.add_argument(
        "--no-braintrust",
        action="store_true",
        help="Disable Braintrust tracing setup.",
    )

    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
