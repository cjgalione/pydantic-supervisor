#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_graph import run_supervisor_with_critic
from src.config import AgentConfig
from src.modeling import ensure_google_api_keys, get_google_api_key
from src.tempo_stress_tracing import emit_supervisor_trace, force_flush, get_export_stats
from src.tracing import configure_adk_tracing, get_trace_backend

load_dotenv()
ensure_google_api_keys()

DEFAULT_MODEL_POOL = ["gemini-2.0-flash-lite"]
QUESTION_GENERATOR_MODEL = "gemini-2.0-flash-lite"

QUESTION_BANK = [
    "What is 37 * 24?",
    "Who won the first modern Olympic Games and in what year?",
    "If a supernova releases 10^44 joules, how many 60W lightbulb-hours is that?",
    "What's the capital of Japan and what is 18% of 250?",
    "Hey, can you help me quickly estimate 15% tip on $86.40?",
    "When was the Eiffel Tower completed?",
    "Compute (1250 / 5) - 73.",
    "I'm frustrated. Just tell me if 144 divided by 12 is actually 11 or 12.",
    "What is the population of Canada and what is 2% of that number?",
    "Convert 10^6 joules to horsepower-seconds.",
    "What is the square root of 2025?",
    "Can you summarize what a quasar is in one sentence?",
    "If GDP is $2.1T and growth is 3.2%, what is the increase?",
    "Who discovered penicillin and in what year?",
    "What is (48 + 72) / 6?",
]


def _extract_json_array(text: str) -> list[str]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
            if text.startswith("json"):
                text = text[4:].strip()

    parsed = json.loads(text)
    if not isinstance(parsed, list) or not all(isinstance(q, str) for q in parsed):
        raise RuntimeError("Question generator did not return a JSON array of strings")
    return parsed


def _fallback_questions(num_questions: int, rng: random.Random) -> list[str]:
    questions = QUESTION_BANK.copy()
    rng.shuffle(questions)
    if num_questions <= len(questions):
        return questions[:num_questions]
    out: list[str] = []
    while len(out) < num_questions:
        remaining = num_questions - len(out)
        out.extend(questions[:remaining])
        rng.shuffle(questions)
    return out


def _is_resource_exhausted_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "resource_exhausted" in text or "quota exceeded" in text or "error code 429" in text


def _is_hard_quota_exhausted(exc: Exception) -> bool:
    text = str(exc).lower()
    return "generaterequestsperday" in text or "limit: 0" in text


def _retry_delay_seconds(exc: Exception) -> float | None:
    text = str(exc)

    m = re.search(r"Please retry in ([0-9]+(?:\.[0-9]+)?)s", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"'retryDelay': '([0-9]+)s'", text)
    if m:
        return float(m.group(1))

    return None


def _parse_model_pool(raw_model_pool: str | None) -> list[str]:
    if not raw_model_pool:
        return DEFAULT_MODEL_POOL.copy()

    models = [candidate.strip() for candidate in raw_model_pool.split(",")]
    models = [model for model in models if model]
    if not models:
        return DEFAULT_MODEL_POOL.copy()
    return models


def _normalize_payload_profile(payload_profile: str | None) -> str:
    profile = (payload_profile or os.environ.get("TRACE_PAYLOAD_PROFILE", "baseline")).strip().lower()
    return profile if profile in {"baseline", "large", "xlarge"} else "baseline"


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions with Gemini."""
    rng = random.Random(seed)
    api_key = get_google_api_key()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY/GOOGLE_API_KEY in environment")

    client = genai.Client(api_key=api_key)
    prompt = f"""Generate exactly {num_questions} realistic user questions that test an AI multi-agent system.

Create a diverse mix of:
- Pure math questions
- Pure research questions
- Hybrid questions (research + math)
- Edge cases (ambiguous, conversational, frustrated)

Output requirements:
- Return ONLY a valid JSON array of strings
- No markdown, no explanation
- Keep each question under 200 characters
"""
    try:
        response = client.models.generate_content(
            model=QUESTION_GENERATOR_MODEL,
            contents=prompt,
        )
        text = (response.text or "").strip()
        questions = _extract_json_array(text)
        rng.shuffle(questions)
        return questions[:num_questions]
    except Exception:
        return _fallback_questions(num_questions=num_questions, rng=rng)


def _quota_preflight_ok() -> tuple[bool, str]:
    api_key = get_google_api_key()
    if not api_key:
        return False, "Missing GEMINI_API_KEY/GOOGLE_API_KEY in environment"

    client = genai.Client(api_key=api_key)
    try:
        client.models.generate_content(
            model=QUESTION_GENERATOR_MODEL,
            contents="Reply with exactly: OK",
        )
        return True, ""
    except Exception as exc:
        if _is_hard_quota_exhausted(exc):
            return False, str(exc)
        return True, ""


def _synthetic_messages(question: str) -> list[dict[str, Any]]:
    """Deterministic fallback message shape when live model calls are disabled."""
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"name": "delegate_to_research_agent", "args": {"query": question}},
                {"name": "delegate_to_math_agent", "args": {"math_task": "synthetic"}},
            ],
        },
        {"role": "tool", "content": "synthetic tool result"},
        {"role": "assistant", "content": f"Synthetic response for: {question}"},
    ]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


async def run_question(
    question: str,
    *,
    model_pool: list[str],
    per_question_timeout_seconds: float,
    max_retries: int,
    base_retry_seconds: float,
    payload_profile: str,
    inject_large_attributes: bool,
    trace_run_tag: str,
    question_index: int,
    round_index: int,
    synthetic_only: bool,
    trace_target_bytes: int | None,
) -> tuple[bool, bool, float, dict[str, Any]]:
    """Run one question through the supervisor with a random model assignment."""
    attempt = 0
    started = time.perf_counter()
    trace_payload: dict[str, Any] = {}

    if synthetic_only:
        selected_model = "synthetic"
        synthetic_messages = _synthetic_messages(question)
        elapsed = time.perf_counter() - started
        if get_trace_backend() == "otlp":
            if trace_target_bytes and trace_target_bytes > 0:
                os.environ["TRACE_TARGET_BYTES"] = str(trace_target_bytes)
            else:
                os.environ.pop("TRACE_TARGET_BYTES", None)
            trace_payload = emit_supervisor_trace(
                query=question,
                final_output=f"Synthetic response for: {question}",
                messages=synthetic_messages,
                payload_profile=payload_profile,
                inject_large_attributes=inject_large_attributes,
                metadata={
                    "trace_run_tag": trace_run_tag,
                    "question_index": question_index,
                    "round_index": round_index,
                    "selected_model": "synthetic",
                    "attempt": 1,
                    "elapsed_seconds": round(elapsed, 6),
                },
            )
        print(f"✅ [synthetic] {question[:80]}")
        return True, False, elapsed, trace_payload

    from src.agent_graph import get_supervisor

    selected_model = random.choice(model_pool)
    config = AgentConfig(
        supervisor_model=selected_model,
        research_model=selected_model,
        math_model=selected_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    while True:
        attempt += 1
        try:
            result = await asyncio.wait_for(
                run_supervisor_with_critic(
                    supervisor=supervisor,
                    query=question,
                    app_name="pydantic-supervisor-batch",
                ),
                timeout=per_question_timeout_seconds,
            )
            elapsed = time.perf_counter() - started

            if get_trace_backend() == "otlp":
                if trace_target_bytes and trace_target_bytes > 0:
                    os.environ["TRACE_TARGET_BYTES"] = str(trace_target_bytes)
                else:
                    os.environ.pop("TRACE_TARGET_BYTES", None)
                trace_payload = emit_supervisor_trace(
                    query=question,
                    final_output=str(result.get("final_output", "")),
                    messages=result.get("messages", []),
                    payload_profile=payload_profile,
                    inject_large_attributes=inject_large_attributes,
                    metadata={
                        "trace_run_tag": trace_run_tag,
                        "question_index": question_index,
                        "round_index": round_index,
                        "selected_model": selected_model,
                        "attempt": attempt,
                        "elapsed_seconds": round(elapsed, 6),
                    },
                )

            print(f"✅ {question[:80]} -> {str(result.get('final_output', ''))[:80]}")
            return True, False, elapsed, trace_payload
        except TimeoutError:
            elapsed = time.perf_counter() - started
            print(f"❌ {question[:80]} -> timed out after {per_question_timeout_seconds:.1f}s")
            return False, False, elapsed, trace_payload
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                elapsed = time.perf_counter() - started
                print(f"❌ {question[:80]} -> {exc}")
                return False, False, elapsed, trace_payload

            if _is_hard_quota_exhausted(exc):
                elapsed = time.perf_counter() - started
                print(f"⏹️ {question[:80]} -> hard quota exhausted ({exc})")
                return False, True, elapsed, trace_payload

            if attempt > max_retries:
                elapsed = time.perf_counter() - started
                print(f"❌ {question[:80]} -> exhausted retries ({exc})")
                return False, False, elapsed, trace_payload

            suggested = _retry_delay_seconds(exc)
            backoff = base_retry_seconds * (2 ** (attempt - 1))
            sleep_s = max(suggested or 0.0, backoff)
            print(f"⏳ {question[:80]} -> retrying in {sleep_s:.1f}s after quota error")
            await asyncio.sleep(sleep_s)


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    if args.quota_preflight:
        ok, reason = _quota_preflight_ok()
        if not ok:
            print("Hard quota appears exhausted; skipping this batch run.")
            print(reason)
            return {
                "status": "skipped",
                "reason": reason,
            }

    effective_questions = args.questions if args.questions is not None else args.num_questions
    if effective_questions is None:
        effective_questions = random.randint(1, 100)

    rng = random.Random(args.seed)
    if args.question_source == "bank":
        questions = _fallback_questions(num_questions=effective_questions, rng=rng)
    else:
        questions = generate_questions(num_questions=effective_questions, seed=args.seed)

    model_pool = _parse_model_pool(args.model_pool)
    payload_profile = _normalize_payload_profile(args.payload_profile)
    trace_target_bytes = args.trace_target_bytes if args.trace_target_bytes and args.trace_target_bytes > 0 else None

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency} rounds={args.rounds}")
    print(f"Model pool: {', '.join(model_pool)}")
    print(f"Question source: {args.question_source}")
    print(f"Per-question timeout: {args.per_question_timeout_seconds:.1f}s")
    print(f"Trace backend: {get_trace_backend()} payload_profile={payload_profile}")
    print("=" * 80)

    successes = 0
    failures = 0
    hard_quota_stop = False
    latencies: list[float] = []
    emitted_traces: list[dict[str, Any]] = []

    run_started_at = datetime.now(timezone.utc)

    for round_index in range(args.rounds):
        if hard_quota_stop:
            break
        print(f"Round {round_index + 1}/{args.rounds}")

        for i in range(0, len(questions), args.concurrency):
            if hard_quota_stop:
                break
            batch = questions[i : i + args.concurrency]
            results = await asyncio.gather(
                *(
                    run_question(
                        q,
                        model_pool=model_pool,
                        per_question_timeout_seconds=args.per_question_timeout_seconds,
                        max_retries=args.max_retries,
                        base_retry_seconds=args.base_retry_seconds,
                        payload_profile=payload_profile,
                        inject_large_attributes=args.inject_large_attributes,
                        trace_run_tag=args.trace_run_tag,
                        question_index=i + idx,
                        round_index=round_index,
                        synthetic_only=args.synthetic_only,
                        trace_target_bytes=trace_target_bytes,
                    )
                    for idx, q in enumerate(batch)
                )
            )
            for ok, hard_stop, elapsed, trace_payload in results:
                latencies.append(elapsed)
                if ok:
                    successes += 1
                else:
                    failures += 1
                if trace_payload:
                    emitted_traces.append(trace_payload)
                if hard_stop:
                    hard_quota_stop = True
            if hard_quota_stop:
                print("Hard quota exhausted; stopping remaining questions to avoid repeated 429s.")
                break
            if args.inter_question_delay_seconds > 0:
                await asyncio.sleep(args.inter_question_delay_seconds)
            print()

    if get_trace_backend() == "otlp":
        force_flush()

    run_finished_at = datetime.now(timezone.utc)
    total_attempts = successes + failures

    latency_summary = {
        "p50_seconds": round(_percentile(latencies, 0.50), 6),
        "p95_seconds": round(_percentile(latencies, 0.95), 6),
        "max_seconds": round(max(latencies) if latencies else 0.0, 6),
        "mean_seconds": round(statistics.fmean(latencies) if latencies else 0.0, 6),
    }

    result_payload: dict[str, Any] = {
        "status": "completed",
        "trace_backend": get_trace_backend(),
        "trace_run_tag": args.trace_run_tag,
        "synthetic_only": bool(args.synthetic_only),
        "question_source": args.question_source,
        "questions": len(questions),
        "rounds": args.rounds,
        "concurrency": args.concurrency,
        "payload_profile": payload_profile,
        "trace_target_bytes": int(trace_target_bytes or 0),
        "inject_large_attributes": bool(args.inject_large_attributes),
        "successes": successes,
        "failures": failures,
        "attempts": total_attempts,
        "success_rate": round((successes / total_attempts) if total_attempts else 0.0, 6),
        "latency_summary": latency_summary,
        "otel_export_stats": get_export_stats(),
        "emitted_trace_count": len(emitted_traces),
        "sample_trace_ids": [trace.get("trace_id", "") for trace in emitted_traces[:15]],
        "run_started_at": run_started_at.isoformat(),
        "run_finished_at": run_finished_at.isoformat(),
    }

    print("=" * 80)
    print(f"Completed. successes={successes} failures={failures}")
    print(f"Latency p95={latency_summary['p95_seconds']:.3f}s")
    print("=" * 80)

    if args.metrics_output:
        output_path = Path(args.metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote metrics output to {output_path}")

    if args.trace_manifest_output:
        manifest_path = Path(args.trace_manifest_output)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        emitted_sorted = sorted(
            emitted_traces,
            key=lambda item: int(item.get("emitted_at_unix_ms", 0) or 0),
        )
        manifest_payload = {
            "trace_run_tag": args.trace_run_tag,
            "trace_backend": get_trace_backend(),
            "run_started_at": run_started_at.isoformat(),
            "run_finished_at": run_finished_at.isoformat(),
            "emitted_trace_count": len(emitted_sorted),
            "latest_emitted_at_unix_ms": max(
                (int(item.get("emitted_at_unix_ms", 0) or 0) for item in emitted_sorted),
                default=0,
            ),
            "traces": emitted_sorted,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote trace manifest to {manifest_path}")

    if args.fail_on_error and failures > 0:
        raise SystemExit(1)

    return result_payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate random questions and run through supervisor locally"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("CONCURRENCY", "1")),
        help="Number of concurrent questions to process (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Exact number of questions to generate.",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Backward-compatible alias for --questions.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.environ.get("ROUNDS", "1")),
        help="How many times to replay the same question corpus (default: 1)",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any request fails",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", "3")),
        help="Max retries for transient quota errors (default: 3)",
    )
    parser.add_argument(
        "--base-retry-seconds",
        type=float,
        default=float(os.environ.get("BASE_RETRY_SECONDS", "15")),
        help="Base retry delay used for exponential backoff (default: 15)",
    )
    parser.add_argument(
        "--inter-question-delay-seconds",
        type=float,
        default=float(os.environ.get("INTER_QUESTION_DELAY_SECONDS", "2")),
        help="Delay between processed batches to reduce burst rate (default: 2)",
    )
    parser.add_argument(
        "--quota-preflight",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("QUOTA_PREFLIGHT", "1") != "0",
        help="Run a lightweight Gemini call before batch and skip run if daily quota is exhausted",
    )
    parser.add_argument(
        "--model-pool",
        default=os.environ.get("MODEL_POOL", ",".join(DEFAULT_MODEL_POOL)),
        help="Comma-separated model IDs to sample from (default: gemini-2.0-flash-lite)",
    )
    parser.add_argument(
        "--question-source",
        choices=("generated", "bank"),
        default=os.environ.get("QUESTION_SOURCE", "generated"),
        help="Question source: generated (Gemini) or bank (deterministic local set)",
    )
    parser.add_argument(
        "--per-question-timeout-seconds",
        type=float,
        default=float(os.environ.get("PER_QUESTION_TIMEOUT_SECONDS", "120")),
        help="Fail a question if supervisor execution exceeds this timeout (default: 120s)",
    )
    parser.add_argument(
        "--payload-profile",
        choices=("baseline", "large", "xlarge"),
        default=os.environ.get("TRACE_PAYLOAD_PROFILE", "baseline"),
        help="Payload profile for synthetic OTEL attributes.",
    )
    parser.add_argument(
        "--inject-large-attributes",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("INJECT_LARGE_ATTRIBUTES", "0") in {"1", "true", "TRUE"},
        help="When enabled, amplify serialized trajectory attributes to emulate agent-scale spans.",
    )
    parser.add_argument(
        "--trace-run-tag",
        default=os.environ.get("TRACE_RUN_TAG", "manual-run"),
        help="Tag recorded on exported OTEL traces to group runs.",
    )
    parser.add_argument(
        "--metrics-output",
        default=None,
        help="Optional path to write a JSON run summary for aggregation.",
    )
    parser.add_argument(
        "--trace-manifest-output",
        default=None,
        help="Optional path to write emitted trace IDs/timestamps for freshness probes.",
    )
    parser.add_argument(
        "--trace-target-bytes",
        type=int,
        default=int(os.environ.get("TRACE_TARGET_BYTES", "0") or 0),
        help="Optional explicit target payload size per emitted trace (bytes).",
    )
    parser.add_argument(
        "--synthetic-only",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("SYNTHETIC_ONLY", "0") in {"1", "true", "TRUE"},
        help="Skip live model calls and emit deterministic synthetic trajectory spans only.",
    )

    args = parser.parse_args()

    if args.questions is None and args.num_questions is not None:
        args.questions = args.num_questions

    trace_backend = get_trace_backend()
    should_configure = trace_backend == "otlp" or bool(os.environ.get("BRAINTRUST_API_KEY"))
    if should_configure:
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
