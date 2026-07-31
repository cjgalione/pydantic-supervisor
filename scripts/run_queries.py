#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.agent_graph import run_supervisor_with_critic  # noqa: E402
from src.config import AgentConfig  # noqa: E402
from src.tracing import configure_adk_tracing  # noqa: E402

load_dotenv()

DEFAULT_MODEL_POOL = ["gpt-4.1-mini"]
MIN_SCHEDULED_TOPIC_TRACES = 50
FAILURE_TIMEOUT = "timeout"
FAILURE_QUOTA = "quota"
FAILURE_UNIT_CONVERSION = "unit_conversion"
FAILURE_PROVIDER_GATEWAY = "provider_gateway"
FAILURE_RESEARCH = "research"
FAILURE_UNEXPECTED = "unexpected"

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


@dataclass
class QuestionResult:
    question: str
    ok: bool
    hard_stop: bool = False
    failure_category: str | None = None
    error: str | None = None
    model: str | None = None
    attempts: int = 1
    final_output_preview: str = ""


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


def _classify_failure(error_text: str) -> str:
    lowered = error_text.lower()

    if "timed out" in lowered or "timeout" in lowered:
        return FAILURE_TIMEOUT

    if any(
        marker in lowered
        for marker in (
            "resource_exhausted",
            "quota exceeded",
            "error code 429",
            "ratelimit",
            "rate limit",
            "generaterequestsperday",
        )
    ):
        return FAILURE_QUOTA

    if any(
        marker in lowered
        for marker in (
            "parserhelper",
            "unit registry",
            "undefinedunit",
            "pint",
            "horsepower_second",
            "unsupported operand type(s) for -",
        )
    ):
        return FAILURE_UNIT_CONVERSION

    if any(
        marker in lowered
        for marker in (
            "gateway",
            "openai",
            "provider",
            "api status",
            "apierror",
            "badrequest",
            "connection error",
            "connectionerror",
        )
    ):
        return FAILURE_PROVIDER_GATEWAY

    if any(marker in lowered for marker in ("tavily", "web search", "search failed")):
        return FAILURE_RESEARCH

    return FAILURE_UNEXPECTED


def _parse_model_pool(raw_model_pool: str | None) -> list[str]:
    if not raw_model_pool:
        return DEFAULT_MODEL_POOL.copy()

    models = [candidate.strip() for candidate in raw_model_pool.split(",")]
    models = [model for model in models if model]
    if not models:
        return DEFAULT_MODEL_POOL.copy()
    return models


def generate_questions(num_questions: int, seed: Optional[int] = None) -> list[str]:
    """Generate realistic, varied questions from the local bank."""
    rng = random.Random(seed)
    return _fallback_questions(num_questions=num_questions, rng=rng)


def _select_questions(
    *,
    num_questions: int | None,
    question_source: str,
    seed: int | None,
) -> list[str]:
    resolved_num_questions = num_questions if num_questions is not None else random.randint(1, 100)
    rng = random.Random(seed)
    if question_source == "bank":
        return _fallback_questions(num_questions=resolved_num_questions, rng=rng)
    return generate_questions(num_questions=resolved_num_questions, seed=seed)


def _enforce_scheduled_topic_minimum(num_questions: int | None) -> int | None:
    if os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        return num_questions

    if num_questions is not None and num_questions >= MIN_SCHEDULED_TOPIC_TRACES:
        return num_questions

    requested = "random default" if num_questions is None else str(num_questions)
    print(
        "Scheduled run requested "
        f"{requested} questions; raising to {MIN_SCHEDULED_TOPIC_TRACES} "
        "so the daily Topics window has enough traces."
    )
    return MIN_SCHEDULED_TOPIC_TRACES


def _trace_context_for_question(*, question_number: int, question_total: int) -> dict[str, object]:
    context: dict[str, object] = {
        "question_number": question_number,
        "question_total": question_total,
        "source": "daily-supervisor-smoke",
    }
    for name in ("GITHUB_EVENT_NAME", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_SHA"):
        value = os.environ.get(name)
        if value:
            context[name.lower()] = value
    return context


def _build_summary(
    results: list[QuestionResult], preflight: dict[str, str] | None = None
) -> dict[str, object]:
    failures = [result for result in results if not result.ok]
    successes = len(results) - len(failures)
    category_counts = Counter(
        result.failure_category or FAILURE_UNEXPECTED for result in failures
    )

    failures_by_question: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in failures:
        failures_by_question[result.question].append(result)

    unique_failed_questions = []
    for question, question_failures in sorted(failures_by_question.items()):
        unique_failed_questions.append(
            {
                "question": question,
                "count": len(question_failures),
                "categories": sorted(
                    {
                        failure.failure_category or FAILURE_UNEXPECTED
                        for failure in question_failures
                    }
                ),
                "errors": sorted(
                    {failure.error or "" for failure in question_failures if failure.error}
                )[:5],
            }
        )

    return {
        "preflight": preflight or {},
        "total": len(results),
        "successes": successes,
        "failures": len(failures),
        "failure_categories": dict(sorted(category_counts.items())),
        "unique_failed_questions": unique_failed_questions,
        "results": [asdict(result) for result in results],
    }


def _write_summary(
    path: str | None,
    results: list[QuestionResult],
    preflight: dict[str, str] | None = None,
) -> None:
    if not path:
        return
    summary_path = Path(path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(_build_summary(results, preflight), indent=2) + "\n")


def _print_failure_summary(results: list[QuestionResult]) -> None:
    failures = [result for result in results if not result.ok]
    if not failures:
        return

    print("Failure summary:")
    for category, count in sorted(
        Counter(result.failure_category or FAILURE_UNEXPECTED for result in failures).items()
    ):
        print(f"- {category}: {count}")

    failures_by_question: dict[str, list[QuestionResult]] = defaultdict(list)
    for result in failures:
        failures_by_question[result.question].append(result)

    print("Unique failed questions:")
    for question, question_failures in sorted(failures_by_question.items()):
        categories = sorted(
            {failure.failure_category or FAILURE_UNEXPECTED for failure in question_failures}
        )
        first_error = next((failure.error for failure in question_failures if failure.error), "")
        print(
            f"- count={len(question_failures)} categories={','.join(categories)} "
            f"question={question!r} error={first_error}"
        )


def _quota_preflight_ok() -> tuple[bool, str]:
    return True, ""


def _preflight_failure_category(exc: Exception) -> str:
    text = str(exc).lower()
    if "insufficient_quota" in text or "exceeded your current quota" in text:
        return "quota"
    if any(
        marker in text
        for marker in ("authentication", "invalid api key", "incorrect api key", "unauthorized", "401")
    ):
        return "authentication"
    if any(marker in text for marker in ("429", "timeout", "connection", "temporarily")):
        return "transient"
    return "provider"


def _run_preflight() -> dict[str, str]:
    missing = [name for name in ("BRAINTRUST_API_KEY", "EXA_API_KEY") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}")

    from openai import OpenAI

    from src.agents.research_agent import _search_exa

    for attempt in range(1, 4):
        try:
            client = OpenAI(
                api_key=os.environ["BRAINTRUST_API_KEY"],
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
            client.chat.completions.create(
                model=_parse_model_pool(os.environ.get("MODEL_POOL"))[0],
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            )
            _search_exa(query="Braintrust", max_results=1)
            return {"braintrust": "ok", "model": "ok", "exa": "ok"}
        except Exception as exc:
            category = _preflight_failure_category(exc)
            if category == "transient" and attempt < 3:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"Provider preflight failed ({category}).") from exc

    raise RuntimeError("Provider preflight failed (transient).")


async def run_question(
    question: str,
    *,
    model_pool: list[str],
    per_question_timeout_seconds: float,
    max_retries: int,
    base_retry_seconds: float,
    trace_context: dict[str, object] | None = None,
) -> QuestionResult:
    """Run one question through the supervisor with a random model assignment."""
    from src.agent_graph import get_supervisor

    selected_model = random.choice(model_pool)
    config = AgentConfig(
        supervisor_model=selected_model,
        research_model=selected_model,
        math_model=selected_model,
    )
    supervisor = get_supervisor(config=config, force_rebuild=True)

    attempt = 0
    while True:
        attempt += 1
        try:
            result = await asyncio.wait_for(
                run_supervisor_with_critic(
                    supervisor=supervisor,
                    query=question,
                    app_name="pydantic-supervisor-batch",
                    trace_context=trace_context,
                ),
                timeout=per_question_timeout_seconds,
            )
            final_output_preview = str(result.get("final_output", ""))[:80]
            print(f"✅ {question[:80]} -> {final_output_preview}")
            return QuestionResult(
                question=question,
                ok=True,
                model=selected_model,
                attempts=attempt,
                final_output_preview=final_output_preview,
            )
        except TimeoutError:
            error = f"timed out after {per_question_timeout_seconds:.1f}s"
            print(f"❌ [{FAILURE_TIMEOUT}] {question[:80]} -> {error}")
            return QuestionResult(
                question=question,
                ok=False,
                failure_category=FAILURE_TIMEOUT,
                error=error,
                model=selected_model,
                attempts=attempt,
            )
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                error = str(exc)
                category = _classify_failure(error)
                print(f"❌ [{category}] {question[:80]} -> {error}")
                return QuestionResult(
                    question=question,
                    ok=False,
                    failure_category=category,
                    error=error,
                    model=selected_model,
                    attempts=attempt,
                )

            if _is_hard_quota_exhausted(exc):
                error = f"hard quota exhausted ({exc})"
                print(f"⏹️ [{FAILURE_QUOTA}] {question[:80]} -> {error}")
                return QuestionResult(
                    question=question,
                    ok=False,
                    hard_stop=True,
                    failure_category=FAILURE_QUOTA,
                    error=error,
                    model=selected_model,
                    attempts=attempt,
                )

            if attempt > max_retries:
                error = f"exhausted retries ({exc})"
                print(f"❌ [{FAILURE_QUOTA}] {question[:80]} -> {error}")
                return QuestionResult(
                    question=question,
                    ok=False,
                    failure_category=FAILURE_QUOTA,
                    error=error,
                    model=selected_model,
                    attempts=attempt,
                )

            suggested = _retry_delay_seconds(exc)
            backoff = base_retry_seconds * (2 ** (attempt - 1))
            sleep_s = max(suggested or 0.0, backoff)
            print(f"⏳ {question[:80]} -> retrying in {sleep_s:.1f}s after quota error")
            await asyncio.sleep(sleep_s)


async def main_async(args: argparse.Namespace) -> None:
    preflight = {} if args.skip_preflight else _run_preflight()
    if args.preflight_only:
        _write_summary(args.summary_path, [], preflight)
        print("Provider preflight passed.")
        return

    num_questions = _enforce_scheduled_topic_minimum(args.num_questions)
    questions = _select_questions(
        num_questions=num_questions,
        question_source=args.question_source,
        seed=args.seed,
    )

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency}")
    model_pool = _parse_model_pool(args.model_pool)
    print(f"Model pool: {', '.join(model_pool)}")
    print(f"Question source: {args.question_source}")
    print(f"Per-question timeout: {args.per_question_timeout_seconds:.1f}s")
    print("=" * 80)

    successes = 0
    failures = 0
    hard_quota_stop = False
    all_results: list[QuestionResult] = []

    for i in range(0, len(questions), args.concurrency):
        if hard_quota_stop:
            break
        batch = list(enumerate(questions[i : i + args.concurrency], start=i + 1))
        results = await asyncio.gather(
            *(
                run_question(
                    q,
                    model_pool=model_pool,
                    per_question_timeout_seconds=args.per_question_timeout_seconds,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                    trace_context=_trace_context_for_question(
                        question_number=question_number,
                        question_total=len(questions),
                    ),
                )
                for question_number, q in batch
            )
        )
        all_results.extend(results)
        for result in results:
            if result.ok:
                successes += 1
            else:
                failures += 1
            if result.hard_stop:
                hard_quota_stop = True
        if hard_quota_stop:
            print("Hard quota exhausted; stopping remaining questions to avoid repeated 429s.")
            break
        if args.inter_question_delay_seconds > 0:
            await asyncio.sleep(args.inter_question_delay_seconds)
        print()

    print("=" * 80)
    print(f"Completed. successes={successes} failures={failures}")
    print("=" * 80)
    _print_failure_summary(all_results)
    _write_summary(args.summary_path, all_results, preflight)

    if args.fail_on_error and failures > 0:
        raise SystemExit(1)


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
        default=int(os.environ["SEED"]) if os.environ.get("SEED") else None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Exact number of questions to generate (default: random 1-100)",
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
        help="Run preflight checks before processing batches",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify the configured model and Exa adapter without running questions",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip provider preflight after a separate successful preflight step",
    )
    parser.add_argument(
        "--model-pool",
        default=os.environ.get("MODEL_POOL", ",".join(DEFAULT_MODEL_POOL)),
        help="Comma-separated model IDs to sample from (default: gpt-4.1-mini)",
    )
    parser.add_argument(
        "--question-source",
        choices=("generated", "bank"),
        default=os.environ.get("QUESTION_SOURCE", "generated"),
        help="Question source: generated (local) or bank (deterministic local set)",
    )
    parser.add_argument(
        "--per-question-timeout-seconds",
        type=float,
        default=float(os.environ.get("PER_QUESTION_TIMEOUT_SECONDS", "120")),
        help="Fail a question if supervisor execution exceeds this timeout (default: 120s)",
    )
    parser.add_argument(
        "--summary-path",
        default=os.environ.get("QUERY_SUMMARY_PATH", ""),
        help="Optional path for a JSON query result summary artifact",
    )
    args = parser.parse_args()

    if os.environ.get("BRAINTRUST_API_KEY"):
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
            org_name=os.environ.get("BRAINTRUST_ORG_NAME"),
        )

    try:
        asyncio.run(main_async(args))
    finally:
        if os.environ.get("BRAINTRUST_API_KEY"):
            from braintrust import flush

            flush()


if __name__ == "__main__":
    main()
