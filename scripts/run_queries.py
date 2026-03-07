#!/usr/bin/env python3
"""Generate test questions and run them through the supervisor concurrently."""

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai

DEFAULT_BRAINTRUST_PROJECT = "pydantic-supervisor"

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.config import AgentConfig
from src.agent_graph import run_supervisor_with_critic
from src.modeling import ensure_google_api_keys, get_google_api_key
from src.tracing import configure_adk_tracing

load_dotenv()
ensure_google_api_keys()

MODEL_POOL = ["gemini-2.0-flash-lite"]
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


async def run_question(
    question: str,
    *,
    max_retries: int,
    base_retry_seconds: float,
) -> tuple[str, bool, bool]:
    """Run one question through the supervisor with a random model assignment."""
    from src.agent_graph import get_supervisor

    selected_model = random.choice(MODEL_POOL)
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
            result = await run_supervisor_with_critic(
                supervisor=supervisor,
                query=question,
                app_name="pydantic-supervisor-batch",
            )
            print(f"✅ {question[:80]} -> {str(result.get('final_output', ''))[:80]}")
            return question, True, False
        except Exception as exc:
            if not _is_resource_exhausted_error(exc):
                print(f"❌ {question[:80]} -> {exc}")
                return question, False, False

            if _is_hard_quota_exhausted(exc):
                print(f"⏹️ {question[:80]} -> hard quota exhausted ({exc})")
                return question, False, True

            if attempt > max_retries:
                print(f"❌ {question[:80]} -> exhausted retries ({exc})")
                return question, False, False

            suggested = _retry_delay_seconds(exc)
            backoff = base_retry_seconds * (2 ** (attempt - 1))
            sleep_s = max(suggested or 0.0, backoff)
            print(f"⏳ {question[:80]} -> retrying in {sleep_s:.1f}s after quota error")
            await asyncio.sleep(sleep_s)


async def main_async(args: argparse.Namespace) -> None:
    if args.quota_preflight:
        ok, reason = _quota_preflight_ok()
        if not ok:
            print("Hard quota appears exhausted; skipping this batch run.")
            print(reason)
            return

    num_questions = args.num_questions if args.num_questions is not None else random.randint(1, 100)
    questions = generate_questions(num_questions=num_questions, seed=args.seed)

    print(f"Generated {len(questions)} questions")
    print(f"Running with concurrency={args.concurrency}")
    print(f"Model pool: {', '.join(MODEL_POOL)}")
    print("=" * 80)

    successes = 0
    failures = 0
    hard_quota_stop = False

    for i in range(0, len(questions), args.concurrency):
        if hard_quota_stop:
            break
        batch = questions[i : i + args.concurrency]
        results = await asyncio.gather(
            *(
                run_question(
                    q,
                    max_retries=args.max_retries,
                    base_retry_seconds=args.base_retry_seconds,
                )
                for q in batch
            )
        )
        for _, ok, hard_stop in results:
            if ok:
                successes += 1
            else:
                failures += 1
            if hard_stop:
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
        default=None,
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
        help="Run a lightweight Gemini call before batch and skip run if daily quota is exhausted",
    )
    args = parser.parse_args()

    if os.environ.get("BRAINTRUST_API_KEY"):
        configure_adk_tracing(
            api_key=os.environ.get("BRAINTRUST_API_KEY"),
            project_id=os.environ.get("BRAINTRUST_PROJECT_ID"),
            project_name=os.environ.get("BRAINTRUST_PROJECT", DEFAULT_BRAINTRUST_PROJECT),
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
