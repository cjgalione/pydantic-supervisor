import json
import re
from pathlib import Path

import pytest

from scripts.run_queries import (
    FAILURE_PROVIDER_GATEWAY,
    FAILURE_QUOTA,
    FAILURE_RESEARCH,
    FAILURE_TIMEOUT,
    FAILURE_UNIT_CONVERSION,
    MIN_SCHEDULED_TOPIC_TRACES,
    QUESTION_BANK,
    QuestionResult,
    _build_summary,
    _classify_failure,
    _enforce_scheduled_topic_minimum,
    _preflight_failure_category,
    _select_questions,
    _trace_context_for_question,
    _write_summary,
)


def _workflow_text() -> str:
    return Path(".github/workflows/run_on_schedule.yml").read_text()


def test_select_questions_daily_bank_smoke_uses_each_bank_question_once() -> None:
    questions = _select_questions(num_questions=15, question_source="bank", seed=0)

    assert len(questions) == len(QUESTION_BANK)
    assert set(questions) == set(QUESTION_BANK)


def test_classify_failure_groups_known_daily_failure_modes() -> None:
    assert (
        _classify_failure("unsupported operand type(s) for -: 'ParserHelper' and 'ParserHelper'")
        == FAILURE_UNIT_CONVERSION
    )
    assert (
        _classify_failure("'horsepower_second' is not defined in the unit registry")
        == FAILURE_UNIT_CONVERSION
    )
    assert _classify_failure("Error code 429: quota exceeded") == FAILURE_QUOTA
    assert _classify_failure("timed out after 120.0s") == FAILURE_TIMEOUT
    assert _classify_failure("Braintrust gateway provider returned API status 500") == FAILURE_PROVIDER_GATEWAY
    assert _classify_failure("Tavily web search failed") == FAILURE_RESEARCH


def test_build_summary_groups_unique_failed_questions() -> None:
    results = [
        QuestionResult(question="ok", ok=True, final_output_preview="done"),
        QuestionResult(
            question="Convert 10^6 joules to horsepower-seconds.",
            ok=False,
            failure_category=FAILURE_UNIT_CONVERSION,
            error="'horsepower_second' is not defined in the unit registry",
        ),
        QuestionResult(
            question="Convert 10^6 joules to horsepower-seconds.",
            ok=False,
            failure_category=FAILURE_UNIT_CONVERSION,
            error="unsupported operand type(s) for -: 'ParserHelper' and 'ParserHelper'",
        ),
    ]

    summary = _build_summary(results)

    assert summary["total"] == 3
    assert summary["successes"] == 1
    assert summary["failures"] == 2
    assert summary["failure_categories"] == {FAILURE_UNIT_CONVERSION: 2}
    assert summary["unique_failed_questions"] == [
        {
            "question": "Convert 10^6 joules to horsepower-seconds.",
            "count": 2,
            "categories": [FAILURE_UNIT_CONVERSION],
            "errors": [
                "'horsepower_second' is not defined in the unit registry",
                "unsupported operand type(s) for -: 'ParserHelper' and 'ParserHelper'",
            ],
        }
    ]


def test_write_summary_outputs_json_artifact(tmp_path) -> None:
    summary_path = tmp_path / "query-summary.json"
    results = [
        QuestionResult(
            question="What is 37 * 24?",
            ok=True,
            model="gpt-4.1-mini",
            attempts=1,
            final_output_preview="888",
        )
    ]

    _write_summary(str(summary_path), results)

    payload = json.loads(summary_path.read_text())
    assert payload["total"] == 1
    assert payload["successes"] == 1
    assert payload["results"][0]["question"] == "What is 37 * 24?"


def test_preflight_classifies_auth_quota_and_transient_failures() -> None:
    assert _preflight_failure_category(RuntimeError("Incorrect API key")) == "authentication"
    assert _preflight_failure_category(RuntimeError("insufficient_quota")) == "quota"
    assert _preflight_failure_category(RuntimeError("connection timed out")) == "transient"


def test_scheduled_workflow_uses_full_traces_for_topics() -> None:
    workflow = _workflow_text()

    scheduled_count = re.search(
        r"NUM_QUESTIONS:\s*\$\{\{\s*github\.event_name == 'schedule' && '(\d+)'",
        workflow,
    )

    assert scheduled_count is not None
    assert int(scheduled_count.group(1)) > 0
    assert 'TRACE_PROFILE: "full"' in workflow


def test_scheduled_runs_raise_question_count_to_topics_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")

    assert _enforce_scheduled_topic_minimum(10) == MIN_SCHEDULED_TOPIC_TRACES
    assert _enforce_scheduled_topic_minimum(None) == MIN_SCHEDULED_TOPIC_TRACES
    assert _enforce_scheduled_topic_minimum(MIN_SCHEDULED_TOPIC_TRACES + 1) == (
        MIN_SCHEDULED_TOPIC_TRACES + 1
    )


def test_manual_runs_keep_requested_question_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")

    assert _enforce_scheduled_topic_minimum(10) == 10


def test_trace_context_makes_daily_smoke_rows_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("GITHUB_RUN_ID", "30651627589")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "abc123")

    first = _trace_context_for_question(question_number=1, question_total=50)
    second = _trace_context_for_question(question_number=2, question_total=50)

    assert first["source"] == "daily-supervisor-smoke"
    assert first["github_event_name"] == "schedule"
    assert first["github_run_id"] == "30651627589"
    assert first["github_run_attempt"] == "1"
    assert first["github_sha"] == "abc123"
    assert first["question_total"] == 50
    assert first["question_number"] == 1
    assert second["question_number"] == 2
