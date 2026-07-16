import json

from scripts.run_queries import (
    FAILURE_PROVIDER_GATEWAY,
    FAILURE_QUOTA,
    FAILURE_RESEARCH,
    FAILURE_TIMEOUT,
    FAILURE_UNIT_CONVERSION,
    QUESTION_BANK,
    QuestionResult,
    _build_summary,
    _classify_failure,
    _preflight_failure_category,
    _select_questions,
    _write_summary,
)


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
