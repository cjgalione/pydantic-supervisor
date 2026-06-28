import math

from src.agents.deep_agent import (
    _fallback_numeric_from_operation_text,
    _fallback_unit_conversion_from_operation_text,
    _format_fallback_math_response,
    _format_fallback_unit_conversion_response,
    _query_needs_math_handoff,
)
from src.agents.math_agent import _convert_units_tool, convert_units


def test_convert_units_horsepower_seconds_alias() -> None:
    result = convert_units(1_000_000, "joules", "horsepower-seconds")
    assert math.isclose(result, 1341.0220895950278, rel_tol=1e-9)


def test_convert_units_horsepower_seconds_observed_variants() -> None:
    variants = [
        "horsepower_seconds",
        "horsepower-second",
        "horsepower-seconds (hp*s)",
        "horsepower - seconds",
        "hp s",
    ]

    for variant in variants:
        result = convert_units(1_000_000, "joules", variant)
        assert math.isclose(result, 1341.0220895950278, rel_tol=1e-9)


def test_convert_units_tolerates_noisy_lightbulb_hours_target() -> None:
    result = convert_units(
        1e44,
        "joules",
        (
            "60W lightbulb-hours. First, calculate how many joules a 60W lightbulb uses in one "
            "hour (60W * 3600 seconds), then divide 10^44 joules by that value."
        ),
    )
    expected = 1e44 / (60 * 3600)
    assert math.isclose(result, expected, rel_tol=1e-12)


def test_convert_units_tolerates_noisy_lightbulb_hours_variants() -> None:
    variants = [
        "60 W lightbulb hours",
        "60-watt light bulb hours",
        "60 watts light-bulb-hours. divide by watts times seconds per hour.",
    ]

    for variant in variants:
        result = convert_units(1e44, "joules", variant)
        expected = 1e44 / (60 * 3600)
        assert math.isclose(result, expected, rel_tol=1e-12)


def test_convert_units_tool_is_registered_under_public_name() -> None:
    assert _convert_units_tool.__tool_name__ == "convert_units"


def test_convert_units_tool_returns_safe_failure_text_without_raising() -> None:
    result = _convert_units_tool(1, "joules", "bananas")

    assert isinstance(result, str)
    assert result.startswith("Unit conversion failed:")


def test_convert_units_tool_returns_numeric_result_for_observed_variant() -> None:
    result = _convert_units_tool(1_000_000, "joules", "horsepower-seconds (hp*s)")

    assert isinstance(result, float)
    assert math.isclose(result, 1341.0220895950278, rel_tol=1e-9)


def test_fallback_numeric_handles_square_root_question() -> None:
    result = _fallback_numeric_from_operation_text("What is the square root of 2025?")
    assert result == 45


def test_format_fallback_math_response_for_square_root() -> None:
    response = _format_fallback_math_response(
        "What is the square root of 2025?",
        45,
        "explanatory",
    )
    assert response == "The square root of 2025 is 45."


def test_fallback_unit_conversion_handles_scheduled_horsepower_seconds_query() -> None:
    result = _fallback_unit_conversion_from_operation_text(
        "Convert 10^6 joules to horsepower-seconds."
    )

    assert result is not None
    assert math.isclose(result, 1341.0220895950278, rel_tol=1e-9)


def test_format_fallback_unit_conversion_response() -> None:
    response = _format_fallback_unit_conversion_response(
        "Convert 10^6 joules to horsepower-seconds.",
        1341.0220895950278,
        "explanatory",
    )

    assert response == "1e+06 joules is approximately 1341.02 horsepower-seconds."


def test_unit_conversion_query_requires_math_handoff() -> None:
    assert _query_needs_math_handoff("Convert 10^6 joules to horsepower-seconds.")
