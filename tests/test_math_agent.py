import math

from src.agents.deep_agent import _fallback_numeric_from_operation_text, _format_fallback_math_response
from src.agents.math_agent import convert_units


def test_convert_units_horsepower_seconds_alias() -> None:
    result = convert_units(1_000_000, "joules", "horsepower-seconds")
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
