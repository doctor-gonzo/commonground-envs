from __future__ import annotations

import json
from math import isclose, nan
from pathlib import Path
from typing import Any

import commonground_score

FIXTURE_DIR = Path(__file__).parent / "fixtures"
EXPECTED_FIXTURES = {
    "comment_stats": "parity_comment_stats.json",
    "prop_test": "parity_prop_test.json",
    "rating_to_vote": "parity_rating_to_vote.json",
    "two_prop_test": "parity_two_prop_test.json",
}


def test_typescript_parity_fixtures() -> None:
    fixture_paths = {
        fixture_path.name: fixture_path
        for fixture_path in sorted(FIXTURE_DIR.glob("parity_*.json"))
    }
    assert set(fixture_paths) == set(EXPECTED_FIXTURES.values())

    for function_name, fixture_name in EXPECTED_FIXTURES.items():
        fixture_path = fixture_paths[fixture_name]
        payload = json.loads(fixture_path.read_text())
        assert payload["function"] == function_name
        function = getattr(commonground_score, function_name)
        for case_index, case in enumerate(payload["cases"]):
            args = [_decode_argument(arg) for arg in case["args"]]
            actual = function(*args)
            _assert_close(actual, case["expected"], fixture_path, case_index)


def _decode_argument(value: Any) -> Any:
    if value == "NaN":
        return nan
    if isinstance(value, list):
        return [_decode_argument(item) for item in value]
    if isinstance(value, dict):
        return {key: _decode_argument(item) for key, item in value.items()}
    return value


def _assert_close(
    actual: Any,
    expected: Any,
    fixture_path: Path,
    case_index: int,
) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_close(actual[key], expected_value, fixture_path, case_index)
        return
    if isinstance(expected, float):
        assert isclose(actual, expected, abs_tol=1e-9), (
            f"{fixture_path.name} case {case_index}: expected {expected!r}, "
            f"got {actual!r}"
        )
        return
    assert actual == expected, (
        f"{fixture_path.name} case {case_index}: expected {expected!r}, "
        f"got {actual!r}"
    )
