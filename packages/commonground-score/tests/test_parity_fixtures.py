from __future__ import annotations

import json
from math import isclose
from pathlib import Path
from typing import Any

import pytest

import commonground_score

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def test_typescript_parity_fixtures() -> None:
    fixture_paths = sorted(FIXTURE_DIR.glob("parity_*.json"))
    if not fixture_paths:
        pytest.skip("no parity fixtures found")

    for fixture_path in fixture_paths:
        payload = json.loads(fixture_path.read_text())
        function_name = payload["function"]
        function = getattr(commonground_score, function_name)
        for case in payload["cases"]:
            actual = function(*case["args"])
            _assert_close(actual, case["expected"])


def _assert_close(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            _assert_close(actual[key], expected_value)
        return
    if isinstance(expected, float):
        assert isclose(actual, expected, abs_tol=1e-9)
        return
    assert actual == expected
