"""``ResultSubmissionInput`` objective-value finiteness enforcement.

Objective values become the surrogate's training targets. A plain
``float`` field accepts NaN/±inf (``json.loads`` and pydantic-core both
admit the non-standard ``NaN``/``Infinity`` literals, and ``1e999``
overflows to ``inf`` in any parser); once such a value persists it
fails every subsequent model fit and results cannot be deleted, so the
intake schema rejects non-finite values outright.

References:
- Pydantic ``allow_inf_nan`` field constraint:
  https://docs.pydantic.dev/latest/api/fields/#pydantic.fields.Field
- IEEE 754 Special Values: https://en.wikipedia.org/wiki/IEEE_754
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from bo_mcp_server.domain import ResultSubmissionInput

_VALID_PARAMS = {"x": 0.5}


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_objective_values_rejected(bad_value: float) -> None:
    """Python-mode validation pins the offending objective entry."""
    with pytest.raises(ValidationError) as exc_info:
        ResultSubmissionInput(parameter_values=_VALID_PARAMS, objective_values={"y": bad_value})
    error = exc_info.value.errors()[0]
    assert error["type"] == "finite_number"
    assert error["loc"] == ("objective_values", "y")


@pytest.mark.parametrize(
    "payload",
    [
        '{"parameter_values": {"x": 0.5}, "objective_values": {"y": NaN}}',
        '{"parameter_values": {"x": 0.5}, "objective_values": {"y": Infinity}}',
        '{"parameter_values": {"x": 0.5}, "objective_values": {"y": -Infinity}}',
        '{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1e999}}',
    ],
    ids=["nan-literal", "infinity-literal", "negative-infinity-literal", "overflow-to-inf"],
)
def test_non_finite_json_payloads_rejected(payload: str) -> None:
    """JSON smuggling routes (non-standard literals, overflow) are closed.

    pydantic-core's JSON parser accepts the ``NaN``/``Infinity``
    literals just like Python's ``json.loads``, and ``1e999`` silently
    becomes ``inf`` — so the finiteness check must fire *after*
    parsing, on the parsed value.
    """
    with pytest.raises(ValidationError) as exc_info:
        ResultSubmissionInput.model_validate_json(payload)
    error = exc_info.value.errors()[0]
    assert error["type"] == "finite_number"
    assert error["loc"] == ("objective_values", "y")


def test_finite_objective_values_accepted() -> None:
    """Ordinary measurements still validate — the constraint is targeted."""
    row = ResultSubmissionInput(parameter_values=_VALID_PARAMS, objective_values={"y": 1.5})
    assert row.objective_values == {"y": 1.5}
