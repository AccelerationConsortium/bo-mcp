"""Non-finite objective cells in tabular result uploads.

Uploaded spreadsheets can carry ``nan``/``inf`` cells — Excel error
propagation, ``1e999`` overflow in CSV exports, or literal ``"inf"``
strings (which Python's ``float()`` happily parses). Objective columns
feed model training, so the parser must report such cells as invalid
rows instead of materialising non-finite floats that would poison the
campaign once persisted.

References:
- Python ``float()`` accepts ``"nan"``/``"inf"``/``"Infinity"`` spellings:
  https://docs.python.org/3/library/functions.html#float
- IEEE 754 Special Values: https://en.wikipedia.org/wiki/IEEE_754
"""

from __future__ import annotations

import pytest

from bo_mcp_server.result_upload_parser import (
    parse_named_result_rows,
    parse_prefixed_result_rows,
)

_NON_FINITE_CELLS = ["nan", "inf", "-inf", "Infinity", float("inf"), float("-inf"), 1e999]
_NON_FINITE_CELL_IDS = [
    "nan-string",
    "inf-string",
    "negative-inf-string",
    "infinity-string",
    "inf-float",
    "negative-inf-float",
    "overflow-to-inf",
]


@pytest.mark.parametrize("cell", _NON_FINITE_CELLS, ids=_NON_FINITE_CELL_IDS)
def test_prefixed_rows_reject_non_finite_objective_cells(cell: object) -> None:
    """A non-finite objective cell fails the row with an explicit error."""
    results, errors = parse_prefixed_result_rows([{"param_x": 0.5, "obj_y": cell}])
    assert results == []
    assert any("Invalid objective value" in e for e in errors)


@pytest.mark.parametrize("cell", _NON_FINITE_CELLS, ids=_NON_FINITE_CELL_IDS)
def test_named_rows_reject_non_finite_objective_cells(cell: object) -> None:
    """The named-column parser applies the same finiteness rule."""
    results, errors = parse_named_result_rows(
        [{"x": 0.5, "y": cell}],
        parameter_names=["x"],
        objective_names=["y"],
    )
    assert results == []
    assert any("Invalid objective values" in e for e in errors)


def test_prefixed_rows_treat_nan_float_cell_as_missing() -> None:
    """A float NaN cell (how xlsx blanks arrive) reads as a missing value.

    Either way the row is rejected — the NaN never reaches a parsed
    result payload.
    """
    results, errors = parse_prefixed_result_rows([{"param_x": 0.5, "obj_y": float("nan")}])
    assert results == []
    assert any("No objective values found" in e for e in errors)


def test_finite_objective_cells_still_parse() -> None:
    """Ordinary numeric cells (including numeric strings) parse cleanly."""
    results, errors = parse_prefixed_result_rows([{"param_x": 0.5, "obj_y": "1.25"}])
    assert errors == []
    assert len(results) == 1
    assert results[0].objective_values == {"y": 1.25}
    assert results[0].parameter_values == {"x": 0.5}
