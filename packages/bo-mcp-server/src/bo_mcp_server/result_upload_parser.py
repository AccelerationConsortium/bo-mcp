"""Helpers for parsing tabular result uploads into result submission payloads."""

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from bo_mcp_server.domain import ResultSubmissionInput

# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def parse_prefixed_result_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metadata_factory: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[list[ResultSubmissionInput], list[str]]:
    """Parse rows using ``param_`` and ``obj_`` column prefixes.

    Returns:
        Tuple of parsed results and row-level parse errors.
    """
    parsed_results: list[ResultSubmissionInput] = []
    parse_errors: list[str] = []

    for row_num, row in enumerate(rows, start=2):
        result, errors = _parse_prefixed_row(row, row_num)
        parse_errors.extend(errors)
        if result is not None:
            if metadata_factory:
                result = ResultSubmissionInput(
                    parameter_values=result.parameter_values,
                    objective_values=result.objective_values,
                    metadata=metadata_factory(row_num),
                )
            parsed_results.append(result)

    return parsed_results, parse_errors


def parse_named_result_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    parameter_names: list[str],
    objective_names: list[str],
    metadata_factory: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[list[ResultSubmissionInput], list[str]]:
    """Parse rows using explicit parameter/objective column names."""
    parsed_results: list[ResultSubmissionInput] = []
    parse_errors: list[str] = []

    for row_num, row in enumerate(rows, start=2):
        result, errors = _parse_named_row(row, row_num, parameter_names, objective_names)
        parse_errors.extend(errors)
        if result is not None:
            if metadata_factory:
                result = ResultSubmissionInput(
                    parameter_values=result.parameter_values,
                    objective_values=result.objective_values,
                    metadata=metadata_factory(row_num),
                )
            parsed_results.append(result)

    return parsed_results, parse_errors


# ---------------------------------------------------------------------------
# Per-row parsing helpers
# ---------------------------------------------------------------------------


def _parse_prefixed_row(
    row: Mapping[str, Any],
    row_num: int,
) -> tuple[ResultSubmissionInput | None, list[str]]:
    """Parse a single row with ``param_`` / ``obj_`` prefixed columns."""
    errors: list[str] = []
    param_values: dict[str, Any] = {}
    obj_values: dict[str, float] = {}

    for key, value in row.items():
        if key is None or _is_missing(value):
            continue
        if key.startswith("param_"):
            param_values[key[6:]] = _parse_scalar(value)
        elif key.startswith("obj_"):
            obj_name = key[4:]
            parsed = _try_parse_float(value)
            if parsed is None:
                errors.append(f"Row {row_num}: Invalid objective value for {obj_name}")
            else:
                obj_values[obj_name] = parsed

    if not param_values:
        errors.append(f"Row {row_num}: No parameter values found (use param_<name> columns)")
        return None, errors

    if not obj_values:
        errors.append(f"Row {row_num}: No objective values found (use obj_<name> columns)")
        return None, errors

    return ResultSubmissionInput(parameter_values=param_values, objective_values=obj_values), errors


def _parse_named_row(
    row: Mapping[str, Any],
    row_num: int,
    parameter_names: list[str],
    objective_names: list[str],
) -> tuple[ResultSubmissionInput | None, list[str]]:
    """Parse a single row using explicit column names."""
    errors: list[str] = []

    missing_params = [n for n in parameter_names if _is_missing(row.get(n))]
    if missing_params:
        errors.append(f"Row {row_num}: Missing parameter values for {missing_params}")
        return None, errors

    missing_objectives = [n for n in objective_names if _is_missing(row.get(n))]
    if missing_objectives:
        errors.append(f"Row {row_num}: Missing objective values for {missing_objectives}")
        return None, errors

    param_values = {name: _parse_scalar(row[name]) for name in parameter_names}
    obj_values, invalid = _parse_objective_values(row, objective_names)

    if invalid:
        errors.append(f"Row {row_num}: Invalid objective values for {invalid}")
        return None, errors

    return ResultSubmissionInput(parameter_values=param_values, objective_values=obj_values), errors


def _parse_objective_values(
    row: Mapping[str, Any],
    objective_names: list[str],
) -> tuple[dict[str, float], list[str]]:
    """Parse objective columns, returning values and list of invalid names."""
    obj_values: dict[str, float] = {}
    invalid: list[str] = []
    for name in objective_names:
        parsed = _try_parse_float(row[name])
        if parsed is None:
            invalid.append(name)
        else:
            obj_values[name] = parsed
    return obj_values, invalid


# ---------------------------------------------------------------------------
# Scalar utilities
# ---------------------------------------------------------------------------


def _try_parse_float(value: Any) -> float | None:
    """Try to convert *value* to float, returning None on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_scalar(value: Any) -> Any:
    """Parse string scalars to int/float when possible; keep other values as-is."""
    if not isinstance(value, str):
        return value

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _is_missing(value: Any) -> bool:
    """Return True when the value is absent from an uploaded row."""
    return value is None or (isinstance(value, float) and math.isnan(value))
