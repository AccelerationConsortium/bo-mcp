"""Helpers for parsing tabular result uploads into result submission payloads."""

import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from bo_mcp_server.domain import ResultSubmissionInput


def parse_prefixed_result_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metadata_factory: Callable[[int], dict[str, Any]] | None = None,
) -> tuple[list[ResultSubmissionInput], list[str]]:
    """Parse rows using `param_` and `obj_` column prefixes.

    Args:
        rows: Iterable of row mappings.
        metadata_factory: Optional callback to build per-row metadata. Receives
            the 1-based CSV row number including the header offset (data starts at 2).

    Returns:
        Tuple of parsed results and row-level parse errors.
    """
    parsed_results: list[ResultSubmissionInput] = []
    parse_errors: list[str] = []

    for row_num, row in enumerate(rows, start=2):
        try:
            param_values: dict[str, Any] = {}
            obj_values: dict[str, float] = {}

            for key, value in row.items():
                if key is None or _is_missing(value):
                    continue
                if key.startswith("param_"):
                    param_values[key[6:]] = _parse_scalar(value)
                elif key.startswith("obj_"):
                    obj_name = key[4:]
                    try:
                        obj_values[obj_name] = float(value)
                    except (TypeError, ValueError):
                        parse_errors.append(
                            f"Row {row_num}: Invalid objective value for {obj_name}"
                        )

            if not param_values:
                parse_errors.append(
                    f"Row {row_num}: No parameter values found (use param_<name> columns)"
                )
                continue

            if not obj_values:
                parse_errors.append(
                    f"Row {row_num}: No objective values found (use obj_<name> columns)"
                )
                continue

            parsed_results.append(
                ResultSubmissionInput(
                    parameter_values=param_values,
                    objective_values=obj_values,
                    metadata=metadata_factory(row_num) if metadata_factory else {},
                )
            )
        except (ValueError, TypeError, KeyError, IndexError) as e:
            parse_errors.append(f"Row {row_num}: {e!s}")

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
        try:
            missing_params = [name for name in parameter_names if _is_missing(row.get(name))]
            if missing_params:
                parse_errors.append(f"Row {row_num}: Missing parameter values for {missing_params}")
                continue

            missing_objectives = [name for name in objective_names if _is_missing(row.get(name))]
            if missing_objectives:
                parse_errors.append(
                    f"Row {row_num}: Missing objective values for {missing_objectives}"
                )
                continue

            param_values = {name: _parse_scalar(row[name]) for name in parameter_names}

            obj_values: dict[str, float] = {}
            invalid_objectives: list[str] = []
            for name in objective_names:
                try:
                    obj_values[name] = float(row[name])
                except (TypeError, ValueError):
                    invalid_objectives.append(name)

            if invalid_objectives:
                parse_errors.append(
                    f"Row {row_num}: Invalid objective values for {invalid_objectives}"
                )
                continue

            parsed_results.append(
                ResultSubmissionInput(
                    parameter_values=param_values,
                    objective_values=obj_values,
                    metadata=metadata_factory(row_num) if metadata_factory else {},
                )
            )
        except (ValueError, TypeError, KeyError, IndexError) as e:
            parse_errors.append(f"Row {row_num}: {e!s}")

    return parsed_results, parse_errors


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
