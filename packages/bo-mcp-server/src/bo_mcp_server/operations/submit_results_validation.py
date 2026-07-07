"""Shape-validation helpers for ``submit_results``.

Split from :mod:`bo_mcp_server.operations.submit_results` so per-row
shape checks (parameter presence, spec-bound enforcement, finite
parameter/objective values, finite measurement uncertainty) plus the
mutable bookkeeping (``_RowError``, ``_SubmitTracking``,
``_record_row_error``) live in a single module keyed by
responsibility.

The companion module :mod:`.submit_results_pipeline` consumes these
helpers from the per-row Phase 1 loop and the duplicate / budget
filter; the public operation entry point in
:mod:`.submit_results` only depends on the response envelope helpers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.domain.campaign_spec import InputParameter, ParameterType
from bo_mcp_server.field_errors import add_row_field_error


@dataclass(frozen=True)
class _RowError:
    """A single row-level validation error with its field path.

    ``field_path`` is dotted relative to the offending row (no
    ``results[i]`` prefix); the prefix is added by
    :func:`_record_row_error` when the error is recorded onto tracking.
    ``message`` is the human-facing string surfaced through the legacy
    ``errors: list[str]`` envelope and the per-row ``partial_results``
    payload — preserving the existing wording so older callers and
    assertions stay green.
    """

    field_path: str
    message: str


@dataclass
class _SubmitTracking:
    """Mutable state accumulated during result validation."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duplicates_detected: list[dict[str, Any]] = field(default_factory=list)
    partial_results: dict[int, str | dict[str, str]] = field(default_factory=dict)
    field_errors: dict[str, list[str]] = field(default_factory=dict)


def _record_row_error(
    tracking: _SubmitTracking,
    row_index: int,
    field_path: str,
    message: str,
) -> None:
    """Record a row error in both the legacy ``errors`` list and ``field_errors``.

    The legacy list keeps the human-readable, prefixed message shape
    (``Result 5: ...``) so existing callers and assertions are
    untouched. The ``field_errors`` map indexes the same finding by
    dotted path so agents can target the field directly.
    """
    tracking.errors.append(message)
    add_row_field_error(tracking.field_errors, row_index, field_path, message)


def _check_numeric_bounds(
    param: InputParameter,
    value: int | float | str,
    index: int,
    warnings: list[str],
) -> None:
    """Check a numeric value against parameter bounds."""
    if param.bounds is None:
        return
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        pname = param.name
        vtype = type(value).__name__
        warnings.append(f"Result {index}: parameter '{pname}' expected numeric, got {vtype}")
        return
    if numeric < param.bounds.lower or numeric > param.bounds.upper:
        warnings.append(
            f"Result {index}: parameter '{param.name}' value {numeric} "
            f"is outside spec bounds "
            f"[{param.bounds.lower}, {param.bounds.upper}]"
        )


def _validate_parameter_value(
    param: InputParameter,
    value: int | float | str,
    index: int,
    warnings: list[str],
) -> None:
    """Check a single parameter value against its spec definition.

    Out-of-bounds values are reported as warnings rather than hard errors
    because real experiments may intentionally exceed spec bounds.
    """
    if param.type == ParameterType.CONTINUOUS:
        _check_numeric_bounds(param, value, index, warnings)
    elif param.type == ParameterType.DISCRETE:
        if param.values is not None and value not in param.values:
            warnings.append(
                f"Result {index}: parameter '{param.name}' value {value} "
                f"is not in allowed discrete values {param.values}"
            )
        else:
            _check_numeric_bounds(param, value, index, warnings)
    elif (
        param.type == ParameterType.CATEGORICAL
        and param.categories is not None
        and value not in param.categories
    ):
        warnings.append(
            f"Result {index}: parameter '{param.name}' value "
            f"'{value}' is not in allowed categories "
            f"{param.categories}"
        )


def _validate_measurement_uncertainty(
    uncertainty: dict[str, float],
    objective_names: set[str],
    index: int,
    warnings: list[str],
) -> _RowError | None:
    """Validate measurement uncertainty keys and values.

    Unknown objective keys are surfaced as warnings (they are dropped on the
    way into bo-engine and do not corrupt the GP).

    Numerical defects on declared-objective values are hard errors: the
    bo-engine squares the stddev into ``train_yvar`` and routes the GP onto a
    ``FixedNoiseGaussianLikelihood``, so a negative value would silently
    become a positive variance and NaN/inf would propagate into MLL. The
    returned :class:`_RowError` pins the offending key in its
    ``field_path`` so the caller can surface it in ``field_errors``.
    """
    invalid_keys = set(uncertainty.keys()) - objective_names
    if invalid_keys:
        warnings.append(
            f"Result {index}: measurement_uncertainty has unknown "
            f"objective keys: {sorted(invalid_keys)}"
        )
    for obj_name, unc_val in uncertainty.items():
        # Unknown objective keys are stripped before reaching the engine, so
        # we tolerate odd values on them with the warning above. Declared
        # objectives, however, drive the GP and must be sane.
        if obj_name not in objective_names:
            continue
        if not isinstance(unc_val, (int, float)) or math.isnan(unc_val) or math.isinf(unc_val):
            return _RowError(
                field_path=f"measurement_uncertainty['{obj_name}']",
                message=(
                    f"Result {index}: measurement_uncertainty['{obj_name}'] "
                    f"is not a finite number: {unc_val}"
                ),
            )
        if unc_val < 0:
            return _RowError(
                field_path=f"measurement_uncertainty['{obj_name}']",
                message=(
                    f"Result {index}: measurement_uncertainty['{obj_name}'] "
                    f"is negative ({unc_val}); expected non-negative std"
                ),
            )
    return None


def _validate_parameter_finiteness(
    parameter_values: dict[str, Any],
    index: int,
) -> _RowError | None:
    """Reject non-finite numeric parameter values.

    A NaN coordinate silently bypasses the spec-bounds warning (every
    comparison against NaN is False) and, once persisted, poisons the
    surrogate's training inputs on every subsequent fit. Results cannot
    be deleted after submission, so the row must be stopped here.
    Non-numeric values are left to the spec validators: categorical
    parameters legitimately carry strings.
    """
    for pname, pvalue in parameter_values.items():
        if isinstance(pvalue, float) and not math.isfinite(pvalue):
            return _RowError(
                field_path=f"parameter_values['{pname}']",
                message=(
                    f"Result {index}: parameter_values['{pname}'] is not a finite number: {pvalue}"
                ),
            )
    return None


def _validate_objective_finiteness(
    objective_values: dict[str, Any],
    index: int,
) -> _RowError | None:
    """Reject non-numeric or non-finite objective values.

    Objective values become the surrogate's training targets, so a
    single NaN/inf entry would make every subsequent model fit fail
    (and thereby block suggestion generation for the whole campaign,
    since persisted results cannot be deleted). The Pydantic intake
    models already reject non-finite floats; this check re-runs at the
    shared operation layer so every transport is covered even when a
    caller bypasses model validation.
    """
    for obj_name, obj_val in objective_values.items():
        if not isinstance(obj_val, (int, float)) or math.isnan(obj_val) or math.isinf(obj_val):
            return _RowError(
                field_path=f"objective_values['{obj_name}']",
                message=(
                    f"Result {index}: objective_values['{obj_name}'] "
                    f"is not a finite number: {obj_val}"
                ),
            )
    return None


def _validate_single_result(
    index: int,
    r: ResultSubmissionInput,
    param_names: set[str],
    objective_names: set[str],
    warnings: list[str],
    parameters: list[InputParameter] | None = None,
) -> _RowError | None:
    """Validate a single result's parameter and objective *shape*.

    Shape-only checks: parameter presence, parameter/objective value
    finiteness, parameter spec validation (bounds/categories),
    objective presence, measurement-uncertainty finiteness. Cross-row
    checks (duplicate parameters, duplicate or stale ``suggestion_id``)
    live in the per-row loop in :func:`_validate_and_create_results` so
    they can see the running state of accepted rows; running them here
    would let a later row be rejected as the duplicate of an earlier
    row that was itself dropped by some other validator.
    """
    if missing_params := (param_names - set(r.parameter_values.keys())):
        return _RowError(
            field_path="parameter_values",
            message=f"Result {index} missing parameters: {missing_params}",
        )

    param_error = _validate_parameter_finiteness(r.parameter_values, index)
    if param_error is not None:
        return param_error

    # Validate parameter values against spec bounds/categories
    if parameters is not None:
        param_by_name = {p.name: p for p in parameters}
        for pname, pvalue in r.parameter_values.items():
            if pname in param_by_name:
                _validate_parameter_value(param_by_name[pname], pvalue, index, warnings)

    if missing_objectives := (objective_names - set(r.objective_values.keys())):
        return _RowError(
            field_path="objective_values",
            message=f"Result {index} missing objectives: {missing_objectives}",
        )

    objective_error = _validate_objective_finiteness(r.objective_values, index)
    if objective_error is not None:
        return objective_error

    if r.measurement_uncertainty is not None:
        unc_error = _validate_measurement_uncertainty(
            r.measurement_uncertainty, objective_names, index, warnings
        )
        if unc_error is not None:
            return unc_error

    return None
