"""Constraint satisfaction diagnostics."""

import logging
from typing import Any

import torch
from bo_engine.constants import CONSTRAINT_CALIBRATION_WARN_THRESHOLD
from bo_engine.device import get_device, get_dtype
from bo_engine.diagnostics import ConstraintSatisfactionMetrics, compute_constraint_satisfaction
from bo_engine.outcome_constraints import (
    OutcomeConstraintSpec as EngineOutcomeConstraintSpec,
)
from bo_engine.outcome_constraints import (
    compute_outcome_constraint_calibration,
)
from bo_engine.transforms import get_bounds_tensor, stack_encoded_values
from bo_engine.types import OptimizationSpec

from bo_mcp_server.converters import campaign_spec_to_optimization_spec
from bo_mcp_server.domain import CampaignSpec, Result
from bo_mcp_server.domain.campaign_spec import OutcomeConstraint

logger = logging.getLogger(__name__)


def compute_constraint_satisfaction_metrics(
    results: list[Result],
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute constraint satisfaction rate over time."""
    if not spec.constraints:
        diagnostics["constraint_satisfaction"] = None
        return

    try:
        result_dicts = [r.parameter_values for r in results]
        constraint_dicts = [
            {
                "type": c.type.value if hasattr(c.type, "value") else str(c.type),
                "parameters": c.parameters,
                "value": c.value,
                "coefficients": c.coefficients if hasattr(c, "coefficients") else None,
            }
            for c in spec.constraints
        ]

        metrics = compute_constraint_satisfaction(result_dicts, constraint_dicts)

        diagnostics["constraint_satisfaction"] = {
            "satisfaction_rate": metrics.satisfaction_rate,
            "recent_satisfaction_rate": metrics.recent_satisfaction_rate,
            "feasible_count": metrics.feasible_count,
            "infeasible_count": metrics.infeasible_count,
            "trend": metrics.trend,
            "interpretation": _interpret_constraint_satisfaction(metrics),
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Constraint satisfaction computation failed: %s", e)
        diagnostics["constraint_satisfaction"] = None
        diagnostics.setdefault("warnings", []).append(
            f"constraint_satisfaction section failed: {e}"
        )


def _interpret_constraint_satisfaction(metrics: ConstraintSatisfactionMetrics) -> str:
    """Provide agent-friendly interpretation of constraint satisfaction."""
    if metrics.satisfaction_rate >= 0.95:
        return "Excellent constraint satisfaction. Almost all suggestions are feasible."
    if metrics.satisfaction_rate >= 0.8:
        return "Good constraint satisfaction. Most suggestions are feasible."
    if metrics.satisfaction_rate >= 0.5:
        return (
            "Moderate constraint satisfaction. Consider reviewing constraint values "
            "or expanding the feasible region."
        )
    return (
        "Low constraint satisfaction. The constraints may be too restrictive. "
        "Consider relaxing constraints or using a different optimization approach."
    )


CALIBRATION_MIN_ROWS = 2


def compute_outcome_constraint_calibration_metrics(
    results: list[Result],
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Assess calibration of outcome-constraint feasibility predictions.

    Fits the same binary feasibility GPs the suggestion pipeline uses (one
    per outcome constraint) on the observed results and compares predicted
    P(feasible) against realized binary feasibility labels. Surfaces
    calibration_error, Brier score, and expected calibration error per
    constraint under ``outcome_constraint_calibration``; an aggregate
    miscalibration warning is appended to ``diagnostics["warnings"]`` when
    the worst constraint's calibration error exceeds
    :data:`CONSTRAINT_CALIBRATION_WARN_THRESHOLD`.

    Per-constraint row alignment: each constraint is fitted against only the
    results that carry its objective. A historical row that lacks the
    constrained objective is skipped for that constraint (it would otherwise
    misalign ``train_x`` against the objective tensor and crash the whole
    section). Constraints with fewer than
    :data:`CALIBRATION_MIN_ROWS` usable rows emit a structured marker
    instead of a numeric report so agents can react ("collect more data
    before trusting this constraint") rather than seeing a generic
    "section failed" warning.

    Mutates ``diagnostics`` in place. Sets
    ``outcome_constraint_calibration`` to ``None`` when there are no
    outcome constraints to assess.
    """
    if not spec.outcome_constraints:
        diagnostics["outcome_constraint_calibration"] = None
        return

    if not results:
        diagnostics["outcome_constraint_calibration"] = None
        return

    try:
        opt_spec = campaign_spec_to_optimization_spec(spec)
        bounds = get_bounds_tensor(opt_spec)
        reports: list[dict[str, Any]] = [
            _per_constraint_report(oc, results, opt_spec, bounds) for oc in spec.outcome_constraints
        ]
        diagnostics["outcome_constraint_calibration"] = reports
        _emit_calibration_warning(reports, diagnostics)
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Outcome-constraint calibration computation failed: %s", exc)
        diagnostics["outcome_constraint_calibration"] = None
        diagnostics.setdefault("warnings", []).append(
            f"outcome_constraint_calibration section failed: {exc}"
        )


def _per_constraint_report(
    oc: OutcomeConstraint,
    results: list[Result],
    opt_spec: OptimizationSpec,
    bounds: torch.Tensor,
) -> dict[str, Any]:
    """Build one calibration entry for a single outcome constraint.

    Filters ``results`` down to rows that carry the constrained objective
    before encoding ``train_x`` and stacking the objective tensor, so the
    GP sees an aligned (parameter, objective) view even when historical
    data has gaps. Returns a structured marker — never raises — when there
    are too few usable rows to fit a meaningful GP.
    """
    base = {
        "constraint_name": oc.objective_name,
        "threshold": oc.threshold,
        "greater_than": oc.greater_than,
    }
    relevant = [r for r in results if oc.objective_name in r.objective_values]
    if not relevant:
        return {**base, "error": "missing_objective"}
    if len(relevant) < CALIBRATION_MIN_ROWS:
        return {**base, "error": "insufficient_data", "n_rows": len(relevant)}

    train_x = stack_encoded_values([r.parameter_values for r in relevant], opt_spec)
    obj_tensor = torch.tensor(
        [float(r.objective_values[oc.objective_name]) for r in relevant],
        dtype=get_dtype(),
        device=get_device(),
    )
    constraint_spec = EngineOutcomeConstraintSpec(
        name=oc.objective_name,
        bound=oc.threshold,
        constraint_type=">=" if oc.greater_than else "<=",
    )
    per_reports = compute_outcome_constraint_calibration(
        constraint_specs=[constraint_spec],
        train_x=train_x,
        objective_values={oc.objective_name: obj_tensor},
        bounds=bounds,
    )
    if not per_reports:
        return {**base, "error": "missing_objective"}
    return per_reports[0]


def _emit_calibration_warning(
    reports: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    """Append a single aggregate WARN when any constraint is miscalibrated."""
    worst_error: float | None = None
    worst_name: str | None = None
    for report in reports:
        error = report.get("calibration_error")
        if not isinstance(error, (int, float)):
            continue
        if worst_error is None or float(error) > worst_error:
            worst_error = float(error)
            worst_name = str(report.get("constraint_name"))
    if worst_error is None or worst_error <= CONSTRAINT_CALIBRATION_WARN_THRESHOLD:
        return
    diagnostics.setdefault("warnings", []).append(
        f"Outcome-constraint model for '{worst_name}' is miscalibrated "
        f"(calibration_error={worst_error:.3f} > "
        f"{CONSTRAINT_CALIBRATION_WARN_THRESHOLD}). Expect overconfident "
        "feasibility predictions; consider collecting more observations near "
        "the constraint boundary or relaxing the threshold."
    )
