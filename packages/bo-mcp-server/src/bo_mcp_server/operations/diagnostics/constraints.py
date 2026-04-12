"""Constraint satisfaction diagnostics."""

import logging
from typing import Any

from bo_engine.diagnostics import compute_constraint_satisfaction

from bo_mcp_server.domain import CampaignSpec, Result

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


def _interpret_constraint_satisfaction(metrics: Any) -> str:
    """Provide agent-friendly interpretation of constraint satisfaction."""
    if metrics.satisfaction_rate >= 0.95:
        return "Excellent constraint satisfaction. Almost all suggestions are feasible."
    elif metrics.satisfaction_rate >= 0.8:
        return "Good constraint satisfaction. Most suggestions are feasible."
    elif metrics.satisfaction_rate >= 0.5:
        return (
            "Moderate constraint satisfaction. Consider reviewing constraint values "
            "or expanding the feasible region."
        )
    else:
        return (
            "Low constraint satisfaction. The constraints may be too restrictive. "
            "Consider relaxing constraints or using a different optimization approach."
        )
