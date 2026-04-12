"""Convergence detection diagnostics."""

import logging
from typing import Any

from bo_engine.convergence import (
    detect_hypervolume_convergence,
    detect_single_objective_convergence,
)

from bo_mcp_server.domain import CampaignSpec

logger = logging.getLogger(__name__)


def compute_convergence_diagnostics(
    spec: CampaignSpec,
    is_single_objective: bool,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence/early stopping detection metrics."""
    try:
        if is_single_objective:
            _compute_single_obj_convergence(spec, diagnostics)
        else:
            _compute_multi_obj_convergence(hypervolume_history, diagnostics)
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Convergence detection failed: %s", e)
        diagnostics["convergence"] = None
        diagnostics.setdefault("warnings", []).append(f"convergence section failed: {e}")


def _compute_single_obj_convergence(
    spec: CampaignSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence for single-objective campaigns."""
    improvement_history = diagnostics.get("improvement_history", [])
    if len(improvement_history) >= 5:
        obj = spec.objectives[0]
        report = detect_single_objective_convergence(
            best_value_history=improvement_history,
            minimize=obj.is_minimize,
        )
        diagnostics["convergence"] = {
            "converged": report.converged,
            "convergence_score": round(report.convergence_score, 4),
            "reason": report.reason,
            "avg_improvement": round(report.avg_improvement, 6),
            "iterations_without_improvement": report.iterations_without_improvement,
            "recommendation": report.recommendation,
        }
    else:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": "Insufficient data for convergence detection",
            "recommendation": "Continue optimization to gather more data.",
        }


def _compute_multi_obj_convergence(
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute convergence for multi-objective campaigns."""
    hv_history = hypervolume_history if hypervolume_history else []
    if len(hv_history) >= 5:
        report = detect_hypervolume_convergence(hv_history)
        diagnostics["convergence"] = {
            "converged": report.converged,
            "convergence_score": round(report.convergence_score, 4),
            "reason": report.reason,
            "avg_improvement": round(report.avg_improvement, 6),
            "iterations_without_improvement": report.iterations_without_improvement,
            "recommendation": report.recommendation,
        }
    elif hv_history:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": f"Insufficient history ({len(hv_history)} points, need 5)",
            "recommendation": (
                "Continue optimization to gather more data for convergence detection."
            ),
        }
    else:
        diagnostics["convergence"] = {
            "converged": False,
            "convergence_score": 0.0,
            "reason": "No hypervolume history available",
            "recommendation": (
                "Submit results to build hypervolume history for convergence detection."
            ),
        }
