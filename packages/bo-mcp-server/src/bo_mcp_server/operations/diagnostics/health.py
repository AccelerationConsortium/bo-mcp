"""Health status and progress computation for campaigns."""

import logging
import math
from typing import Any

from bo_engine.constants import MIN_IMPROVEMENT_RATE
from bo_engine.diagnostics import (
    determine_health_status,
    determine_progress_status,
    determine_single_objective_health_status,
)

from bo_mcp_server.constants import (
    FALLBACK_HYPERVOLUME_IMPROVEMENT,
    HYPERVOLUME_STABILITY_THRESHOLD,
)
from bo_mcp_server.domain import CampaignSpec, Result

logger = logging.getLogger(__name__)


def compute_health_and_progress(
    _spec: CampaignSpec,
    results: list[Result],
    _campaign_iteration: int,
    is_single_objective: bool,
    model_correlation: float,
    hypervolume_history: list[float],
    diagnostics: dict[str, Any],
) -> None:
    """Compute health status and progress status."""
    if is_single_objective:
        health_status, warnings, progress_status = _single_obj_health(
            diagnostics,
            model_correlation,
        )
    else:
        health_status, warnings, progress_status = _multi_obj_health(
            results,
            diagnostics,
            model_correlation,
            hypervolume_history,
        )

    diagnostics["health_status"] = health_status
    diagnostics["warnings"] = warnings
    diagnostics["progress_status"] = progress_status
    diagnostics["hypervolume_history"] = hypervolume_history


def _single_obj_health(
    diagnostics: dict[str, Any],
    model_correlation: float,
) -> tuple[str, list[str], str]:
    """Compute health/progress for single-objective campaigns."""
    improvement_history = diagnostics.get("improvement_history", [])
    health_status, warnings = determine_single_objective_health_status(
        improvement_history=improvement_history,
        model_correlation=model_correlation,
    )
    improvement_rate = diagnostics.get("improvement_rate", 0)
    progress_status = "improving" if improvement_rate > MIN_IMPROVEMENT_RATE else "stable"
    return health_status, warnings, progress_status


def _multi_obj_health(
    results: list[Result],
    diagnostics: dict[str, Any],
    model_correlation: float,
    hypervolume_history: list[float],
) -> tuple[str, list[str], str]:
    """Compute health/progress for multi-objective campaigns."""
    hv_improvement, iters_stagnant = _analyze_hypervolume_history(
        hypervolume_history,
        results,
        diagnostics,
    )

    health_status, warnings = determine_health_status(
        n_results=len(results),
        hypervolume_improvement=hv_improvement,
        model_correlation=model_correlation,
        iterations_without_improvement=iters_stagnant,
    )

    hv_hist = hypervolume_history if hypervolume_history else [diagnostics.get("hypervolume", 0)]
    progress_status = determine_progress_status(hv_hist)
    return health_status, warnings, progress_status


def _analyze_hypervolume_history(
    hypervolume_history: list[float],
    results: list[Result],
    diagnostics: dict[str, Any],
) -> tuple[float, int]:
    """Analyze hypervolume history for improvement and stagnation."""
    if len(hypervolume_history) >= 2:
        recent_improvement = hypervolume_history[-1] - hypervolume_history[-2]
        hv_improvement = max(0.0, recent_improvement)

        threshold = (
            HYPERVOLUME_STABILITY_THRESHOLD * abs(hypervolume_history[-1])
            if not math.isclose(hypervolume_history[-1], 0.0, abs_tol=1e-12)
            else HYPERVOLUME_STABILITY_THRESHOLD
        )
        iters_stagnant = 0
        for i in range(len(hypervolume_history) - 1, 0, -1):
            if hypervolume_history[i] - hypervolume_history[i - 1] < threshold:
                iters_stagnant += 1
            else:
                break
        return hv_improvement, iters_stagnant

    if len(results) >= 2:
        has_hv = diagnostics.get("hypervolume", 0) > 0
        hv_improvement = FALLBACK_HYPERVOLUME_IMPROVEMENT if has_hv else 0.0
        return hv_improvement, 0

    return 0.0, 0
