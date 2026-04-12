"""Suggestion-based diagnostics: uncertainty trends, exploration/exploitation, diversity."""

import logging
from typing import Any

import torch
from bo_engine.diagnostics import (
    compute_exploration_exploitation_metrics,
    compute_suggestion_diversity,
    compute_uncertainty_trend,
)
from bo_engine.transforms import encode_categorical
from bo_engine.types import OptimizationSpec

from bo_mcp_server.constants import DIVERSITY_HIGH_THRESHOLD, DIVERSITY_MODERATE_THRESHOLD
from bo_mcp_server.domain import CampaignSpec, Result, Suggestion, SuggestionStatus

logger = logging.getLogger(__name__)


def compute_uncertainty_trends(
    suggestions: list[Suggestion],
    diagnostics: dict[str, Any],
) -> None:
    """Compute uncertainty trends from suggestion history."""
    uncertainty_history = []
    for sugg in suggestions:
        if sugg.provenance.model_uncertainty is not None:
            uncertainty_history.append(sugg.provenance.model_uncertainty)

    if not uncertainty_history:
        diagnostics["uncertainty_trend"] = None
        return

    trend = compute_uncertainty_trend(uncertainty_history)
    diagnostics["uncertainty_trend"] = {
        "mean_uncertainty": round(trend.mean_uncertainty, 4),
        "std_uncertainty": round(trend.std_uncertainty, 4),
        "trend": trend.trend,
        "slope": round(trend.slope, 6),
        "interpretation": _interpret_uncertainty_trend(trend.trend),
    }


def _interpret_uncertainty_trend(trend: str) -> str:
    """Provide agent-friendly interpretation of uncertainty trend."""
    if trend == "decreasing":
        return (
            "Model uncertainty is decreasing. The model is becoming more confident "
            "as it learns from observations. This is expected behavior."
        )
    elif trend == "increasing":
        return (
            "Model uncertainty is increasing. This may indicate the optimization is "
            "exploring new regions, or the model is struggling to fit the data."
        )
    else:
        return (
            "Model uncertainty is stable. The model maintains consistent confidence "
            "across recent suggestions."
        )


def compute_exploration_exploitation(
    suggestions: list[Suggestion],
    results: list[Result],
    spec: CampaignSpec,
    opt_spec: OptimizationSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute exploration/exploitation balance metrics."""
    if len(suggestions) < 2:
        diagnostics["exploration_exploitation"] = None
        return

    try:
        pending = [s for s in suggestions if s.status == SuggestionStatus.PENDING]
        if not pending:
            pending = suggestions[-5:]

        sugg_list = []
        uncertainties = []
        for s in pending:
            x_encoded = encode_categorical(s.parameter_values, opt_spec)
            sugg_list.append(x_encoded)
            if s.provenance.model_uncertainty is not None:
                uncertainties.append(s.provenance.model_uncertainty)

        if not sugg_list:
            diagnostics["exploration_exploitation"] = None
            return

        from bo_engine.diagnostics import compute_best_value
        from bo_engine.transforms import get_bounds_tensor

        suggestions_tensor = torch.stack(sugg_list)

        best_point = None
        if results:
            obj = spec.objectives[0]
            values = [r.objective_values[obj.name] for r in results]
            if values:
                _, best_idx = compute_best_value(values, minimize=obj.is_minimize)
                if best_idx >= 0:
                    best_encoded = encode_categorical(results[best_idx].parameter_values, opt_spec)
                    best_point = best_encoded

        bounds = get_bounds_tensor(opt_spec)

        metrics = compute_exploration_exploitation_metrics(
            suggestions=suggestions_tensor,
            best_point=best_point,
            uncertainties=uncertainties,
            bounds=bounds,
        )

        diagnostics["exploration_exploitation"] = {
            "exploration_ratio": round(metrics.exploration_ratio, 4),
            "diversity_score": round(metrics.diversity_score, 4),
            "average_distance_to_best": round(metrics.average_distance_to_best, 4),
            "balance_assessment": metrics.balance_assessment,
            "recommendation": metrics.recommendation,
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Exploration/exploitation metrics computation failed: %s", e)
        diagnostics["exploration_exploitation"] = None
        diagnostics.setdefault("warnings", []).append(
            f"exploration_exploitation section failed: {e}"
        )


def compute_suggestion_diversity_metrics(
    suggestions: list[Suggestion],
    opt_spec: OptimizationSpec,
    diagnostics: dict[str, Any],
) -> None:
    """Compute diversity metrics for recent suggestions."""
    pending = [s for s in suggestions if s.status == SuggestionStatus.PENDING]
    if len(pending) < 2:
        diagnostics["suggestion_diversity"] = None
        return

    try:
        sugg_list = []
        for s in pending:
            x_encoded = encode_categorical(s.parameter_values, opt_spec)
            sugg_list.append(x_encoded)

        if len(sugg_list) < 2:
            diagnostics["suggestion_diversity"] = None
            return

        suggestions_tensor = torch.stack(sugg_list)
        diversity = compute_suggestion_diversity(suggestions_tensor)

        diagnostics["suggestion_diversity"] = {
            "diversity_score": round(diversity, 4),
            "n_suggestions": len(sugg_list),
            "interpretation": _interpret_diversity(diversity),
        }
    except (RuntimeError, ValueError, TypeError) as e:
        logger.warning("Suggestion diversity computation failed: %s", e)
        diagnostics["suggestion_diversity"] = None
        diagnostics.setdefault("warnings", []).append(f"suggestion_diversity section failed: {e}")


def _interpret_diversity(diversity: float) -> str:
    """Provide agent-friendly interpretation of diversity score."""
    if diversity >= DIVERSITY_HIGH_THRESHOLD:
        return "High diversity. Suggestions explore diverse regions of the parameter space."
    elif diversity >= DIVERSITY_MODERATE_THRESHOLD:
        return "Moderate diversity. Suggestions cover a reasonable range of the parameter space."
    else:
        return (
            "Low diversity. Suggestions are clustered together, which may indicate "
            "convergence or over-exploitation. Consider increasing exploration if stuck."
        )
