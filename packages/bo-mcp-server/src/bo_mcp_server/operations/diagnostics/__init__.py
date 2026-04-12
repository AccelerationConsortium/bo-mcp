"""Diagnostics helper modules.

Extracts focused concerns from the monolithic get_diagnostics.py into
testable, cohesive modules. The public entry point remains
``operations.get_diagnostics.get_diagnostics_operation``.
"""

from bo_mcp_server.operations.diagnostics.actions import (
    compute_next_action_recommendation,
)
from bo_mcp_server.operations.diagnostics.constraints import (
    compute_constraint_satisfaction_metrics,
)
from bo_mcp_server.operations.diagnostics.convergence import (
    compute_convergence_diagnostics,
)
from bo_mcp_server.operations.diagnostics.enrichment import (
    compute_objective_ranges,
    enrich_diagnostics,
    enrich_outlier_results,
    get_model_info,
)
from bo_mcp_server.operations.diagnostics.health import (
    compute_health_and_progress,
)
from bo_mcp_server.operations.diagnostics.suggestions_analysis import (
    compute_exploration_exploitation,
    compute_suggestion_diversity_metrics,
    compute_uncertainty_trends,
)

__all__ = [
    "compute_constraint_satisfaction_metrics",
    "compute_convergence_diagnostics",
    "compute_exploration_exploitation",
    "compute_health_and_progress",
    "compute_next_action_recommendation",
    "compute_objective_ranges",
    "compute_suggestion_diversity_metrics",
    "compute_uncertainty_trends",
    "enrich_diagnostics",
    "enrich_outlier_results",
    "get_model_info",
]
