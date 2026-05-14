"""Health status and progress computation for campaigns.

The status/threshold logic lives in :func:`bo_engine.diagnostics.compute_campaign_health`;
this module is a thin adapter that runs the engine helper and writes the
result back into the transport-shaped ``diagnostics`` dict.
"""

import logging
from typing import Any

from bo_engine.diagnostics import compute_campaign_health

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
    health_status, warnings, progress_status = compute_campaign_health(
        is_single_objective=is_single_objective,
        n_results=len(results),
        diagnostics=diagnostics,
        model_correlation=model_correlation,
        hypervolume_history=hypervolume_history,
    )

    diagnostics["health_status"] = health_status
    diagnostics["warnings"] = warnings
    diagnostics["progress_status"] = progress_status
    diagnostics["hypervolume_history"] = hypervolume_history
