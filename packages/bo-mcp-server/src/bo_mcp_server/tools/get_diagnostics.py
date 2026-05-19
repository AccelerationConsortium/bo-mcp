"""Get diagnostics tool wrapper for MCP."""

from typing import Any

from mcp.server.fastmcp import Context

from bo_mcp_server.operations.get_diagnostics import (
    ALL_SECTIONS,
    get_diagnostics_operation,
)
from bo_mcp_server.progress_bridge import make_progress_callback_from_context
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY

# Re-export for backward compatibility
__all__ = ["ALL_SECTIONS", "get_diagnostics"]


@mcp.tool(name="bo_get_diagnostics", annotations=READ_ONLY)
async def get_diagnostics(
    campaign_id: str,
    use_cache: bool = True,
    verbosity: str = "standard",
    sections: list[str] | None = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Get diagnostic information for a campaign.

    Workflow: Call after bo_submit_results to check model health, convergence,
    and get a next_action recommendation. Use sections=["health"] for fast
    status checks in tight loops.

    Args:
        campaign_id: UUID of the campaign
        use_cache: Whether to use cached results (default True).
        verbosity: Response verbosity level (minimal, standard, detailed).
        sections: Optional list of sections to compute. When omitted, all
            sections are computed. Valid values:
            - "health": health_status, progress_status, next_action
            - "objectives": best_value/pareto_front, objective_ranges
            - "model": feature_importance, loo_cv_metrics, correlation
            - "convergence": convergence detection
            - "suggestions": uncertainty_trend, exploration, diversity
            - "outliers": outlier detection
            - "constraints": constraint satisfaction tracking
            Use ["health"] for fast status checks.
        ctx: FastMCP request context used for progress notifications.
            Supplied automatically by the MCP runtime.

    Returns:
        Dictionary with diagnostic fields based on requested sections.
    """
    return await get_diagnostics_operation(
        campaign_id=campaign_id,
        use_cache=use_cache,
        verbosity=verbosity,
        sections=sections,
        progress_callback=make_progress_callback_from_context(ctx),
    )
