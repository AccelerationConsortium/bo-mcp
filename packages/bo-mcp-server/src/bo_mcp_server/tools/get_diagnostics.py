"""Get diagnostics tool wrapper for MCP."""

from typing import Literal, cast

from mcp.server.fastmcp import Context

from bo_mcp_server.operations.get_diagnostics import (
    ALL_SECTIONS,
    get_diagnostics_operation,
)
from bo_mcp_server.progress_bridge import make_progress_callback_from_context
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.common import VerbosityLiteral
from bo_mcp_server.tools.response_models import GetDiagnosticsResponse

# Re-export for backward compatibility
__all__ = ["ALL_SECTIONS", "get_diagnostics"]

# Mirrors ``ALL_SECTIONS`` so the generated MCP tool schema declares an
# ``enum`` constraint on each list item. Keep aligned with
# :data:`bo_mcp_server.operations.get_diagnostics.ALL_SECTIONS`.
SectionLiteral = Literal[
    "health",
    "objectives",
    "model",
    "convergence",
    "suggestions",
    "outliers",
    "constraints",
]


@mcp.tool(name="bo_get_diagnostics", annotations=READ_ONLY)
async def get_diagnostics(
    campaign_id: str,
    use_cache: bool = True,
    verbosity: VerbosityLiteral = "standard",
    sections: list[SectionLiteral] | None = None,
    ctx: Context | None = None,
) -> GetDiagnosticsResponse:
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
    # ``list`` is invariant, so ``list[SectionLiteral]`` does not
    # structurally satisfy the operation's ``list[str]`` parameter even
    # though every ``SectionLiteral`` value is a ``str`` at runtime; the
    # explicit annotation here is the widening, not a behavior change.
    sections_str: list[str] | None = list(sections) if sections is not None else None
    return cast(
        GetDiagnosticsResponse,
        await get_diagnostics_operation(
            campaign_id=campaign_id,
            use_cache=use_cache,
            verbosity=verbosity,
            sections=sections_str,
            progress_callback=make_progress_callback_from_context(ctx),
        ),
    )
