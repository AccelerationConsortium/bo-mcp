"""Transfer candidate discovery tool wrapper for MCP."""

from typing import Any

from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_discover_transfer_candidates")
async def discover_transfer_candidates(
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: str = "standard",
) -> dict[str, Any]:
    """Discover campaigns suitable for transfer learning.

    Workflow: Call before bo_create_campaign to find prior campaigns whose
    data can accelerate optimization on a new but related problem.

    Note: Similarity is computed syntactically (matching parameter names,
    objective names, and bounds overlap). Parameters with different names
    but the same physical meaning (e.g. "temperature" vs "temp_celsius")
    are treated as unrelated. The data richness score scales with the
    source campaign's dimensionality.

    Args:
        campaign_id: UUID of the target campaign to find transfer candidates for.
        similarity_threshold: Minimum similarity score (0.0-1.0, default 0.5).
        max_candidates: Maximum number of candidates to return (default 5).
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with:
            - success: Boolean
            - target_campaign: Target campaign info
            - candidates: List of similar campaigns with similarity scores
            - overall_recommendation: Transfer learning recommendation
            - errors: List of error messages
    """
    return await discover_transfer_candidates_operation(
        campaign_id=campaign_id,
        similarity_threshold=similarity_threshold,
        max_candidates=max_candidates,
        verbosity=verbosity,
    )
