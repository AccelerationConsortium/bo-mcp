"""Transfer candidate discovery tool wrapper for MCP."""

from typing import cast

from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import READ_ONLY
from bo_mcp_server.tools.common import VerbosityLiteral
from bo_mcp_server.tools.response_models import DiscoverTransferCandidatesResponse


@mcp.tool(name="bo_discover_transfer_candidates", annotations=READ_ONLY)
async def discover_transfer_candidates(
    campaign_id: str,
    similarity_threshold: float = 0.5,
    max_candidates: int = 5,
    verbosity: VerbosityLiteral = "standard",
    parameter_aliases: dict[str, list[str]] | None = None,
) -> DiscoverTransferCandidatesResponse:
    """Discover campaigns suitable for transfer learning.

    Workflow: Call before bo_create_campaign to find prior campaigns whose
    data can accelerate optimization on a new but related problem.

    Similarity is computed syntactically (matching parameter names,
    objective names, and bounds overlap). Supply ``parameter_aliases``
    to bridge naming drift across campaigns — e.g.
    ``{"temperature": ["temp_c", "temp_celsius"]}`` treats all three
    names as the same physical parameter when computing the
    parameter-set and bounds-overlap components. The data richness
    score still scales with the source campaign's dimensionality.

    Args:
        campaign_id: UUID of the target campaign to find transfer candidates for.
        similarity_threshold: Minimum similarity score (0.0-1.0, default 0.5).
        max_candidates: Maximum number of candidates to return (default 5).
        verbosity: Response verbosity level (minimal, standard, detailed).
        parameter_aliases: Optional ``{canonical: [synonym, ...]}`` map
            used to unify parameter names that drift across campaigns.

    Returns:
        Dictionary with:
            - success: Boolean
            - target_campaign: Target campaign info
            - candidates: List of similar campaigns with similarity scores
            - overall_recommendation: Transfer learning recommendation
            - errors: List of error messages
    """
    return cast(
        DiscoverTransferCandidatesResponse,
        await discover_transfer_candidates_operation(
            campaign_id=campaign_id,
            similarity_threshold=similarity_threshold,
            max_candidates=max_candidates,
            verbosity=verbosity,
            parameter_aliases=parameter_aliases,
        ),
    )
