"""Submit results tool wrapper for MCP."""

from typing import Any, Literal

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.server import mcp


@mcp.tool(name="bo_submit_results")
async def submit_results(
    campaign_id: str,
    results: list[ResultSubmissionInput],
    submitted_by: str,
    source: str = "api",
    force: bool = False,
    atomic: bool = True,
    continue_on_error: bool = False,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """Submit experimental results for a campaign.

    Args:
        campaign_id: UUID of the campaign.
        results: List of result payloads with parameter_values and
            objective_values.
        submitted_by: UUID of the user submitting results.
        source: Result source (gui, file_upload, or api).
        force: If True, skip duplicate detection.
        atomic: If True, rollback all results if any fail.
        continue_on_error: If True, continue after errors (non-atomic).
        verbosity: Response verbosity level (minimal, standard, detailed).

    Returns:
        Dictionary with success, result_ids, errors, warnings.
    """
    return await submit_results_operation(
        campaign_id=campaign_id,
        results=results,
        submitted_by=submitted_by,
        source=source,
        force=force,
        atomic=atomic,
        continue_on_error=continue_on_error,
        verbosity=verbosity,
    )
