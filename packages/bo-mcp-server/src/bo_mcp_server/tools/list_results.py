"""List results and export campaign tools for MCP."""

import csv
import io
import logging
from typing import Any, Literal
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

MAX_RESULTS_LIMIT = 500


@mcp.tool(name="bo_list_results")
async def list_results(
    campaign_id: str,
    limit: int = 50,
    offset: int = 0,
    verbosity: Literal["minimal", "standard", "detailed"] = "standard",
) -> dict[str, Any]:
    """List experimental results for a campaign.

    Returns structured result data for review, audit, or correction workflows.

    Args:
        campaign_id: UUID of the campaign.
        limit: Maximum number of results to return (default 50, max 500).
        offset: Number of results to skip for pagination (default 0).
        verbosity: Response verbosity level. Options:
            - "minimal": result_id, objective_values only
            - "standard": includes parameter_values, suggestion_id, created_at
            - "detailed": includes metadata, measurement_uncertainty, source

    Returns:
        Dictionary with:
            - success: Boolean
            - results: List of result dictionaries
            - total_count: Total results for this campaign
            - errors: List of error messages
    """
    logger.info(
        "Listing results: campaign_id=%s, limit=%d, offset=%d",
        campaign_id,
        limit,
        offset,
    )

    try:
        verbosity_level = VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        )

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    limit = max(1, min(limit, MAX_RESULTS_LIMIT))
    offset = max(0, offset)

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        result_repo = ResultRepository(session)
        all_results = await result_repo.list_by_campaign(campaign_uuid)

    # Sort by created_at descending (most recent first)
    all_results.sort(key=lambda r: r.created_at, reverse=True)
    total_count = len(all_results)
    page = all_results[offset : offset + limit]

    results_out: list[dict[str, Any]] = []
    for r in page:
        if verbosity_level == VerbosityLevel.MINIMAL:
            results_out.append(
                {
                    "result_id": str(r.id),
                    "objective_values": r.objective_values,
                }
            )
        elif verbosity_level == VerbosityLevel.STANDARD:
            results_out.append(
                {
                    "result_id": str(r.id),
                    "parameter_values": r.parameter_values,
                    "objective_values": r.objective_values,
                    "suggestion_id": str(r.suggestion_id) if r.suggestion_id else None,
                    "created_at": r.created_at.isoformat(),
                }
            )
        else:
            results_out.append(
                {
                    "result_id": str(r.id),
                    "parameter_values": r.parameter_values,
                    "objective_values": r.objective_values,
                    "suggestion_id": str(r.suggestion_id) if r.suggestion_id else None,
                    "source": r.source.value,
                    "submitted_by": str(r.submitted_by),
                    "measurement_uncertainty": r.measurement_uncertainty,
                    "metadata": r.metadata,
                    "created_at": r.created_at.isoformat(),
                }
            )

    return {
        "success": True,
        "results": results_out,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "errors": [],
    }


@mcp.tool(name="bo_export_campaign")
async def export_campaign(
    campaign_id: str,
    format: str = "csv",
) -> dict[str, Any]:
    """Export all results for a campaign as CSV.

    Returns the full dataset (parameters + objectives) in a format suitable
    for downstream analysis or archival.

    Args:
        campaign_id: UUID of the campaign.
        format: Export format. Currently only "csv" is supported.

    Returns:
        Dictionary with:
            - success: Boolean
            - format: The export format used
            - content: CSV string with all results
            - n_results: Number of results exported
            - errors: List of error messages
    """
    logger.info("Exporting campaign %s as %s", campaign_id, format)

    if format != "csv":
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unsupported export format: {format}. Only 'csv' supported.",
            details={"format": format},
        )

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return make_error_response(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(campaign.spec_id)

        result_repo = ResultRepository(session)
        results = await result_repo.list_by_campaign(campaign_uuid)

    if not results:
        return {
            "success": True,
            "format": "csv",
            "content": "",
            "n_results": 0,
            "errors": [],
        }

    # Derive column names from spec (if available) or first result
    param_names = (
        [p.name for p in spec.parameters] if spec else sorted(results[0].parameter_values.keys())
    )
    obj_names = (
        [o.name for o in spec.objectives] if spec else sorted(results[0].objective_values.keys())
    )

    # Sort by creation time
    results.sort(key=lambda r: r.created_at)

    output = io.StringIO()
    fieldnames = (
        [f"param_{n}" for n in param_names]
        + [f"obj_{n}" for n in obj_names]
        + ["result_id", "suggestion_id", "created_at"]
    )
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        row: dict[str, Any] = {}
        for name in param_names:
            row[f"param_{name}"] = r.parameter_values.get(name)
        for name in obj_names:
            row[f"obj_{name}"] = r.objective_values.get(name)
        row["result_id"] = str(r.id)
        row["suggestion_id"] = str(r.suggestion_id) if r.suggestion_id else ""
        row["created_at"] = r.created_at.isoformat()
        writer.writerow(row)

    csv_content = output.getvalue()
    logger.info("Exported %d results for campaign %s", len(results), campaign_id)

    return {
        "success": True,
        "format": "csv",
        "content": csv_content,
        "n_results": len(results),
        "errors": [],
    }
