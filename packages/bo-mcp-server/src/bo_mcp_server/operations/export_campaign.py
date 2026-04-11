"""Export campaign operation - protocol-neutral business logic."""

import csv
import io
import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)


async def export_campaign_operation(
    campaign_id: str,
    format: str = "csv",
) -> dict[str, Any]:
    """Export all results for a campaign as CSV.

    Returns the full dataset (parameters + objectives) as a CSV string.
    Transport layers wrap this appropriately (MCP: JSON with content field,
    HTTP: StreamingResponse with Content-Disposition).

    Args:
        campaign_id: UUID string of the campaign.
        format: Export format. Currently only "csv" is supported.

    Returns:
        Dictionary with success, format, content, n_results, errors.
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
