"""Upload results file tool for MCP."""

import csv
import io
import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.result_upload_parser import parse_prefixed_result_rows
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.submit_results import submit_results

logger = logging.getLogger(__name__)


@mcp.tool()
async def upload_results_file(
    campaign_id: str,
    file_content: str,
    file_format: str = "csv",
    submitted_by: str | None = None,
) -> dict[str, Any]:
    """Upload experimental results from a CSV file.

    CSV format should have columns:
    - param_<name>: Parameter values (e.g., param_temperature, param_pressure)
    - obj_<name>: Objective values (e.g., obj_yield, obj_cost)

    Args:
        campaign_id: UUID of the campaign
        file_content: File content as string (CSV format)
        file_format: File format ("csv" supported)
        submitted_by: UUID of user submitting (optional, defaults to campaign_id)

    Returns:
        Dictionary with:
            - success: Boolean
            - results_created: Number of results saved
            - errors: List of row-level errors
    """
    logger.info(
        "Uploading results file for campaign %s (format=%s, size=%d bytes)",
        campaign_id,
        file_format,
        len(file_content),
    )

    if file_format != "csv":
        logger.warning("Unsupported file format: %s", file_format)
        return {
            "success": False,
            "results_created": 0,
            "errors": [f"Unsupported format: {file_format}. Only 'csv' supported."],
        }

    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return {
            "success": False,
            "results_created": 0,
            "errors": ["Invalid campaign_id format"],
        }

    # Parse submitted_by
    submitter_uuid = campaign_uuid
    if submitted_by:
        try:
            submitter_uuid = UUID(submitted_by)
        except ValueError:
            return {
                "success": False,
                "results_created": 0,
                "errors": ["Invalid submitted_by format"],
            }

    # Parse CSV
    try:
        reader = csv.DictReader(io.StringIO(file_content))
    except Exception as e:
        return {
            "success": False,
            "results_created": 0,
            "errors": [f"Failed to parse CSV: {e}"],
        }

    parsed_results, parse_errors = parse_prefixed_result_rows(
        reader,
        metadata_factory=lambda row_num: {"source_row": row_num},
    )

    if not parsed_results:
        return {
            "success": False,
            "results_created": 0,
            "errors": parse_errors or ["No valid results found in uploaded file"],
        }

    submit_result = await submit_results(
        campaign_id=campaign_id,
        results=parsed_results,
        submitted_by=str(submitter_uuid),
        source="file_upload",
        atomic=False,
        continue_on_error=True,
    )

    result_ids = submit_result.get("result_ids", [])
    errors = parse_errors + submit_result.get("errors", [])
    warnings = submit_result.get("warnings", [])
    duplicates_detected = submit_result.get("duplicates_detected", [])

    if errors:
        logger.warning(
            "File upload completed with errors: %d results created, %d errors",
            len(result_ids),
            len(errors),
        )
    else:
        logger.info(
            "File upload successful: %d results created for campaign %s",
            len(result_ids),
            campaign_id,
        )

    response: dict[str, Any] = {
        "success": submit_result.get("success", False) and not parse_errors,
        "results_created": len(result_ids),
        "errors": errors,
    }
    if warnings:
        response["warnings"] = warnings
    if duplicates_detected:
        response["duplicates_detected"] = duplicates_detected

    return response
