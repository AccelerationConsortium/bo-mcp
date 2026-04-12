"""Upload results file tool for MCP."""

import csv
import io
import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.result_upload_parser import parse_prefixed_result_rows
from bo_mcp_server.server import mcp

logger = logging.getLogger(__name__)

# 10 MB — prevents memory exhaustion from arbitrarily large uploads.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024


def _parse_uuids(campaign_id: str, submitted_by: str | None) -> dict[str, Any] | tuple[UUID, UUID]:
    """Validate and parse campaign_id and submitted_by UUIDs.

    Returns an error response dict on failure, or (campaign_uuid, submitter_uuid) on success.
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    submitter_uuid = campaign_uuid
    if submitted_by:
        try:
            submitter_uuid = UUID(submitted_by)
        except ValueError:
            return make_error_response(
                ErrorCode.VALIDATION_FAILED,
                message="Invalid submitted_by format",
                details={"submitted_by": submitted_by},
            )

    return campaign_uuid, submitter_uuid


@mcp.tool(name="bo_upload_results_file")
async def upload_results_file(
    campaign_id: str,
    file_content: str,
    file_format: str = "csv",
    submitted_by: str | None = None,
) -> dict[str, Any]:
    """Upload experimental results from a CSV file.

    Workflow: Alternative to bo_submit_results for bulk uploads. Use when
    you have historical data in CSV format.

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

    if len(file_content) > MAX_UPLOAD_SIZE_BYTES:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"File too large ({len(file_content)} bytes). "
                f"Maximum upload size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            ),
            details={
                "size_bytes": len(file_content),
                "max_bytes": MAX_UPLOAD_SIZE_BYTES,
            },
        )

    if file_format != "csv":
        logger.warning("Unsupported file format: %s", file_format)
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unsupported format: {file_format}. Only 'csv' supported.",
            details={"file_format": file_format},
        )

    uuid_result = _parse_uuids(campaign_id, submitted_by)
    if isinstance(uuid_result, dict):
        return uuid_result
    _, submitter_uuid = uuid_result

    # Parse CSV
    try:
        reader = csv.DictReader(io.StringIO(file_content))
    except csv.Error as e:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Failed to parse CSV: {e}",
        )

    parsed_results, parse_errors = parse_prefixed_result_rows(
        reader,
        metadata_factory=lambda row_num: {"source_row": row_num},
    )

    if not parsed_results:
        response = make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message="No valid results found in uploaded file",
            details={"parse_errors": parse_errors} if parse_errors else None,
        )
        if parse_errors:
            response["errors"] = parse_errors
        return response

    submit_result = await submit_results_operation(
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
