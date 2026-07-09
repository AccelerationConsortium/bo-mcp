"""Export campaign operation - protocol-neutral business logic."""

import csv
import io
import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.response_formatter import with_response_metadata
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)

logger = logging.getLogger(__name__)

# Ceiling for inline CSV content on context-bound transports (MCP): a
# multi-hundred-KB string injected into an agent's context is pure token
# waste, so the export truncates at a row boundary and tells the caller
# how to page the remainder. Download-style transports (the REST
# StreamingResponse route) pass ``max_content_bytes=None`` to disable
# the cap.
MAX_EXPORT_CONTENT_BYTES = 256 * 1024

# Conservative floor for one rendered CSV data row, used to convert the
# byte budget into a database-level row LIMIT so a capped export never
# hydrates the whole campaign. Every row carries at least the three
# QUOTE_ALL fixed fields (result_id UUID 38 chars, suggestion_id,
# ISO-8601 created_at ~34 chars) plus separators — ~78 bytes before any
# parameter/objective columns — so 64 keeps the over-fetch bounded at
# ~25% in the worst case while never under-fetching a row that fits.
MIN_EXPORT_ROW_BYTES = 64


@with_response_metadata
async def export_campaign_operation(
    campaign_id: str,
    output_format: str = "csv",
    max_content_bytes: int | None = MAX_EXPORT_CONTENT_BYTES,
) -> dict[str, Any]:
    """Export all results for a campaign as CSV.

    Returns the dataset (parameters + objectives) as a CSV string.
    Transport layers wrap this appropriately (MCP: JSON with content field,
    HTTP: StreamingResponse with Content-Disposition).

    Args:
        campaign_id: UUID string of the campaign.
        output_format: Export format. Currently only "csv" is supported.
        max_content_bytes: Truncate ``content`` at the last full row that
            fits within this many bytes; ``None`` disables the cap. When
            truncation happens the envelope carries ``truncated=True``,
            ``n_results_included`` and a warning advising row-level
            pagination via ``bo_list_results``.

    Returns:
        Dictionary with success, format, content, campaign_name,
        n_results, n_results_included, truncated, errors (plus
        ``warnings`` when truncated).
    """
    logger.info("Exporting campaign %s as %s", campaign_id, output_format)

    if output_format != "csv":
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unsupported export format: {output_format}. Only 'csv' supported.",
            details={"format": output_format},
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
        n_total = (await result_repo.count_by_campaigns([campaign_uuid])).get(campaign_uuid, 0)
        # Bound the read at the database: the byte budget can hold at
        # most budget / MIN_EXPORT_ROW_BYTES rows, so a capped export
        # never hydrates (or sorts) the whole campaign.
        fetch_limit = (
            None if max_content_bytes is None else max_content_bytes // MIN_EXPORT_ROW_BYTES + 1
        )
        results = await result_repo.list_by_campaign(campaign_uuid, limit=fetch_limit)

    campaign_name = spec.name if spec else f"campaign_{campaign_id[:8]}"

    if not results:
        return {
            "success": True,
            "format": "csv",
            "content": "",
            "campaign_name": campaign_name,
            "n_results": n_total,
            "n_results_included": 0,
            "truncated": n_total > 0,
            "errors": [],
        }

    # Derive column names from spec (if available) or first result
    param_names = (
        [p.name for p in spec.parameters] if spec else sorted(results[0].parameter_values.keys())
    )
    obj_names = (
        [o.name for o in spec.objectives] if spec else sorted(results[0].objective_values.keys())
    )

    csv_content, n_included = _render_csv(results, param_names, obj_names, max_content_bytes)
    truncated = n_included < n_total
    logger.info("Exported %d/%d results for campaign %s", n_included, n_total, campaign_id)

    response: dict[str, Any] = {
        "success": True,
        "format": "csv",
        "content": csv_content,
        "campaign_name": campaign_name,
        "n_results": n_total,
        "n_results_included": n_included,
        "truncated": truncated,
        "errors": [],
    }
    if truncated and n_included == 0:
        response["warnings"] = [
            f"Export content omitted: the CSV header alone exceeds "
            f"{max_content_bytes} bytes of inline content for this spec. "
            "Use bo_list_results with cursor pagination, or the REST "
            "export route for a full download."
        ]
    elif truncated:
        response["warnings"] = [
            f"Export truncated to the first {n_included} of {n_total} "
            f"results to stay within {max_content_bytes} bytes of inline "
            "content. Use bo_list_results with cursor pagination to read "
            "the remaining rows, or the REST export route for a full download."
        ]
    return response


def _render_csv(
    results: list[Any],
    param_names: list[str],
    obj_names: list[str],
    max_content_bytes: int | None,
) -> tuple[str, int]:
    """Render results as CSV, stopping at the last full row within the cap.

    Returns the CSV string and the number of result rows it contains.
    Rows are appended until the next row would push the payload past
    ``max_content_bytes`` (``None`` = no cap). When even the header
    exceeds the budget (a pathologically wide spec, or a tiny cap),
    the render returns empty content instead of silently violating the
    byte contract — the caller's truncation envelope points at the
    paginated tool.
    """
    output = io.StringIO()
    fieldnames = (
        [f"param_{n}" for n in param_names]
        + [f"obj_{n}" for n in obj_names]
        + ["result_id", "suggestion_id", "created_at"]
    )
    # QUOTE_ALL ensures field names containing commas or quotes are safe.
    writer = csv.DictWriter(output, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    if max_content_bytes is not None and output.tell() > max_content_bytes:
        return "", 0

    n_included = 0
    kept_length = output.tell()
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
        if max_content_bytes is not None and output.tell() > max_content_bytes:
            break
        n_included += 1
        kept_length = output.tell()

    return output.getvalue()[:kept_length], n_included
