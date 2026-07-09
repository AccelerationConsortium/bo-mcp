"""Upload results file tool for MCP."""

import asyncio
import csv
import io
import logging
from typing import Any, cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.client import AuthenticationConfigurationError, resolve_mcp_user
from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.idempotency import apply_idempotency, digest_large_field
from bo_mcp_server.operations.submit_results import submit_results_operation
from bo_mcp_server.response_formatter import attach_response_metadata, with_response_metadata
from bo_mcp_server.result_upload_parser import parse_prefixed_result_rows
from bo_mcp_server.server import mcp
from bo_mcp_server.tools.annotations import NON_IDEMPOTENT_MUTATION
from bo_mcp_server.tools.common import mcp_identity_error
from bo_mcp_server.tools.response_models import UploadResultsResponse
from bo_mcp_server.trace_context import bind_trace_id

logger = logging.getLogger(__name__)

# 10 MB — prevents memory exhaustion from arbitrarily large uploads.
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024

# (parsed rows, per-row parse errors) — the successful outcome of
# ``_validate_upload_payload``, threaded through the pipeline so the
# CSV is parsed exactly once per call.
_ParsedUpload = tuple[list[ResultSubmissionInput], list[str]]


def _build_upload_response(
    submit_result: dict[str, Any],
    parse_errors: list[str],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    """Compose the upload-results-file response envelope.

    Splits the assembly out of the main pipeline so the inner runner
    stays under ruff's cognitive-complexity ceiling. Carries the
    inner submit-results preview verbatim when running in dry-run
    mode so callers see both the parse-side and the submit-side view
    of "what would happen".
    """
    result_ids = submit_result.get("result_ids", [])
    errors = parse_errors + submit_result.get("errors", [])
    warnings = submit_result.get("warnings", [])
    duplicates_detected = submit_result.get("duplicates_detected", [])
    response: dict[str, Any] = {
        "success": submit_result.get("success", False) and not parse_errors,
        "results_created": len(result_ids),
        "errors": errors,
    }
    if warnings:
        response["warnings"] = warnings
    if duplicates_detected:
        response["duplicates_detected"] = duplicates_detected
    if dry_run:
        response["dry_run"] = True
        if "preview" in submit_result:
            response["preview"] = submit_result["preview"]
    # Inherit the inner ``submit_results`` envelope's metadata so the
    # outer upload response also carries ``_metadata.trace_id`` /
    # backend / server_version. The tool wrapper itself runs inside
    # ``bind_trace_id``; ``_build_upload_response`` is the only place
    # that rebuilds the envelope, so the inheritance has to happen
    # here to survive the rebuild.
    metadata = submit_result.get("_metadata")
    if isinstance(metadata, dict):
        response["_metadata"] = metadata
    return response


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


def _mcp_identity_error(exc: AuthenticationConfigurationError) -> dict[str, Any]:
    return mcp_identity_error(exc)


async def upload_results_file(
    campaign_id: str,
    file_content: str,
    file_format: str = "csv",
    submitted_by: str | None = None,
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Compatibility helper for Python callers that already have a user id.

    The registered MCP tool below resolves ``submitted_by`` internally and
    does not expose database user ids to agents.
    """
    with bind_trace_id(trace_id):
        return await _upload_dispatch(
            campaign_id=campaign_id,
            file_content=file_content,
            file_format=file_format,
            submitted_by=submitted_by,
            idempotency_key=idempotency_key,
            dry_run=dry_run,
        )


@mcp.tool(name="bo_upload_results_file", annotations=NON_IDEMPOTENT_MUTATION)
async def _upload_results_file_tool(
    campaign_id: str,
    file_content: str,
    file_format: str = "csv",
    idempotency_key: str | None = None,
    dry_run: bool = False,
    trace_id: str | None = None,
) -> UploadResultsResponse:
    """Upload experimental results from a CSV file.

    Workflow: Alternative to bo_submit_results for bulk uploads. Use when
    you have historical data in CSV format.

    CSV format should have columns:
    - param_<name>: Parameter values (e.g., param_temperature, param_pressure)
    - obj_<name>: Objective values (e.g., obj_yield, obj_cost)

    The MCP transport resolves ``submitted_by`` internally from the current
    BO-MCP user identity. Agents must not provide database user ids.
    """
    with bind_trace_id(trace_id):
        # Validate the file payload before requiring identity so a malformed
        # upload returns an actionable field error even when MCP identity is
        # unconfigured — mirrors bo_create_campaign / bo_submit_results.
        # Offloaded: parsing a multi-MB CSV must not block the event loop.
        validated = await asyncio.to_thread(_validate_upload_payload, file_content, file_format)
        if isinstance(validated, dict):
            return cast(UploadResultsResponse, attach_response_metadata(validated))

        try:
            user = await resolve_mcp_user()
        except AuthenticationConfigurationError as exc:
            return cast(UploadResultsResponse, _mcp_identity_error(exc))

        return cast(
            UploadResultsResponse,
            await _upload_dispatch(
                campaign_id=campaign_id,
                file_content=file_content,
                file_format=file_format,
                submitted_by=str(user.id),
                idempotency_key=idempotency_key,
                dry_run=dry_run,
                parsed_payload=validated,
            ),
        )


async def _upload_dispatch(
    campaign_id: str,
    file_content: str,
    file_format: str,
    submitted_by: str | None,
    idempotency_key: str | None,
    dry_run: bool,
    parsed_payload: _ParsedUpload | None = None,
) -> dict[str, Any]:
    """Pick the dry-run vs idempotent code path for the upload tool.

    Kept separate from ``upload_results_file`` so the public tool
    function stays under ruff's cognitive-complexity ceiling while we
    still wrap the call in :func:`bind_trace_id`. ``parsed_payload``
    carries rows already parsed by the caller so the inner pipeline
    does not re-parse the same CSV.
    """
    if dry_run:
        return await _upload_results_file_inner(
            campaign_id=campaign_id,
            file_content=file_content,
            file_format=file_format,
            submitted_by=submitted_by,
            dry_run=True,
            parsed_payload=parsed_payload,
        )

    # Pre-digest ``file_content`` so a multi-MB CSV upload does not
    # inflate the idempotency cache row. The digest is collision-safe
    # within this cache's lifetime, so a retry with the same bytes
    # still matches, and a retry with different bytes (a corrected
    # CSV) still produces an ``idempotency_conflict`` envelope.
    request_payload = {
        "campaign_id": campaign_id,
        "file_content_digest": digest_large_field(file_content),
        "file_format": file_format,
        "submitted_by": submitted_by,
    }

    async def run(session: AsyncSession) -> dict[str, Any]:
        return await _upload_results_file_inner(
            campaign_id=campaign_id,
            file_content=file_content,
            file_format=file_format,
            submitted_by=submitted_by,
            session=session,
            parsed_payload=parsed_payload,
        )

    return await apply_idempotency(
        tool_name="bo_upload_results_file",
        idempotency_key=idempotency_key,
        request_payload=request_payload,
        executor=run,
    )


def _validate_upload_payload(
    file_content: str,
    file_format: str,
) -> dict[str, Any] | tuple[list[ResultSubmissionInput], list[str]]:
    """Validate the upload file payload — size, format, parseable rows — without DB or identity.

    Single source of upload payload validation, shared by the MCP tool's
    pre-identity check and the inner ingest path so a malformed file produces
    the same structured envelope regardless of where it is caught. Returns the
    parsed rows and any per-row parse errors on success, or an error envelope
    dict on failure.
    """
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

    return parsed_results, parse_errors


@with_response_metadata
async def _upload_results_file_inner(
    campaign_id: str,
    file_content: str,
    file_format: str,
    submitted_by: str | None,
    *,
    session: AsyncSession | None = None,
    dry_run: bool = False,
    parsed_payload: _ParsedUpload | None = None,
) -> dict[str, Any]:
    """Core upload-file pipeline used by the public tool and the cache path.

    Decorated with ``with_response_metadata`` so every early validation
    return (oversized file, unsupported format, invalid UUID, empty CSV,
    parse error) carries ``_metadata.trace_id`` like the success path
    already does via the inherited submit-results envelope. Without
    this, traced uploads that failed validation were the only
    upload-tool returns missing the trace echo.

    ``parsed_payload`` carries rows the caller already parsed (the MCP
    tool validates pre-identity); when omitted the parse runs here,
    offloaded to a worker thread. Either way the CSV is parsed exactly
    once per call.
    """
    logger.info(
        "Uploading results file for campaign %s (format=%s, size=%d bytes)",
        campaign_id,
        file_format,
        len(file_content),
    )

    validated: dict[str, Any] | _ParsedUpload = (
        parsed_payload
        if parsed_payload is not None
        else await asyncio.to_thread(_validate_upload_payload, file_content, file_format)
    )
    if isinstance(validated, dict):
        return validated
    parsed_results, parse_errors = validated

    uuid_result = _parse_uuids(campaign_id, submitted_by)
    if isinstance(uuid_result, dict):
        return uuid_result
    _, submitter_uuid = uuid_result

    submit_result = await submit_results_operation(
        campaign_id=campaign_id,
        results=parsed_results,
        submitted_by=str(submitter_uuid),
        source="file_upload",
        atomic=False,
        continue_on_error=True,
        session=session,
        dry_run=dry_run,
    )

    result_ids = submit_result.get("result_ids", [])
    n_errors = len(parse_errors) + len(submit_result.get("errors", []))
    if n_errors:
        logger.warning(
            "File upload completed with errors: %d results created, %d errors",
            len(result_ids),
            n_errors,
        )
    else:
        logger.info(
            "File upload successful: %d results created for campaign %s",
            len(result_ids),
            campaign_id,
        )

    return _build_upload_response(submit_result, parse_errors, dry_run=dry_run)
