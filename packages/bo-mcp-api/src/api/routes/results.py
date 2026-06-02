"""Results routes."""

import logging
from uuid import UUID

from bo_mcp_server.client import (
    CampaignSpec,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    ResultSubmissionInput,
    canonical_submit_results_payload,
    get_campaign_with_spec,
    http_status_for_error,
    list_campaign_results,
    list_results_operation,
    parse_named_result_rows,
    run_idempotent_operation,
    submit_results_operation,
)
from fastapi import APIRouter, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import CurrentUser, IdempotencyKey, get_authorized_campaign
from api.limits import (
    MAX_BATCH_RESULTS,
    MAX_UPLOAD_FILE_SIZE_BYTES,
    UPLOAD_READ_CHUNK_BYTES,
)
from api.schemas.common import API_RESPONSE_SCHEMA_VERSION
from api.schemas.errors import (
    COMMON_HTTP_ERROR_RESPONSES,
    IDEMPOTENCY_ERROR_RESPONSES,
    HttpErrorResponse,
    operation_failure_response,
)
from api.schemas.result import (
    ResultBatchCreate,
    ResultQueryRequest,
    ResultQueryResponse,
    ResultResponse,
    ResultSubmitResponse,
)
from api.upload_parser import UploadParseError, parse_upload_rows

logger = logging.getLogger(__name__)

router = APIRouter(responses=COMMON_HTTP_ERROR_RESPONSES)


def _results_location(campaign_id: str) -> str:
    """URL of the canonical GET that resolves this campaign's results.

    Batch creates have no per-row GET, so we point ``Location`` at the
    collection that contains every result row produced by the call.
    """
    return f"/api/v1/results/{campaign_id}"


async def _read_upload_bounded(file: UploadFile) -> bytes:
    """Read an :class:`UploadFile` in chunks, refusing past the size cap.

    Streaming the read means an oversize payload is rejected before
    it occupies the full memory footprint; the alternative
    (``await file.read()``) would buffer the whole body first and
    only check length afterwards.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(UPLOAD_READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=(f"Upload exceeds the {MAX_UPLOAD_FILE_SIZE_BYTES}-byte limit."),
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/{campaign_id}",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: operation_failure_response(
            model=ResultSubmitResponse,
            description=(
                "Operation-level result submission rejection. The HTTP request was "
                "processed, but no result rows were persisted; inspect success=false, "
                "errors, and field_errors."
            ),
            example={
                "schema_version": API_RESPONSE_SCHEMA_VERSION,
                "success": False,
                "result_ids": [],
                "errors": ["Result row failed validation."],
                "warnings": [],
                "field_errors": {"results.0.objective_values": ["Missing objective y"]},
                "idempotency_replay": False,
            },
        ),
        **IDEMPOTENCY_ERROR_RESPONSES,
    },
)
async def submit_campaign_results(
    campaign_id: str,
    request: ResultBatchCreate,
    current_user: CurrentUser,
    idempotency_key: IdempotencyKey,
    response: Response,
) -> ResultSubmitResponse:
    """Submit results for a campaign.

    Returns ``201 Created`` with a ``Location`` header pointing at
    :func:`list_campaign_results_route` for the freshly-inserted
    batch. Operation-level rejections (``success=False`` envelopes
    from validation failures) keep the historical ``200 OK`` shape
    so existing tests for that path still see the envelope rather
    than a routed-out HTTP error.

    Honours the ``Idempotency-Key`` request header (same cache
    namespace as the MCP ``bo_submit_results`` tool) so a retry
    replays the cached response instead of persisting the batch
    twice.
    """
    await get_authorized_campaign(campaign_id, current_user)

    results_data = [
        ResultSubmissionInput(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            suggestion_id=r.suggestion_id,
            measurement_uncertainty=r.measurement_uncertainty,
            metadata=r.metadata.model_dump(exclude_unset=True, mode="json"),
        )
        for r in request.results
    ]
    submitted_by = str(current_user.id)

    async def run(session: AsyncSession) -> dict:
        return await submit_results_operation(
            campaign_id=campaign_id,
            results=results_data,
            submitted_by=submitted_by,
            source=request.source,
            session=session,
        )

    # Shared canonical builder so REST and MCP hash the same shape for
    # semantically identical batches — the precondition for the
    # audit's "same cache namespace" promise.
    result = await run_idempotent_operation(
        operation_name="bo_submit_results",
        idempotency_key=idempotency_key,
        request_payload=canonical_submit_results_payload(
            campaign_id=campaign_id,
            results=results_data,
            submitted_by=submitted_by,
            source=request.source,
        ),
        executor=run,
    )
    # Idempotency-layer envelopes lack the operation's success-shape
    # fields (``result_ids`` here); promote them to a typed HTTPException
    # rather than letting the unpacking below trigger a KeyError 500.
    if "result_ids" not in result:
        raise HTTPException(
            status_code=http_status_for_error(result),
            detail=result.get("error", {"message": "Idempotency error"}),
        )

    if result.get("success"):
        response.headers["Location"] = _results_location(campaign_id)
    else:
        response.status_code = status.HTTP_200_OK

    return ResultSubmitResponse(
        success=result["success"],
        result_ids=result["result_ids"],
        errors=result["errors"],
        warnings=result["warnings"],
        field_errors=result.get("field_errors", {}),
        # Forward the wrapper's replay marker so REST clients can
        # distinguish a cached batch from a fresh insert.
        idempotency_replay=bool(result.get("idempotency_replay", False)),
    )


async def _resolve_upload_spec(campaign_id: str, user_id: UUID) -> CampaignSpec:
    """Resolve the campaign spec for an upload, mapping facade errors to HTTP."""
    try:
        _, spec = await get_campaign_with_spec(campaign_id, user_id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid campaign_id format",
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign",
        ) from None
    except NotFoundError as exc:
        detail = (
            "Campaign spec not found"
            if exc.resource == "Campaign spec"
            else f"Campaign {campaign_id} not found"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        ) from None
    return spec


@router.post(
    "/{campaign_id}/upload",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: operation_failure_response(
            model=ResultSubmitResponse,
            description=(
                "Operation-level upload rejection after parsing succeeded. Inspect "
                "success=false, errors, and field_errors."
            ),
            example={
                "schema_version": API_RESPONSE_SCHEMA_VERSION,
                "success": False,
                "result_ids": [],
                "errors": ["Uploaded results failed validation."],
                "warnings": [],
                "field_errors": {"rows.2": ["Parameter value is out of bounds"]},
                "idempotency_replay": False,
            },
        ),
        413: {
            "model": HttpErrorResponse,
            "description": "Uploaded file or parsed result batch exceeds the configured limit.",
        },
    },
)
async def upload_results_file(
    campaign_id: str,
    file: UploadFile,
    current_user: CurrentUser,
    response: Response,
) -> ResultSubmitResponse:
    """Upload results from CSV or Excel file.

    Streams the upload through :func:`_read_upload_bounded` (refusing
    over :data:`api.limits.MAX_UPLOAD_FILE_SIZE_BYTES`) and parses
    with :func:`api.upload_parser.parse_upload_rows`, which uses
    ``openpyxl(read_only=True)`` so a workbook is iterated rather
    than materialised in full. Parse failures surface as a sanitized
    400 — the underlying library exception is logged server-side
    but not echoed to the client (so library names / versions stay
    out of the response body).
    """
    spec = await _resolve_upload_spec(campaign_id, current_user.id)

    filename = file.filename or ""
    content = await _read_upload_bounded(file)

    try:
        headers, upload_rows = parse_upload_rows(filename, content)
    except UploadParseError as exc:
        logger.warning(
            "Rejected upload for campaign %s (%s)",
            campaign_id,
            type(exc.__cause__).__name__ if exc.__cause__ else "no cause",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    param_names = [p.name for p in spec.parameters]
    objective_names = [o.name for o in spec.objectives]

    # Column validation against the *header row* — not against the
    # union of data-row keys. Pre-fix, a header-only CSV would be
    # misreported as "Missing columns" even when the column names
    # were declared correctly; the real defect there is "no data
    # rows", which we surface separately below.
    missing_cols = set(param_names + objective_names) - set(headers)
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing columns: {sorted(missing_cols)}",
        )

    if not upload_rows:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload contains no result rows.",
        )

    # Apply the same per-batch cap as the JSON ``ResultBatchCreate``
    # schema (``MAX_BATCH_RESULTS``). Without this the upload route
    # would silently accept arbitrarily many rows for any CSV/XLSX
    # under the 25 MiB file-size cap; the JSON-body cap is on the
    # ``results`` list only, not on a different transport's parse
    # output.
    if len(upload_rows) > MAX_BATCH_RESULTS:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                f"Upload contains {len(upload_rows)} rows, exceeding the "
                f"per-batch cap of {MAX_BATCH_RESULTS}. Split the file into "
                "smaller batches and resubmit."
            ),
        )

    results_data, parse_errors = parse_named_result_rows(
        upload_rows,
        parameter_names=param_names,
        objective_names=objective_names,
        metadata_factory=lambda row_num: {"source_file": filename, "source_row": row_num},
    )

    if parse_errors:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=parse_errors,
        )

    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=results_data,
        submitted_by=str(current_user.id),
        source="file_upload",
    )

    if result.get("success"):
        response.headers["Location"] = _results_location(campaign_id)
    else:
        response.status_code = status.HTTP_200_OK

    return ResultSubmitResponse(
        success=result["success"],
        result_ids=result["result_ids"],
        errors=result["errors"],
        warnings=result["warnings"],
        field_errors=result.get("field_errors", {}),
    )


@router.post("/{campaign_id}/query")
async def query_campaign_results(
    campaign_id: str,
    request: ResultQueryRequest,
    current_user: CurrentUser,
) -> ResultQueryResponse:
    """Query results for a campaign with pagination and verbosity control."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await list_results_operation(
        campaign_id=campaign_id,
        limit=request.limit,
        offset=request.offset,
        verbosity=request.verbosity,
    )
    return ResultQueryResponse(**result)


@router.get("/{campaign_id}")
async def list_campaign_results_route(
    campaign_id: str,
    current_user: CurrentUser,
) -> list[ResultResponse]:
    """List results for a campaign."""
    try:
        results = await list_campaign_results(campaign_id, current_user.id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid campaign_id format",
        ) from None
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {campaign_id} not found",
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign",
        ) from None

    return [
        ResultResponse(
            id=str(r.id),
            campaign_id=str(r.campaign_id),
            suggestion_id=str(r.suggestion_id) if r.suggestion_id else None,
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            source=r.source.value,
            submitted_by=str(r.submitted_by),
            measurement_uncertainty=r.measurement_uncertainty,
            created_at=r.created_at,
        )
        for r in results
    ]
