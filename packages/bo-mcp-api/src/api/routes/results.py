"""Results routes."""

import io
from typing import Any

import pandas as pd
from bo_mcp_server.client import (
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    ResultSubmissionInput,
    get_campaign_with_spec,
    list_campaign_results,
    list_results_operation,
    parse_named_result_rows,
    submit_results_operation,
)
from fastapi import APIRouter, HTTPException, UploadFile, status

from api.deps import CurrentUser, get_authorized_campaign
from api.schemas.result import (
    ResultBatchCreate,
    ResultQueryRequest,
    ResultQueryResponse,
    ResultResponse,
    ResultSubmitResponse,
)

router = APIRouter()


@router.post("/{campaign_id}", response_model=ResultSubmitResponse)
async def submit_campaign_results(
    campaign_id: str,
    request: ResultBatchCreate,
    current_user: CurrentUser,
) -> ResultSubmitResponse:
    """Submit results for a campaign."""
    await get_authorized_campaign(campaign_id, current_user)

    results_data = [
        ResultSubmissionInput(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            suggestion_id=r.suggestion_id,
            metadata=r.metadata,
        )
        for r in request.results
    ]

    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=results_data,
        submitted_by=str(current_user.id),
        source=request.source,
    )

    return ResultSubmitResponse(
        success=result["success"],
        result_ids=result["result_ids"],
        errors=result["errors"],
        warnings=result["warnings"],
    )


@router.post("/{campaign_id}/upload", response_model=ResultSubmitResponse)
async def upload_results_file(
    campaign_id: str,
    file: UploadFile,
    current_user: CurrentUser,
) -> ResultSubmitResponse:
    """Upload results from CSV or Excel file."""
    try:
        _, spec = await get_campaign_with_spec(campaign_id, current_user.id)
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

    # Parse file
    filename = file.filename or ""
    content = await file.read()

    try:
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Use CSV or Excel.",
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse file: {e}",
        ) from e

    param_names = [p.name for p in spec.parameters]
    objective_names = [o.name for o in spec.objectives]

    # Validate columns exist
    missing_cols = set(param_names + objective_names) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing columns: {missing_cols}",
        )

    upload_rows: list[dict[str, Any]] = [
        {str(key): value for key, value in row.items()} for row in df.to_dict(orient="records")
    ]

    results_data, parse_errors = parse_named_result_rows(
        upload_rows,
        parameter_names=param_names,
        objective_names=objective_names,
        metadata_factory=lambda _row_num: {"source_file": filename},
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

    return ResultSubmitResponse(
        success=result["success"],
        result_ids=result["result_ids"],
        errors=result["errors"],
        warnings=result["warnings"],
    )


@router.post("/{campaign_id}/query", response_model=ResultQueryResponse)
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


@router.get("/{campaign_id}", response_model=list[ResultResponse])
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
            created_at=r.created_at,
        )
        for r in results
    ]
