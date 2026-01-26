"""Results routes."""

import io

import pandas as pd
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    get_session,
)
from bo_mcp_server.tools.submit_results import submit_results
from fastapi import APIRouter, HTTPException, UploadFile, status

from api.deps import CurrentUser, get_authorized_campaign, validate_uuid
from api.schemas.result import (
    ResultBatchCreate,
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
    """Submit results for a campaign.

    This is a thin proxy to the MCP submit_results tool.
    """
    await get_authorized_campaign(campaign_id, current_user)

    # Convert to MCP format
    results_data = [
        {
            "parameter_values": r.parameter_values,
            "objective_values": r.objective_values,
            "suggestion_id": r.suggestion_id,
            "metadata": r.metadata,
        }
        for r in request.results
    ]

    # Submit via MCP tool
    result = await submit_results(
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
    campaign = await get_authorized_campaign(campaign_id, current_user)

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

    # Get spec to identify columns
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(campaign.spec_id)

        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign spec not found",
            )

    param_names = [p.name for p in spec.parameters]
    objective_names = [o.name for o in spec.objectives]

    # Validate columns exist
    missing_cols = set(param_names + objective_names) - set(df.columns)
    if missing_cols:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing columns: {missing_cols}",
        )

    # Convert to results
    results_data = []
    for _, row in df.iterrows():
        results_data.append(
            {
                "parameter_values": {name: row[name] for name in param_names},
                "objective_values": {name: float(row[name]) for name in objective_names},
                "metadata": {"source_file": filename},
            }
        )

    # Submit via MCP tool
    result = await submit_results(
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


@router.get("/{campaign_id}", response_model=list[ResultResponse])
async def list_campaign_results(
    campaign_id: str,
    current_user: CurrentUser,
) -> list[ResultResponse]:
    """List results for a campaign."""
    campaign_uuid = validate_uuid(campaign_id, "campaign_id")

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        result_repo = ResultRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign {campaign_id} not found",
            )

        if campaign.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this campaign",
            )

        results = await result_repo.list_by_campaign(campaign_uuid)

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
