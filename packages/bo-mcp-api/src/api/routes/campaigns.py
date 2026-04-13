"""Campaign routes."""

from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.operations.create_campaign import create_campaign_operation
from bo_mcp_server.operations.export_campaign import export_campaign_operation
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.operations.validate_intake import validate_intake_operation
from bo_mcp_server.response_formatter import VerbosityLevel, format_validate_intake_response
from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from api.deps import (
    CurrentUser,
    ensure_owned_campaigns,
    get_authorized_campaign,
    validate_uuid,
)
from api.schemas.campaign import (
    BatchStatusRequest,
    BatchStatusResponse,
    CampaignCreate,
    CampaignCreateResponse,
    CampaignLifecycleRequest,
    CampaignLifecycleResponse,
    CampaignListResponse,
    CampaignQueryRequest,
    CampaignQueryResponse,
    CampaignResponse,
    CompareCampaignsRequest,
    CompareCampaignsResponse,
    TransferCandidatesRequest,
    TransferCandidatesResponse,
    ValidateIntakeRequest,
    ValidateIntakeResponse,
)

router = APIRouter()


@router.post("", response_model=CampaignCreateResponse)
async def create_new_campaign(
    request: CampaignCreate,
    current_user: CurrentUser,
) -> CampaignCreateResponse:
    """Create a new optimization campaign."""
    result = await create_campaign_operation(
        intake_data=request.intake.to_dict(),
        owner_id=str(current_user.id),
    )
    return CampaignCreateResponse(
        success=result["success"],
        campaign_id=result["campaign_id"],
        spec_id=result["spec_id"],
        warnings=result.get("warnings", []),
        errors=result["errors"],
    )


@router.get("", response_model=CampaignListResponse)
async def list_campaigns(current_user: CurrentUser) -> CampaignListResponse:
    """List campaigns for the current user.

    Uses batch loading to avoid N+1 query problem: fetches all campaigns
    in one query, then fetches all needed specs in a second query.
    """
    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        # Query 1: Get all campaigns for the user
        campaigns = await campaign_repo.list_by_owner(current_user.id)

        if not campaigns:
            return CampaignListResponse(campaigns=[], total=0)

        # Query 2: Batch fetch all specs in a single query (fixes N+1)
        spec_ids = [campaign.spec_id for campaign in campaigns]
        specs_by_id = await spec_repo.get_by_ids(spec_ids)

        # Build responses using the prefetched specs
        responses = []
        for campaign in campaigns:
            spec = specs_by_id.get(campaign.spec_id)
            if spec:
                responses.append(
                    CampaignResponse(
                        id=str(campaign.id),
                        spec_id=str(campaign.spec_id),
                        name=spec.name,
                        description=spec.description,
                        status=campaign.status.value,
                        iteration=campaign.iteration,
                        created_at=campaign.created_at,
                        updated_at=campaign.updated_at,
                        n_parameters=spec.n_parameters,
                        n_objectives=spec.n_objectives,
                    )
                )

        return CampaignListResponse(campaigns=responses, total=len(responses))


@router.post("/validate", response_model=ValidateIntakeResponse)
async def validate_campaign_intake(
    request: ValidateIntakeRequest,
    current_user: CurrentUser,
) -> ValidateIntakeResponse:
    """Validate a campaign specification without creating a campaign (dry-run)."""
    full_result = validate_intake_operation(request.intake.to_dict())
    formatted = format_validate_intake_response(full_result, VerbosityLevel.STANDARD)

    return ValidateIntakeResponse(
        valid=formatted["valid"],
        errors=formatted.get("errors", []),
        warnings=formatted.get("warnings", []),
        spec_summary=formatted.get("spec_summary"),
    )


@router.post("/query", response_model=CampaignQueryResponse)
async def query_campaigns(
    request: CampaignQueryRequest,
    current_user: CurrentUser,
) -> CampaignQueryResponse:
    """Query campaigns with filtering, pagination, and verbosity control."""
    result = await list_campaigns_operation(
        owner_id=current_user.id,
        status=request.status,
        limit=request.limit,
        offset=request.offset,
        verbosity=request.verbosity,
    )
    return CampaignQueryResponse(**result)


@router.post("/status/batch", response_model=BatchStatusResponse)
async def batch_campaign_status(
    request: BatchStatusRequest,
    current_user: CurrentUser,
) -> BatchStatusResponse:
    """Get status for multiple campaigns."""
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await batch_get_status_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return BatchStatusResponse(**result)


@router.post("/compare", response_model=CompareCampaignsResponse)
async def compare_campaign_group(
    request: CompareCampaignsRequest,
    current_user: CurrentUser,
) -> CompareCampaignsResponse:
    """Compare multiple campaigns."""
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await compare_campaigns_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return CompareCampaignsResponse(**result)


@router.post(
    "/{campaign_id}/lifecycle",
    response_model=CampaignLifecycleResponse,
)
async def manage_campaign(
    campaign_id: str,
    request: CampaignLifecycleRequest,
    current_user: CurrentUser,
) -> CampaignLifecycleResponse:
    """Manage campaign lifecycle."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await manage_campaign_lifecycle_operation(
        campaign_id=campaign_id,
        action=request.action,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    )
    return CampaignLifecycleResponse(**result)


@router.post(
    "/{campaign_id}/transfer-candidates",
    response_model=TransferCandidatesResponse,
)
async def discover_campaign_transfer_candidates(
    campaign_id: str,
    request: TransferCandidatesRequest,
    current_user: CurrentUser,
) -> TransferCandidatesResponse:
    """Discover transfer-learning candidates for a campaign."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await discover_transfer_candidates_operation(
        campaign_id=campaign_id,
        similarity_threshold=request.similarity_threshold,
        max_candidates=request.max_candidates,
        verbosity=request.verbosity.value,
    )
    return TransferCandidatesResponse(**result)


@router.get("/{campaign_id}/export")
async def export_campaign(
    campaign_id: str,
    current_user: CurrentUser,
    format: str = Query(default="csv"),
) -> StreamingResponse:
    """Export all campaign results as a downloadable CSV file."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await export_campaign_operation(
        campaign_id=campaign_id,
        format=format,
    )

    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("errors", ["Export failed"])[0]
            if result.get("errors")
            else result.get("message", "Export failed"),
        )

    # Sanitize campaign name for filename (name comes from the operation)
    raw_name = result.get("campaign_name", f"campaign_{campaign_id[:8]}")
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in raw_name)
    filename = safe_name.strip().replace(" ", "_")

    csv_content = result.get("content", "")
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.csv"',
        },
    )


@router.get("/spec/{spec_id}")
async def get_campaign_spec(spec_id: str, current_user: CurrentUser) -> dict:
    """Get campaign spec details."""
    spec_uuid = validate_uuid(spec_id, "spec_id")

    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(spec_uuid)

        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Spec {spec_id} not found",
            )

        return {
            "id": spec_id,
            "name": spec.name,
            "description": spec.description,
            "parameters": [p.model_dump() for p in spec.parameters],
            "objectives": [o.model_dump() for o in spec.objectives],
            "constraints": [c.model_dump() for c in spec.constraints] if spec.constraints else [],
            "batch_size": spec.batch_size,
            "created_at": "",  # Spec doesn't have created_at, use empty string
        }


@router.get("/{campaign_id}", response_model=CampaignResponse)
async def get_campaign(campaign_id: str, current_user: CurrentUser) -> CampaignResponse:
    """Get campaign details."""
    campaign = await get_authorized_campaign(campaign_id, current_user)

    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Campaign spec not found",
            )

        return CampaignResponse(
            id=str(campaign.id),
            spec_id=str(campaign.spec_id),
            name=spec.name,
            description=spec.description,
            status=campaign.status.value,
            iteration=campaign.iteration,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at,
            n_parameters=spec.n_parameters,
            n_objectives=spec.n_objectives,
        )
