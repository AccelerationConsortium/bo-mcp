"""Campaign routes."""

from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session
from bo_mcp_server.tools.create_campaign import create_campaign
from fastapi import APIRouter, HTTPException, status

from api.deps import CurrentUser, get_authorized_campaign, validate_uuid
from api.schemas.campaign import (
    CampaignCreate,
    CampaignCreateResponse,
    CampaignListResponse,
    CampaignResponse,
)

router = APIRouter()


@router.post("", response_model=CampaignCreateResponse)
async def create_new_campaign(
    request: CampaignCreate,
    current_user: CurrentUser,
) -> CampaignCreateResponse:
    """Create a new optimization campaign.

    This is a thin proxy to the MCP create_campaign tool.
    """
    result = await create_campaign(
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
