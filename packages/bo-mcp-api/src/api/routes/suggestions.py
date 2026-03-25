"""Suggestion routes."""

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.storage import CampaignRepository, SuggestionRepository, get_session
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from fastapi import APIRouter, HTTPException, Query, status

from api.deps import (
    CurrentUser,
    get_authorized_campaign,
    get_authorized_suggestion,
    validate_uuid,
)
from api.schemas.suggestion import (
    SuggestionExplanationResponse,
    SuggestionResponse,
    SuggestionsGenerateResponse,
)
from api.schemas.suggestion import (
    SuggestionProvenance as SuggestionProvenanceSchema,
)

router = APIRouter()


@router.post("/{campaign_id}/generate", response_model=SuggestionsGenerateResponse)
async def generate_campaign_suggestions(
    campaign_id: str,
    current_user: CurrentUser,
    batch_size: int | None = Query(default=None, ge=1),
) -> SuggestionsGenerateResponse:
    """Generate new suggestions for a campaign.

    This is a thin proxy to the MCP bo_generate_suggestions tool.
    """
    await get_authorized_campaign(campaign_id, current_user)

    # Generate suggestions via MCP tool
    result = await generate_suggestions(
        campaign_id=campaign_id,
        batch_size=batch_size,
    )

    if not result["success"]:
        return SuggestionsGenerateResponse(
            success=False,
            suggestions=[],
            iteration=result.get("iteration"),
            errors=result["errors"],
        )

    # Convert to response format
    suggestions = [
        SuggestionResponse(
            id=s["id"],
            campaign_id=campaign_id,
            parameter_values=s["parameter_values"],
            status="pending",
            provenance=s["provenance"],
            created_at=s.get("created_at", ""),
        )
        for s in result["suggestions"]
    ]

    return SuggestionsGenerateResponse(
        success=True,
        suggestions=suggestions,
        iteration=result["iteration"],
        errors=[],
    )


@router.get(
    "/{suggestion_id}/explanation",
    response_model=SuggestionExplanationResponse,
)
async def get_campaign_suggestion_explanation(
    suggestion_id: str,
    current_user: CurrentUser,
) -> SuggestionExplanationResponse:
    """Get a detailed explanation for a suggestion."""
    await get_authorized_suggestion(suggestion_id, current_user)

    result = await get_suggestion_explanation_operation(suggestion_id)
    return SuggestionExplanationResponse(**result)


@router.get("/{campaign_id}", response_model=list[SuggestionResponse])
async def list_campaign_suggestions(
    campaign_id: str,
    current_user: CurrentUser,
    status_filter: str | None = Query(default=None, alias="status"),
) -> list[SuggestionResponse]:
    """List suggestions for a campaign."""
    campaign_uuid = validate_uuid(campaign_id, "campaign_id")

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        suggestion_repo = SuggestionRepository(session)

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

        # Get suggestions
        status_enum = None
        if status_filter:
            try:
                status_enum = SuggestionStatus(status_filter)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}",
                ) from None

        suggestions = await suggestion_repo.list_by_campaign(campaign_uuid, status_enum)

        return [
            SuggestionResponse(
                id=str(s.id),
                campaign_id=str(s.campaign_id),
                parameter_values=s.parameter_values,
                status=s.status.value,
                provenance=SuggestionProvenanceSchema(
                    iteration=s.provenance.iteration,
                    batch_index=s.provenance.batch_index,
                    acquisition_value=s.provenance.acquisition_value,
                    model_uncertainty=s.provenance.model_uncertainty,
                    generation_method=s.provenance.generation_method,
                    acquisition_function=s.provenance.acquisition_function,
                    model_type=s.provenance.model_type,
                    random_seed=s.provenance.random_seed,
                    model_version=s.provenance.model_version,
                    confidence_level=s.provenance.confidence_level,
                    explanation=s.provenance.explanation,
                ),
                created_at=s.created_at,
            )
            for s in suggestions
        ]
