"""Suggestion routes."""

from typing import Annotated

from bo_mcp_server.client import (
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    SuggestionStatus,
    generate_suggestions_operation,
    get_suggestion_explanation_operation,
    list_campaign_suggestions,
    list_suggestions_operation,
    update_suggestion_status_operation,
)
from fastapi import APIRouter, HTTPException, Query, Response, status

from api.deps import (
    CurrentUser,
    get_authorized_campaign,
    get_authorized_suggestion,
)
from api.schemas.common import API_RESPONSE_SCHEMA_VERSION
from api.schemas.errors import COMMON_HTTP_ERROR_RESPONSES, operation_failure_response
from api.schemas.suggestion import (
    SuggestionExplanationResponse,
    SuggestionQueryRequest,
    SuggestionQueryResponse,
    SuggestionResponse,
    SuggestionsGenerateResponse,
    SuggestionStatusUpdateRequest,
    SuggestionStatusUpdateResponse,
)
from api.schemas.suggestion import (
    SuggestionProvenance as SuggestionProvenanceSchema,
)

router = APIRouter(responses=COMMON_HTTP_ERROR_RESPONSES)


@router.post(
    "/{campaign_id}/generate",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: operation_failure_response(
            model=SuggestionsGenerateResponse,
            description=(
                "Operation-level suggestion generation rejection. No suggestions were "
                "persisted; inspect success=false and errors."
            ),
            example={
                "schema_version": API_RESPONSE_SCHEMA_VERSION,
                "success": False,
                "suggestions": [],
                "iteration": None,
                "errors": ["Stopping criteria have already been met."],
            },
        ),
    },
)
async def generate_campaign_suggestions(
    campaign_id: str,
    current_user: CurrentUser,
    response: Response,
    batch_size: Annotated[int | None, Query(ge=1)] = None,
) -> SuggestionsGenerateResponse:
    """Generate new suggestions for a campaign.

    Returns ``201 Created`` with a ``Location`` header pointing at
    :func:`list_campaign_suggestions_route` for the freshly-created
    batch. Operation-level rejections (stopping criteria triggered,
    backend failure, etc.) keep the historical ``200 OK`` shape so
    existing tests that inspect the ``success=False`` envelope still
    see it rather than a redirected HTTP error.
    """
    await get_authorized_campaign(campaign_id, current_user)

    result = await generate_suggestions_operation(
        campaign_id=campaign_id,
        batch_size=batch_size,
    )

    if not result["success"]:
        # Operation-level rejection: no suggestions were persisted,
        # so 201 would mislead clients. Drop back to 200 with the
        # structured envelope.
        response.status_code = status.HTTP_200_OK
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

    response.headers["Location"] = f"/api/v1/suggestions/{campaign_id}"
    return SuggestionsGenerateResponse(
        success=True,
        suggestions=suggestions,
        iteration=result["iteration"],
        errors=[],
    )


@router.get("/{suggestion_id}/explanation")
async def get_campaign_suggestion_explanation(
    suggestion_id: str,
    current_user: CurrentUser,
) -> SuggestionExplanationResponse:
    """Get a detailed explanation for a suggestion."""
    await get_authorized_suggestion(suggestion_id, current_user)

    result = await get_suggestion_explanation_operation(suggestion_id)
    return SuggestionExplanationResponse(**result)


@router.post("/{campaign_id}/query")
async def query_campaign_suggestions(
    campaign_id: str,
    request: SuggestionQueryRequest,
    current_user: CurrentUser,
) -> SuggestionQueryResponse:
    """Query suggestions for a campaign with filtering, pagination, and verbosity control."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await list_suggestions_operation(
        campaign_id=campaign_id,
        status_filter=request.status_filter,
        limit=request.limit,
        offset=request.offset,
        verbosity=request.verbosity,
    )
    return SuggestionQueryResponse(**result)


@router.post("/{suggestion_id}/status")
async def update_suggestion_status(
    suggestion_id: str,
    request: SuggestionStatusUpdateRequest,
    current_user: CurrentUser,
) -> SuggestionStatusUpdateResponse:
    """Update the status of a suggestion (accept, reject, or expire)."""
    await get_authorized_suggestion(suggestion_id, current_user)

    result = await update_suggestion_status_operation(
        suggestion_id=suggestion_id,
        status=request.status,
    )
    return SuggestionStatusUpdateResponse(
        success=result["success"],
        suggestion_id=result.get("suggestion_id"),
        status=result.get("status"),
        previous_status=result.get("previous_status"),
        errors=result.get("errors", []),
    )


@router.get("/{campaign_id}")
async def list_campaign_suggestions_route(
    campaign_id: str,
    current_user: CurrentUser,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
) -> list[SuggestionResponse]:
    """List suggestions for a campaign."""
    status_enum: SuggestionStatus | None = None
    if status_filter:
        try:
            status_enum = SuggestionStatus(status_filter)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status_filter}",
            ) from None

    try:
        suggestions = await list_campaign_suggestions(
            campaign_id,
            current_user.id,
            status_filter=status_enum,
        )
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
