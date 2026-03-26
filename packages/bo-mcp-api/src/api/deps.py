"""FastAPI dependencies."""

from typing import Annotated
from uuid import UUID

from bo_mcp_server.domain import Campaign, Suggestion, User
from bo_mcp_server.storage import (
    CampaignRepository,
    SuggestionRepository,
    get_session,
)
from fastapi import Depends, Header, HTTPException, status

from api.dev_auth import ensure_dev_user


async def get_current_user(
    x_api_key: Annotated[str | None, Header()] = None,
) -> User:
    """Return the shared development user for every request.

    This is a temporary development-only bypass. It intentionally ignores the
    incoming ``X-API-Key`` header so route ownership and submission logic can
    continue to use a concrete ``current_user`` without enforcing real auth.
    This is not a sustainable production configuration.
    """
    # Revert reference: restore the original API-key behavior by re-adding the
    # imports below at module scope and replacing this function body with the
    # commented block that follows.
    #
    # from bo_mcp_server.storage import UserRepository
    # import hashlib
    #
    # Original implementation:
    # api_key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()
    #
    # async with get_session() as session:
    #     user_repo = UserRepository(session)
    #     user = await user_repo.get_by_api_key_hash(api_key_hash)
    #
    #     if user is None:
    #         raise HTTPException(
    #             status_code=status.HTTP_401_UNAUTHORIZED,
    #             detail="Invalid API key",
    #         )
    #
    #     if not user.is_active:
    #         raise HTTPException(
    #             status_code=status.HTTP_403_FORBIDDEN,
    #             detail="User account is deactivated",
    #         )
    #
    #     return user
    _ = x_api_key
    return await ensure_dev_user()


# Type alias for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_optional_user(
    x_api_key: Annotated[str | None, Header()] = None,
) -> User | None:
    """Get current user if API key provided, None otherwise.

    Used for endpoints that work with or without authentication.
    """
    if x_api_key is None:
        return None

    try:
        return await get_current_user(x_api_key)
    except HTTPException:
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]


def validate_uuid(value: str, name: str = "id") -> UUID:
    """Validate and parse a UUID string.

    Args:
        value: String representation of UUID
        name: Parameter name for error message (e.g., "campaign_id", "spec_id")

    Returns:
        Parsed UUID object

    Raises:
        HTTPException: If the string is not a valid UUID format
    """
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {name} format",
        ) from None


async def get_authorized_campaign(
    campaign_id: str,
    current_user: User,
) -> Campaign:
    """Fetch and authorize a campaign for the current user.

    Args:
        campaign_id: Campaign UUID string
        current_user: Authenticated user

    Returns:
        Campaign entity if authorized

    Raises:
        HTTPException: 400 if invalid UUID, 404 if not found, 403 if not authorized
    """
    campaign_uuid = validate_uuid(campaign_id, "campaign_id")

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
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

        return campaign


async def get_authorized_suggestion(
    suggestion_id: str,
    current_user: User,
) -> Suggestion:
    """Fetch and authorize a suggestion via its owning campaign."""
    suggestion_uuid = validate_uuid(suggestion_id, "suggestion_id")

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)
        campaign_repo = CampaignRepository(session)

        suggestion = await suggestion_repo.get(suggestion_uuid)
        if suggestion is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Suggestion {suggestion_id} not found",
            )

        campaign = await campaign_repo.get(suggestion.campaign_id)
        if campaign is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Campaign {suggestion.campaign_id} not found",
            )

        if campaign.owner_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this suggestion",
            )

        return suggestion


async def ensure_owned_campaigns(
    campaign_ids: list[str],
    current_user: User,
) -> None:
    """Reject requests that include campaigns owned by another user.

    Invalid or missing campaign IDs are intentionally ignored here so the shared
    operation can surface them in the MCP-aligned response payload.
    """
    async with get_session() as session:
        campaign_repo = CampaignRepository(session)

        for campaign_id in campaign_ids:
            try:
                campaign_uuid = UUID(campaign_id)
            except ValueError:
                continue

            campaign = await campaign_repo.get(campaign_uuid)
            if campaign is None:
                continue

            if campaign.owner_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not authorized to access campaign {campaign_id}",
                )
