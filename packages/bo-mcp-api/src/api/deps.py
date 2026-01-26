"""FastAPI dependencies."""

import hashlib
from typing import Annotated
from uuid import UUID

from bo_mcp_server.domain import Campaign, User
from bo_mcp_server.storage import CampaignRepository, UserRepository, get_session
from fastapi import Depends, Header, HTTPException, status


async def get_current_user(
    x_api_key: Annotated[str, Header()],
) -> User:
    """Get current user from API key header.

    Args:
        x_api_key: API key from header

    Returns:
        User entity

    Raises:
        HTTPException: If API key is invalid
    """
    # Hash the API key for lookup
    api_key_hash = hashlib.sha256(x_api_key.encode()).hexdigest()

    async with get_session() as session:
        user_repo = UserRepository(session)
        user = await user_repo.get_by_api_key_hash(api_key_hash)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is deactivated",
            )

        return user


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
