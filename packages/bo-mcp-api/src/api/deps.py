"""FastAPI dependencies.

The REST transport layer authorizes requests using the helpers exposed
on :mod:`bo_mcp_server.client`. Storage / repository access is **not**
imported here directly; the facade owns that translation.
"""

from typing import Annotated
from uuid import UUID

from bo_mcp_server.client import (
    Campaign,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    Suggestion,
    User,
    authorize_campaign,
    authorize_suggestion,
    get_user_by_api_key,
    parse_uuid,
)
from bo_mcp_server.client import (
    ensure_owned_campaigns as _ensure_owned_campaigns,
)
from fastapi import Depends, Header, HTTPException, status


async def get_current_user(
    x_api_key: Annotated[str | None, Header()] = None,
) -> User:
    """Resolve the caller from the ``X-API-Key`` header.

    The header value is hashed (SHA-256) and looked up against the
    persisted user table — the same algorithm
    :func:`bo_mcp_server.client.ensure_dev_user` uses to populate
    ``UserModel.api_key_hash``. Missing or unknown keys yield 401 so
    multi-tenancy boundaries downstream (ownership checks,
    audit attribution) always see a real principal.

    The 401 challenge body is intentionally generic; we do not
    distinguish "missing key" from "unknown key" so an attacker cannot
    enumerate which keys are merely malformed.
    """
    if x_api_key is None or not x_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'ApiKey realm="bo-mcp-api"'},
        )

    user = await get_user_by_api_key(x_api_key)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": 'ApiKey realm="bo-mcp-api"'},
        )
    return user


# Type alias for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_optional_user(
    x_api_key: Annotated[str | None, Header()] = None,
) -> User | None:
    """Get current user if API key provided and valid, None otherwise."""
    if x_api_key is None or not x_api_key.strip():
        return None
    return await get_user_by_api_key(x_api_key)


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
        return parse_uuid(value, name)
    except InvalidIdentifierError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid {exc.name} format",
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
    try:
        return await authorize_campaign(campaign_id, current_user.id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid campaign_id format",
        ) from None
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {exc.identifier} not found",
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign",
        ) from None


async def get_authorized_suggestion(
    suggestion_id: str,
    current_user: User,
) -> Suggestion:
    """Fetch and authorize a suggestion via its owning campaign."""
    try:
        return await authorize_suggestion(suggestion_id, current_user.id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid suggestion_id format",
        ) from None
    except NotFoundError as exc:
        # Distinguish suggestion-not-found from owning-campaign-not-found by
        # the resource label the facade set, so the message matches what the
        # repository-direct version returned.
        detail = (
            f"Suggestion {exc.identifier} not found"
            if exc.resource == "Suggestion"
            else f"Campaign {exc.identifier} not found"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this suggestion",
        ) from None


async def ensure_owned_campaigns(
    campaign_ids: list[str],
    current_user: User,
) -> None:
    """Reject requests that include campaigns owned by another user.

    Invalid or missing campaign IDs are intentionally ignored here so the shared
    operation can surface them in the MCP-aligned response payload.
    """
    try:
        await _ensure_owned_campaigns(campaign_ids, current_user.id)
    except NotAuthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized to access campaign {exc.identifier}",
        ) from None
