"""Temporary development authentication helpers.

This module exists only to support local development while API-key validation
is bypassed. It is not a sustainable authentication model and must not be used
as-is for production deployments.
"""

import hashlib
import logging

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session

logger = logging.getLogger(__name__)

DEV_API_KEY = "dev-api-key-12345"
DEV_USER_NAME = "Test User"
DEV_USER_EMAIL = "test@example.com"


async def ensure_dev_user() -> User:
    """Ensure the shared development user exists and return it."""
    api_key_hash = hashlib.sha256(DEV_API_KEY.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        existing = await repo.get_by_email(DEV_USER_EMAIL)
        if existing is not None:
            return existing

        user = User(
            name=DEV_USER_NAME,
            email=DEV_USER_EMAIL,
            api_key_hash=api_key_hash,
        )
        saved = await repo.save(user)
        logger.warning(
            "Created shared development user %s for temporary auth bypass",
            saved.id,
        )
        return saved
