"""Shared row factories for tests that need real parent rows.

The SQLite test engine enforces foreign keys (mirroring the PostgreSQL
production behavior), so a campaign can no longer be created against a
synthetic ``owner_id`` UUID with no matching ``users`` row. Tests that
previously used ``owner_id = str(uuid4())`` seed a real user through
this module instead.
"""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from bo_mcp_server.domain import (
    CampaignSpec,
    InputParameter,
    Objective,
    ParameterType,
    User,
)
from bo_mcp_server.storage import CampaignSpecRepository, UserRepository, get_session


async def seed_owner(name: str = "Test Owner") -> str:
    """Insert a user row and return its id as a string.

    Each call creates a distinct user (unique email and API-key hash),
    matching the old ``str(uuid4())`` behavior of giving every test its
    own owner while satisfying the ``campaigns.owner_id`` foreign key.
    """
    unique = str(uuid4())
    async with get_session() as session:
        user = await UserRepository(session).save(
            User(
                name=name,
                email=f"owner-{unique}@example.com",
                api_key_hash=hashlib.sha256(unique.encode()).hexdigest(),
            )
        )
    return str(user.id)


async def seed_campaign_parents() -> tuple[UUID, UUID]:
    """Insert a user and a minimal campaign spec; return ``(owner_id, spec_id)``.

    For tests that construct ``Campaign`` entities directly against the
    repository layer: both foreign keys (``owner_id``, ``spec_id``) must
    reference committed parent rows under FK enforcement.
    """
    owner_id = UUID(await seed_owner())
    spec = CampaignSpec(
        name="Factory Spec",
        parameters=(
            InputParameter(
                name="x",
                type=ParameterType.CONTINUOUS,
                bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            ),
        ),
        objectives=(Objective(name="y", direction="minimize"),),
    )
    spec_id = uuid4()
    async with get_session() as session:
        await CampaignSpecRepository(session).save(spec, spec_id=spec_id)
    return owner_id, spec_id
