"""Repair semantics for :func:`bo_mcp_server.client.ensure_dev_user`.

The dev-auth bootstrap must produce a user whose API key actually
authenticates under the new real-auth flow. "A user with the canonical
email exists" is no longer sufficient — the real-auth path also checks
``api_key_hash`` and ``is_active``. These tests pin the repair contract:

* If a previous run left a record with the wrong hash (e.g. the hash
  scheme changed), the next bootstrap must update it to match
  ``DEV_API_KEY``.
* If the account was deactivated, the next bootstrap must reactivate
  it. Otherwise the dev key would be silently rejected by
  :func:`get_user_by_api_key` (which filters on ``is_active``).
* Fresh databases must still create the canonical record.
* The happy path (existing record already correct) must be a no-op.

Reference: OWASP ASVS V2.5 — credential lifecycle.
"""

from __future__ import annotations

import hashlib

import pytest

from bo_mcp_server.client import (
    DEV_API_KEY,
    DEV_USER_EMAIL,
    DEV_USER_NAME,
    ensure_dev_user,
    get_user_by_api_key,
)
from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session

pytestmark = pytest.mark.usefixtures("setup_database")


def _dev_hash() -> str:
    return hashlib.sha256(DEV_API_KEY.encode()).hexdigest()


@pytest.mark.asyncio
async def test_creates_user_on_empty_db() -> None:
    user = await ensure_dev_user()
    assert user.email == DEV_USER_EMAIL
    assert user.name == DEV_USER_NAME
    assert user.api_key_hash == _dev_hash()
    assert user.is_active is True

    # The standard auth path must accept the dev key after bootstrap.
    via_auth = await get_user_by_api_key(DEV_API_KEY)
    assert via_auth is not None
    assert via_auth.id == user.id


@pytest.mark.asyncio
async def test_returns_existing_user_unchanged_when_already_correct() -> None:
    first = await ensure_dev_user()
    second = await ensure_dev_user()
    assert first.id == second.id  # same row, no churn


@pytest.mark.asyncio
async def test_repairs_user_with_stale_api_key_hash() -> None:
    """A leftover record with the wrong hash must be repaired in place.

    Reproduces the situation a developer falls into after a hash-scheme
    change: a previous run created the dev row, the canonical hash
    changed (or someone hand-edited it), and the dev key would now
    silently fail to authenticate.
    """
    stale = User(
        name=DEV_USER_NAME,
        email=DEV_USER_EMAIL,
        api_key_hash="legacy-wrong-hash",
    )
    async with get_session() as session:
        await UserRepository(session).save(stale)

    repaired = await ensure_dev_user()
    assert repaired.api_key_hash == _dev_hash()
    assert repaired.is_active is True

    via_auth = await get_user_by_api_key(DEV_API_KEY)
    assert via_auth is not None
    assert via_auth.id == repaired.id


@pytest.mark.asyncio
async def test_reactivates_deactivated_dev_user() -> None:
    """A deactivated record must be reactivated.

    Without this, :func:`get_user_by_api_key` (which filters on
    ``is_active``) would reject the dev key even though the lifespan
    appeared to bootstrap a usable account.
    """
    deactivated = User(
        name=DEV_USER_NAME,
        email=DEV_USER_EMAIL,
        api_key_hash=_dev_hash(),
        is_active=False,
    )
    async with get_session() as session:
        await UserRepository(session).save(deactivated)

    repaired = await ensure_dev_user()
    assert repaired.is_active is True

    via_auth = await get_user_by_api_key(DEV_API_KEY)
    assert via_auth is not None
    assert via_auth.id == repaired.id
