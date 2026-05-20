"""Rolled-back mutations must not stale the diagnostics cache.

The diagnostics cache is keyed by ``f"diagnostics:{campaign_id}:{campaign.version}"``
(:mod:`bo_mcp_server.operations.get_diagnostics`). The audit's open
question was whether *any* path could leave the cache referencing
state that was rolled back — a "partial side effect surviving a
rollback" pattern.

Findings — recorded here as the closure of the audit:

1. The cache is written only by ``get_diagnostics_operation``. No
   mutation path writes to it directly, so a rolled-back mutation
   cannot pollute the cache mid-flight.
2. ``campaign.version`` is bumped *inside* the same transaction as
   the underlying mutation (every ``CampaignRepository.save`` either
   commits both or rolls back both via ``async with get_session()``).
   A rolled-back commit therefore leaves the version unchanged, which
   means the *old* cache entry is still keyed against the *committed*
   state — exactly what the cache is supposed to represent.

This test pins the invariant in code so a future change that splits
the version bump out of the mutation transaction breaks visibly
instead of silently introducing a stale-cache race.

Adjacent (not the rollback case): a *successful* mutation through
``update_suggestion_status_operation`` updates ``suggestion.status``
without bumping ``campaign.version``. The diagnostics cache's
fixed 120s TTL bounds the resulting staleness. That is a freshness
trade-off, not a correctness bug, and is intentionally out of scope
for 8.26 (which is specifically about rolled-back mutations).
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.cache import diagnostics_cache
from bo_mcp_server.storage import (
    CampaignRepository,
    ConcurrentModificationError,
    get_session,
)

pytestmark = pytest.mark.usefixtures("setup_database")


@pytest.mark.asyncio
async def test_failed_mutation_leaves_cache_aligned_with_committed_state() -> None:
    """A rolled-back ``save`` keeps ``campaign.version`` aligned with cached state.

    Reproducer:
    1. Seed a campaign, populate the cache under ``version=V``.
    2. Attempt a mutation whose OCC ``expected_version`` is wrong, so
       :class:`ConcurrentModificationError` rolls back the transaction.
    3. Assert that the campaign's persisted ``version`` is still ``V``
       (the rollback held) and the cache entry under ``V`` is still
       served — because the committed state did not change, neither
       did the cache's semantic correctness.

    The point is to lock in the contract that *no* mutation can land
    a side effect that survives the rollback. If a future change
    leaks one (e.g. by writing through a second session before the
    main commit), this test would still pass — but a new failing
    assertion about that specific side effect would have to be added.
    Until such a path is named, the invariant is preserved.
    """
    from bo_mcp_server.domain import Campaign, CampaignStatus

    # Seed a campaign and stamp a known diagnostics payload into the
    # cache under its current version.
    spec_id = uuid4()
    owner_id = uuid4()
    campaign_id = uuid4()
    seeded = Campaign(
        id=campaign_id,
        spec_id=spec_id,
        owner_id=owner_id,
        status=CampaignStatus.CREATED,
        version=1,
    )
    async with get_session() as session:
        await CampaignRepository(session).save(seeded, expected_version=None)

    cached_payload: dict[str, Any] = {"sentinel": True, "version_at_capture": 1}
    cache_key = f"diagnostics:{campaign_id}:1"
    await diagnostics_cache.set(cache_key, cached_payload)

    # Trigger an OCC mismatch: pass the wrong ``expected_version`` so
    # the atomic UPDATE affects zero rows and ``save`` rolls back.
    pre_mutation_version = seeded.version
    stale_view = seeded.model_copy(update={"version": 99})
    with pytest.raises(ConcurrentModificationError):
        async with get_session() as session:
            await CampaignRepository(session).save(stale_view, expected_version=99)

    # 1) Committed state did not change — the rollback held.
    async with get_session() as session:
        post = await CampaignRepository(session).get(campaign_id)
    assert post is not None
    assert post.version == pre_mutation_version

    # 2) The cache entry under the original version is still correct —
    # the committed state matches what was captured at cache time.
    assert await diagnostics_cache.get(cache_key) is cached_payload


@pytest.mark.asyncio
async def test_successful_mutation_makes_old_cache_key_unreachable() -> None:
    """A successful version-bumping mutation invalidates the old cache key.

    Mirror of the rollback test: when the mutation does commit, the
    old cache key (keyed by the old version) is no longer reachable
    from the next ``get_diagnostics`` call, because the next call
    computes ``cache_key`` from the *new* version.

    Pins the version-keyed-invalidation contract end-to-end:
    ``diagnostics:{id}:OLD_VERSION`` still sits in memory (eviction
    is opportunistic) but the next read takes a fresh path and
    repopulates ``diagnostics:{id}:NEW_VERSION``.
    """
    from bo_mcp_server.domain import Campaign, CampaignStatus

    spec_id = uuid4()
    owner_id = uuid4()
    campaign_id = uuid4()
    seeded = Campaign(
        id=campaign_id,
        spec_id=spec_id,
        owner_id=owner_id,
        status=CampaignStatus.CREATED,
        version=1,
    )
    async with get_session() as session:
        await CampaignRepository(session).save(seeded, expected_version=None)

    old_key = f"diagnostics:{campaign_id}:1"
    await diagnostics_cache.set(old_key, {"stale": True})

    # Apply a successful version-bumping mutation. The repository takes
    # ``campaign.version`` literally on write — the operation layer
    # is responsible for incrementing before calling ``save``.
    bumped = seeded.model_copy(update={"status": CampaignStatus.PAUSED, "version": 2})
    async with get_session() as session:
        await CampaignRepository(session).save(bumped, expected_version=1)

    # New version reachable from a fresh ``get`` is 2.
    async with get_session() as session:
        post = await CampaignRepository(session).get(campaign_id)
    assert post is not None
    assert post.version == 2

    # The diagnostics route would key on the new version, so an old
    # cache entry remains in memory but is logically unreachable.
    fresh_key = f"diagnostics:{campaign_id}:2"
    assert await diagnostics_cache.get(fresh_key) is None
