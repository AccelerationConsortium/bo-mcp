"""``update_suggestion_status`` surfaces a CMR envelope on a soft-deleted row.

TODO 8.11 follow-up: previously the storage layer silently no-op'd a
save against a tombstoned suggestion, so a race between an active read
and a concurrent soft-delete could return ``success=True`` to the
caller even though the persisted row never changed. ``SuggestionRepository.save``
now raises :class:`ConcurrentModificationError`, and the operation
catches it to return a structured ``CONCURRENT_MODIFICATION`` envelope.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from bo_mcp_server.domain import (
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.errors import ErrorCode
from bo_mcp_server.operations.update_suggestion_status import (
    update_suggestion_status_operation,
)
from bo_mcp_server.storage import (
    CampaignRepository,
    SuggestionRepository,
    UserRepository,
    get_session,
)
from bo_mcp_server.storage.models import CampaignSpecModel


async def _seed_pending_suggestion() -> Suggestion:
    """Create the user + spec + campaign + a PENDING suggestion in storage."""
    from bo_mcp_server.domain import Campaign, CampaignStatus

    user = User(
        name="Owner",
        email=f"owner-{uuid4()}@example.com",
        api_key_hash=f"hash-{uuid4()}",
    )
    spec_id = uuid4()
    async with get_session() as session:
        await UserRepository(session).save(user)
        session.add(
            CampaignSpecModel(
                id=str(spec_id),
                name="Update-status soft-delete test",
                parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
                objectives_json='[{"name":"y","direction":"minimize"}]',
            )
        )
        await session.flush()
        campaign = Campaign(
            spec_id=spec_id,
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
            version=1,
            iteration=0,
        )
        await CampaignRepository(session).save(campaign)

    suggestion = Suggestion(
        campaign_id=campaign.id,
        parameter_values={"x": 0.5},
        provenance=SuggestionProvenance(iteration=1, batch_index=0),
        status=SuggestionStatus.PENDING,
    )
    async with get_session() as session:
        await SuggestionRepository(session).save(suggestion)
    return suggestion


@pytest.mark.asyncio
async def test_update_status_returns_concurrent_modification_when_row_was_soft_deleted(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Soft-deleting between read and save surfaces a structured CMR envelope.

    Simulates the race the friend's audit flagged: the operation
    reads an active row, then a concurrent soft-delete lands before
    the save commits. Without the new raise, the save would silently
    no-op and the operation would report success even though no
    state changed. With the fix the save raises
    :class:`ConcurrentModificationError` and the operation routes
    it through the existing ``CONCURRENT_MODIFICATION`` envelope.

    The race is modelled deterministically by monkey-patching
    ``SuggestionRepository.save`` to soft-delete the row *before*
    delegating to the real save — there is no SQLite primitive
    that lets us interleave two real transactions inside the same
    test, but the post-read / pre-save window is exactly what the
    monkey-patch reproduces.
    """
    _ = setup_database
    suggestion = await _seed_pending_suggestion()

    original_transition = SuggestionRepository.transition_status

    async def racing_transition(self, sid, from_status, to_status):
        # Soft-delete on a side session before the operation's
        # transition UPDATE runs, then delegate. The conditional
        # ``WHERE … AND deleted_at IS NULL`` should see the
        # tombstone and report ``rowcount == 0``.
        if sid == suggestion.id:
            async with get_session() as side:
                await SuggestionRepository(side).delete(sid)
        return await original_transition(self, sid, from_status, to_status)

    monkeypatch.setattr(SuggestionRepository, "transition_status", racing_transition)

    response = await update_suggestion_status_operation(
        suggestion_id=str(suggestion.id),
        status=SuggestionStatus.ACCEPTED.value,
    )

    assert response["success"] is False
    assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
    # The operation must NOT report a status change that did not happen.
    assert response["previous_status"] == SuggestionStatus.PENDING.value
    assert response["status"] == SuggestionStatus.PENDING.value
