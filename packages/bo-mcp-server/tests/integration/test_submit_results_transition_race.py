"""Operation-level race tests for the COMPLETED transition in submit_results.

Phase 1 of ``submit_results_operation`` classifies a suggestion as
actionable (PENDING / ACCEPTED) at read time. Phase 2 then has to
atomically transition that suggestion to COMPLETED while persisting
the new result. If a concurrent caller (another submit, a manual
status update, or an admin soft-delete) moves the suggestion to a
non-actionable state between phase 1 and the phase-2 transition,
``SuggestionRepository.transition_status`` returns ``False`` and
``_resolve_suggestion_id`` raises :class:`ConcurrentModificationError`.

The friend's audit pointed out that the previous behavior — appending
a warning and still persisting an active result linked to a now-stale
suggestion — left the result row attached to a rejected / expired /
soft-deleted suggestion, which makes the audit trail incoherent and
would in some cases collide with the partial-unique-index on
``results.suggestion_id``. The new policy treats the lost transition
as a row-level concurrency conflict:

* ``atomic=True`` (default) → the operation aborts with a structured
  ``CONCURRENT_MODIFICATION`` envelope; nothing is persisted.
* ``atomic=False`` + ``continue_on_error=True`` → the affected row is
  skipped with a row-level error; the rest of the batch still commits.

Reference: classic DBMS read/write skew under READ COMMITTED
isolation (Berenson et al., 1995, "A Critique of ANSI SQL Isolation
Levels"). Application-level CAS via ``transition_status`` is the
standard remediation.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from bo_mcp_server.domain import (
    ResultSubmissionInput,
    SuggestionStatus,
)
from bo_mcp_server.errors import ErrorCode


def _to_result_inputs(rows: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in rows]


async def _build_campaign_with_suggestions(
    batch_size: int = 2,
) -> tuple[str, list[dict[str, Any]], str]:
    """Create a campaign and generate ``batch_size`` PENDING suggestions."""
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Submit transition race",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    generated = await generate_suggestions(campaign_id, batch_size=batch_size)
    return campaign_id, generated["suggestions"], owner_id


def _patch_transition_to_fail_for(
    monkeypatch: pytest.MonkeyPatch, suggestion_ids: set[str]
) -> None:
    """Force ``transition_status`` to report ``False`` for the named IDs.

    Simulates a concurrent caller that already completed / rejected /
    soft-deleted the row between phase-1 classification and the
    phase-2 atomic UPDATE. Other suggestions still transition
    normally so the ``continue_on_error`` path can be exercised.
    """
    from bo_mcp_server.storage.repositories import SuggestionRepository

    original = SuggestionRepository.transition_status

    async def racing_transition(
        self: SuggestionRepository,
        sid: UUID,
        from_status: SuggestionStatus | tuple[SuggestionStatus, ...],
        to_status: SuggestionStatus,
    ) -> bool:
        if str(sid) in suggestion_ids:
            return False
        return await original(self, sid, from_status, to_status)

    monkeypatch.setattr(SuggestionRepository, "transition_status", racing_transition)


def _patch_get_to_simulate_soft_delete(
    monkeypatch: pytest.MonkeyPatch, suggestion_ids: set[str]
) -> None:
    """Force ``SuggestionRepository.get`` to act as if the row is soft-deleted.

    Simulates an admin soft-delete landing between phase 1 (which saw
    the suggestion as actionable) and phase 2's
    ``_resolve_suggestion_id`` (which re-reads it). Default reads
    filter ``deleted_at IS NULL``, so the production behaviour after
    a real soft-delete is exactly this: ``get()`` returns ``None``.
    """
    from bo_mcp_server.storage.repositories import SuggestionRepository

    original = SuggestionRepository.get

    async def disappearing_get(self, sid, *, include_deleted: bool = False):
        if str(sid) in suggestion_ids and not include_deleted:
            return None
        return await original(self, sid, include_deleted=include_deleted)

    monkeypatch.setattr(SuggestionRepository, "get", disappearing_get)


class TestSubmitResultsTransitionRace:
    """The COMPLETED transition race is mapped to a concurrency conflict."""

    @pytest.mark.asyncio
    async def test_atomic_mode_aborts_with_cmr_envelope_and_persists_nothing(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict mode treats a lost transition as a full-batch conflict."""
        _ = setup_database
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import (
            list_suggestions_operation,
        )
        from bo_mcp_server.operations.submit_results import (
            submit_results_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        _patch_transition_to_fail_for(monkeypatch, {suggestions[0]["id"]})

        rows = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": 0.5},
            }
            for s in suggestions
        ]
        response = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            source="api",
            atomic=True,
        )

        assert response["success"] is False
        assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
        # No result rows committed.
        listed_results = await list_results_operation(campaign_id=campaign_id)
        assert listed_results["results"] == []
        # The non-racing suggestion stays PENDING because the whole
        # transaction rolled back — the second row never reached its
        # transition UPDATE.
        listed_suggestions = await list_suggestions_operation(campaign_id=campaign_id)
        statuses = {s["status"] for s in listed_suggestions["suggestions"]}
        assert statuses == {SuggestionStatus.PENDING.value}

    @pytest.mark.asyncio
    async def test_continue_on_error_skips_only_the_racing_row(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``continue_on_error`` records the rest of the batch and skips the conflict."""
        _ = setup_database
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import (
            submit_results_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        racing_id = suggestions[0]["id"]
        winning_id = suggestions[1]["id"]
        _patch_transition_to_fail_for(monkeypatch, {racing_id})

        rows = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": 0.5},
            }
            for s in suggestions
        ]
        response = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            source="api",
            atomic=False,
            continue_on_error=True,
        )

        # The non-racing row committed.
        assert len(response["result_ids"]) == 1, response
        # The row-level error names the failing suggestion.
        assert any(racing_id in str(e) for e in response.get("errors", [])), response
        # Storage: exactly one result, linked to the winning suggestion.
        listed_results = await list_results_operation(campaign_id=campaign_id)
        persisted = listed_results["results"]
        assert len(persisted) == 1
        assert persisted[0]["suggestion_id"] == winning_id


class TestSubmitResultsSoftDeleteBetweenPhases:
    """A phase-1-actionable suggestion that disappears mid-submit is a conflict.

    The friend's audit pointed out that the previous
    ``_resolve_suggestion_id`` could not distinguish "caller supplied
    a bad id" (warn, persist as free-floating) from "phase 1 saw the
    id as actionable but it disappeared before phase 2" (a real race
    that must be surfaced as a conflict). With the
    ``actionable_ids`` set threaded through, ``get() -> None`` for
    an id in that set now raises ``ConcurrentModificationError`` and
    is routed per submission mode — never silently demoted to a
    free-floating result.
    """

    @pytest.mark.asyncio
    async def test_atomic_mode_aborts_when_actionable_id_was_soft_deleted(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Strict mode rejects the whole batch and persists nothing."""
        _ = setup_database
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import (
            submit_results_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        _patch_get_to_simulate_soft_delete(monkeypatch, {suggestions[0]["id"]})

        rows = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": 0.5},
            }
            for s in suggestions
        ]
        response = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            source="api",
            atomic=True,
        )

        assert response["success"] is False
        assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
        listed = await list_results_operation(campaign_id=campaign_id)
        assert listed["results"] == []

    @pytest.mark.asyncio
    async def test_continue_on_error_skips_the_vanished_row(
        self, setup_database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``continue_on_error`` skips the vanished id and commits the rest.

        Crucially, the surviving result is NOT silently degraded to
        free-floating — the row is *skipped* with a row-level error
        so callers can distinguish "we couldn't find your id" from
        "your id was actionable but vanished mid-submit". The latter
        is a concurrency conflict and requires explicit caller
        acknowledgement via ``continue_on_error``.
        """
        _ = setup_database
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import (
            submit_results_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        vanished_id = suggestions[0]["id"]
        winning_id = suggestions[1]["id"]
        _patch_get_to_simulate_soft_delete(monkeypatch, {vanished_id})

        rows = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": 0.5},
            }
            for s in suggestions
        ]
        response = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            source="api",
            atomic=False,
            continue_on_error=True,
        )

        assert len(response["result_ids"]) == 1, response
        assert any(vanished_id in str(e) for e in response.get("errors", [])), response
        listed = await list_results_operation(campaign_id=campaign_id)
        persisted = listed["results"]
        assert len(persisted) == 1
        # The surviving result is linked to the winning suggestion,
        # not silently degraded to free-floating.
        assert persisted[0]["suggestion_id"] == winning_id
        # And there is no free-floating row for the vanished id.
        assert all(p["suggestion_id"] is not None for p in persisted)
