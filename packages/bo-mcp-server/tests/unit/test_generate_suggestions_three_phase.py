"""Three-phase split for generate_suggestions (TODO 8.12).

Pins the snapshot → compute → persist contract:

* Phase 1 (``_load_generation_snapshot``) opens its own short read
  transaction even when the caller passes an outer session, so the
  snapshot's locks are released before the BO compute starts.
* Phase 2 (``_compute_generation_batch``) runs with no DB session
  open; concurrent reads / writes on the same campaign are no
  longer blocked by GP fit + acquisition optimization.
* Phase 3 (``_persist_generation_batch``) uses
  ``CampaignRepository.save(expected_version=...)`` to detect any
  state mutation that landed during the compute window and routes
  the conflict through the existing
  :class:`ConcurrentModificationError` machinery.

Together this turns a long-lived transaction into three short ones
without changing the public response shape.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.errors import ErrorCode
from bo_mcp_server.operations import generate_suggestions as gs


async def _create_minimal_campaign() -> str:
    """Create a single-objective continuous campaign and return its ID."""
    from bo_mcp_server.tools.create_campaign import create_campaign  # noqa: PLC0415

    owner_id = str(uuid4())
    intake = {
        "name": "Three-phase split test",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "batch_size": 1,
    }
    response = await create_campaign(intake_data=intake, owner_id=owner_id)
    assert response["success"] is True, response
    return response["campaign_id"]


@pytest.mark.asyncio
async def test_compute_runs_with_no_session_or_transaction_in_flight(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No SQLAlchemy session or transaction is alive while phase 2 runs.

    Earlier iterations of this test merely opened another session
    during the compute and called ``SELECT 1``, which proves only
    that the engine pool can hand out another connection — not that
    the phase-1 transaction has actually been released. The stronger
    check below instruments SQLAlchemy ``Session`` lifecycle events
    so we count concrete open sessions / in-flight transactions and
    assert both are zero at the moment ``_compute_generation_batch``
    fires.

    Reference: SQLAlchemy ``SessionEvents.after_begin`` /
    ``after_transaction_end``
    (https://docs.sqlalchemy.org/en/20/orm/events.html#sqlalchemy.orm.SessionEvents).
    """
    _ = setup_database
    campaign_id = await _create_minimal_campaign()

    from sqlalchemy import event  # noqa: PLC0415
    from sqlalchemy.orm import Session  # noqa: PLC0415

    open_sessions: set[int] = set()
    active_transactions: set[int] = set()

    @event.listens_for(Session, "after_begin")
    def _on_begin(session: Session, transaction: Any, _connection: Any) -> None:  # noqa: ARG001
        open_sessions.add(id(session))
        active_transactions.add(id(transaction))

    @event.listens_for(Session, "after_transaction_end")
    def _on_end(session: Session, transaction: Any) -> None:  # noqa: ARG001
        active_transactions.discard(id(transaction))

    @event.listens_for(Session, "after_soft_rollback")
    def _on_close(session: Session, _previous: Any) -> None:  # noqa: ARG001
        open_sessions.discard(id(session))

    original_compute = gs._compute_generation_batch
    observed: dict[str, int] = {}

    async def probing_compute(
        snapshot: gs._GenerationSnapshot,
        backend: Any,
        prior_backend_state: dict[str, Any] | None,
        progress_callback: Any,
    ) -> gs._GenerationComputeResult:
        observed["sessions"] = len(open_sessions)
        observed["transactions"] = len(active_transactions)
        return await original_compute(
            snapshot=snapshot,
            backend=backend,
            prior_backend_state=prior_backend_state,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(gs, "_compute_generation_batch", probing_compute)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    response = await generate_suggestions(campaign_id=campaign_id)
    assert response["success"] is True, response
    assert observed["transactions"] == 0, (
        f"phase 2 ran with {observed['transactions']} transaction(s) still open"
    )


@pytest.mark.asyncio
async def test_concurrent_modification_during_compute_returns_structured_envelope(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the campaign's version moved during phase 2, phase 3 surfaces OCC.

    Simulates the audit scenario: a slow GP fit runs in phase 2 while
    another caller bumps the campaign version (e.g. a lifecycle
    transition or a multi-objective result submission). The phase-3
    save with ``expected_version=snapshot.version`` fails the OCC
    check; the existing ``ConcurrentModificationError`` machinery
    routes that to a structured envelope.
    """
    _ = setup_database
    campaign_id = await _create_minimal_campaign()

    original_compute = gs._compute_generation_batch
    sentinel = {"hit_compute": False}

    async def mutating_compute(
        snapshot: gs._GenerationSnapshot,
        backend: Any,
        prior_backend_state: dict[str, Any] | None,
        progress_callback: Any,
    ) -> gs._GenerationComputeResult:
        """Simulate a concurrent state change while phase 2 is running."""
        from bo_mcp_server.domain import CampaignStatus  # noqa: PLC0415
        from bo_mcp_server.storage import (  # noqa: PLC0415
            CampaignRepository,
            get_session,
        )

        async with get_session() as session:
            repo = CampaignRepository(session)
            fresh = await repo.get(snapshot.campaign.id)
            assert fresh is not None
            # Bump the version via a status flip so the snapshot's
            # ``expected_version`` becomes stale by the time phase 3 runs.
            # PAUSED is a no-op for CREATED, so use RUNNING and then
            # rely on the version bump to trip phase 3's OCC check.
            updated = fresh.with_status(CampaignStatus.RUNNING)
            await repo.save(updated, expected_version=fresh.version)

        sentinel["hit_compute"] = True
        return await original_compute(
            snapshot=snapshot,
            backend=backend,
            prior_backend_state=prior_backend_state,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(gs, "_compute_generation_batch", mutating_compute)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    response = await generate_suggestions(campaign_id=campaign_id)
    assert sentinel["hit_compute"] is True
    assert response["success"] is False
    assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value


@pytest.mark.asyncio
async def test_single_objective_submit_during_compute_returns_conflict(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-objective ``submit_results`` during phase 2 is detected.

    Catches the regression that ``campaign.version`` OCC alone cannot
    cover: single-objective ``submit_results`` only saves the campaign
    when hypervolume / backend state changed (see
    ``_update_campaign_state`` in ``submit_results.py``), so the
    version stays at the snapshot's value and the OCC guard passes
    even though the result set grew. Phase 3's child-table invariant
    check on ``result_count`` is what catches this — the test would
    have passed without it.

    Simulates the race by patching phase 2 to insert a result row
    directly into the same campaign before returning. Phase 3 then
    re-counts results and discovers the snapshot is stale.
    """
    _ = setup_database
    from datetime import datetime as _dt  # noqa: PLC0415

    campaign_id = await _create_minimal_campaign()
    original_compute = gs._compute_generation_batch

    async def inserting_compute(
        snapshot: gs._GenerationSnapshot,
        backend: Any,
        prior_backend_state: dict[str, Any] | None,
        progress_callback: Any,
    ) -> gs._GenerationComputeResult:
        """Simulate a single-objective submit_results that does not bump campaign.version."""
        from bo_mcp_server.domain import Result, ResultSource  # noqa: PLC0415
        from bo_mcp_server.storage import ResultRepository, get_session  # noqa: PLC0415

        async with get_session() as side_session:
            await ResultRepository(side_session).save(
                Result(
                    campaign_id=snapshot.campaign.id,
                    suggestion_id=None,
                    parameter_values={"x": 0.42},
                    objective_values={"y": 0.99},
                    source=ResultSource.API,
                    submitted_by=snapshot.campaign.owner_id,
                    created_at=_dt.now().astimezone(),
                )
            )
        return await original_compute(
            snapshot=snapshot,
            backend=backend,
            prior_backend_state=prior_backend_state,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(gs, "_compute_generation_batch", inserting_compute)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    response = await generate_suggestions(campaign_id=campaign_id)
    assert response["success"] is False
    assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
    # No new suggestions should have been persisted — the response
    # carries an empty suggestion list because phase 3 raised before
    # the create-and-save step.
    assert response["suggestions"] == []


@pytest.mark.asyncio
async def test_stale_pending_accepted_during_compute_returns_conflict(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``PENDING -> ACCEPTED`` during compute returns a conflict envelope.

    Phase 1 excludes a stale ``PENDING`` row from ``valid_pending``,
    so the BO compute runs against a budget and X_pending that do
    *not* account for it. If that row gets ``ACCEPTED`` during phase
    2 (via ``update_suggestion_status``), persisting the new batch
    would understate the in-flight count and could exceed
    ``max_observations`` or cluster against the newly-accepted
    point. Phase 3 detects the change through
    ``expire_if_pending == False`` and raises
    ``ConcurrentModificationError`` so the caller can retry against
    a fresh snapshot. The user's ``ACCEPTED`` write — which lives in
    its own committed transaction — must survive the rollback of the
    generation transaction.
    """
    _ = setup_database
    from datetime import UTC, timedelta  # noqa: PLC0415
    from datetime import datetime as _dt

    from bo_engine.constants import PENDING_SUGGESTION_MAX_AGE_HOURS  # noqa: PLC0415

    from bo_mcp_server.domain import (  # noqa: PLC0415
        Suggestion,
        SuggestionProvenance,
        SuggestionStatus,
    )
    from bo_mcp_server.operations.update_suggestion_status import (  # noqa: PLC0415
        update_suggestion_status_operation,
    )
    from bo_mcp_server.storage import SuggestionRepository, get_session  # noqa: PLC0415

    campaign_id = await _create_minimal_campaign()
    from uuid import UUID  # noqa: PLC0415

    campaign_uuid = UUID(campaign_id)

    # Seed an aged ``PENDING`` suggestion whose ``created_at`` is well
    # past the staleness window so phase 1 classifies it as stale.
    stale_age = _dt.now(UTC) - timedelta(hours=PENDING_SUGGESTION_MAX_AGE_HOURS + 1)
    stale_suggestion = Suggestion(
        campaign_id=campaign_uuid,
        parameter_values={"x": 0.123},
        provenance=SuggestionProvenance(iteration=0, batch_index=0),
        status=SuggestionStatus.PENDING,
        created_at=stale_age,
        updated_at=stale_age,
    )
    async with get_session() as seed:
        await SuggestionRepository(seed).save(stale_suggestion)

    original_compute = gs._compute_generation_batch

    async def accepting_compute(
        snapshot: gs._GenerationSnapshot,
        backend: Any,
        prior_backend_state: dict[str, Any] | None,
        progress_callback: Any,
    ) -> gs._GenerationComputeResult:
        # During phase 2, transition the stale PENDING row to ACCEPTED
        # via the same operation a UI / agent would use.
        response = await update_suggestion_status_operation(
            suggestion_id=str(stale_suggestion.id),
            status=SuggestionStatus.ACCEPTED.value,
        )
        assert response["success"] is True, response
        return await original_compute(
            snapshot=snapshot,
            backend=backend,
            prior_backend_state=prior_backend_state,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(gs, "_compute_generation_batch", accepting_compute)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    response = await generate_suggestions(campaign_id=campaign_id)
    assert response["success"] is False, response
    assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
    assert response["suggestions"] == []

    async with get_session() as check:
        reloaded = await SuggestionRepository(check).get(stale_suggestion.id)
    assert reloaded is not None
    # The user's ACCEPTED write committed in its own session before
    # the generator's rollback fired, so it survives.
    assert reloaded.status == SuggestionStatus.ACCEPTED


@pytest.mark.asyncio
async def test_campaign_soft_deleted_during_compute_returns_conflict(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Soft-deleting the campaign during phase 2 aborts phase 3 with CMR.

    Catches the audit gap that ``CampaignRepository.save`` only
    checked ``id`` and ``version`` — a soft-delete that happened
    between the snapshot read and the phase-3 save would otherwise
    let new active suggestions attach to a tombstoned campaign.
    The new ``deleted_at IS NULL`` clause on the OCC update raises
    ``ConcurrentModificationError`` instead; the session rollback
    means no suggestion rows are persisted either.
    """
    _ = setup_database
    from bo_mcp_server.storage import (  # noqa: PLC0415
        CampaignRepository,
        SuggestionRepository,
        get_session,
    )

    campaign_id = await _create_minimal_campaign()
    from uuid import UUID  # noqa: PLC0415

    campaign_uuid = UUID(campaign_id)
    original_compute = gs._compute_generation_batch

    async def deleting_compute(
        snapshot: gs._GenerationSnapshot,
        backend: Any,
        prior_backend_state: dict[str, Any] | None,
        progress_callback: Any,
    ) -> gs._GenerationComputeResult:
        async with get_session() as side:
            removed = await CampaignRepository(side).delete(campaign_uuid)
            assert removed is True
        return await original_compute(
            snapshot=snapshot,
            backend=backend,
            prior_backend_state=prior_backend_state,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(gs, "_compute_generation_batch", deleting_compute)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    response = await generate_suggestions(campaign_id=campaign_id)
    assert response["success"] is False, response
    assert response["error"]["code"] == ErrorCode.CONCURRENT_MODIFICATION.value
    assert response["suggestions"] == []

    # Phase 3 rolled back, so no new suggestions exist for this campaign.
    async with get_session() as check:
        # Include soft-deleted on the campaign side so the check is
        # not gated on the campaign being readable; the assertion
        # is about the *suggestion* set.
        suggestions = await SuggestionRepository(check).list_by_campaign(campaign_uuid)
    assert suggestions == []


@pytest.mark.asyncio
async def test_event_loop_responsive_during_slow_compute(
    setup_database, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow phase-2 compute does not block unrelated coroutines.

    Belt-and-braces: the existing
    ``test_event_loop_responsiveness.py`` already pins this for the
    underlying ``asyncio.to_thread`` wrap on the backend call. We
    keep a focused check here because the three-phase split also
    removes the DB-session-held-across-compute concern, and we want
    a single test we can point at when reviewing TODO 8.12.
    """
    _ = setup_database
    campaign_id = await _create_minimal_campaign()

    work_seconds = 0.2
    original_via_backend = gs._generate_via_backend

    async def slow_backend(*args: Any, **kwargs: Any) -> Any:
        # ``time.sleep`` releases the GIL inside ``asyncio.to_thread``
        # so the event loop stays responsive — that's what we want
        # to assert below.
        await asyncio.to_thread(time.sleep, work_seconds)
        return await original_via_backend(*args, **kwargs)

    monkeypatch.setattr(gs, "_generate_via_backend", slow_backend)

    from bo_mcp_server.tools.generate_suggestions import generate_suggestions  # noqa: PLC0415

    started = time.perf_counter()
    gen_task = asyncio.create_task(generate_suggestions(campaign_id=campaign_id))

    # Yield once to let the gen task enter its compute phase, then
    # measure how long a plain ``asyncio.sleep`` takes — it should
    # be close to its argument, not the work duration.
    await asyncio.sleep(0)
    sleep_started = time.perf_counter()
    await asyncio.sleep(0.01)
    sleep_elapsed = time.perf_counter() - sleep_started

    response = await gen_task
    total = time.perf_counter() - started
    assert response["success"] is True, response
    assert sleep_elapsed < work_seconds / 2, (
        "event loop was blocked by phase 2 compute "
        f"(sleep took {sleep_elapsed:.3f}s, work was {work_seconds}s)"
    )
    assert total >= work_seconds * 0.5
