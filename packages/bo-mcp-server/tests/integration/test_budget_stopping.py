"""Integration tests for budget / convergence-based stopping.

The server-side ``generate_suggestions`` operation must respect three
``OptimizationSpec`` budgets:

* ``max_iterations`` — the legacy field that historically was dropped at the
  domain-to-engine boundary, so unattended agent loops could blow through
  their iteration cap silently;
* ``max_observations`` — a new per-result cap independent of iteration
  grouping;
* ``convergence_tolerance`` — the existing
  :func:`detect_single_objective_convergence` threshold, now consumed by
  the suggestion entry point rather than only the diagnostics layer.

Each stop emits a structured envelope (``code`` ∈ ``E012``, ``E013``)
with ``details["next_action_recommendation"] == "terminate_campaign"`` so
agent loops can branch without parsing free-form messages.
"""

from __future__ import annotations

from typing import Any

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from tests.factories import seed_owner

pytestmark = pytest.mark.usefixtures("setup_database")


def _to_result_inputs(results: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


@pytest.mark.asyncio
async def test_max_iterations_stops_after_budget() -> None:
    """``max_iterations=2`` returns BUDGET_EXCEEDED on the third request.

    The campaign runs two normal iterations, then the third call to
    ``generate_suggestions`` short-circuits without producing new
    suggestions.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Iteration Budget",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_iterations": 2,
        "initial_design_size": 2,
        "random_seed": 42,
    }
    create_result = await create_campaign(intake, owner_id)
    assert create_result["success"] is True
    campaign_id = create_result["campaign_id"]

    # Two normal cycles.
    for cycle in range(2):
        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True, f"Cycle {cycle} unexpectedly errored: {gen['errors']}"
        results = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": float(s["parameter_values"]["x"])},
            }
            for s in gen["suggestions"]
        ]
        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(results),
            submitted_by=owner_id,
        )

    # Third call must short-circuit with BUDGET_EXCEEDED.
    blocked = await generate_suggestions(campaign_id)
    assert blocked["success"] is False
    assert blocked["suggestions"] == []
    error = blocked["error"]
    assert error["code"] == "E012"  # BUDGET_EXCEEDED
    details = error["details"]
    assert details["stopping_reason"] == "budget_exceeded_iterations"
    assert details["next_action_recommendation"] == "terminate_campaign"


@pytest.mark.asyncio
async def test_max_observations_stops_at_cap() -> None:
    """Reaching ``max_observations`` blocks the next suggestion request.

    Observation cap is decoupled from iteration grouping, so a single
    multi-result submit can trip it without requiring a second
    ``generate_suggestions`` call.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Observation Budget",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "random_seed": 42,
        "batch_size": 2,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    assert gen["success"] is True
    results = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
        }
        for s in gen["suggestions"]
    ]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(results),
        submitted_by=owner_id,
    )

    # Cap of 2 hit on the very next call.
    blocked = await generate_suggestions(campaign_id)
    assert blocked["success"] is False
    error = blocked["error"]
    assert error["code"] == "E012"  # BUDGET_EXCEEDED
    assert error["details"]["stopping_reason"] == "budget_exceeded_observations"
    assert error["details"]["n_observations"] == 2
    assert error["details"]["next_action_recommendation"] == "terminate_campaign"


@pytest.mark.asyncio
async def test_pending_suggestions_count_against_max_observations(caplog) -> None:
    """Generated-but-unsubmitted suggestions consume budget reservations.

    Two invariants:
    * When ``existing + pending < max_observations`` the next generate call
      clamps its batch to the slack remaining (verified via the
      ``Clamping batch_size`` log line emitted by the operation).
    * When ``existing + pending >= max_observations`` the next generate call
      short-circuits with ``BUDGET_EXCEEDED`` carrying ``n_pending`` so the
      agent can see which slot is reserved by an in-flight experiment.
    """
    import logging

    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Pending Counts Toward Budget",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 3,
        "initial_design_size": 2,
        "random_seed": 42,
        "batch_size": 2,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    # First batch of 2 suggestions; submit only the first to leave the
    # second PENDING. Now: 1 submitted + 1 pending = 2 reserved out of 3.
    gen1 = await generate_suggestions(campaign_id)
    first, _ = gen1["suggestions"]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )

    # Generate again. Remaining = 3 - 1 - 1 = 1, so a batch_size=2 request
    # must be clamped to 1. We verify via the operation's own log line --
    # the engine's Sobol continuation can produce fewer than the clamped
    # count, which is a separate concern (and is harmless for the cap).
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="bo_mcp_server.operations.generate_suggestions"):
        await generate_suggestions(campaign_id)
    clamp_logs = [r.getMessage() for r in caplog.records if "Clamping batch_size" in r.getMessage()]
    assert clamp_logs, "Generation must clamp batch when pending consumes budget"
    assert "n_pending=1" in clamp_logs[0]

    # Force the budget to its limit with a freestanding result. Whether it
    # lands depends on the backend: BoTorch's Sobol continuation deduped the
    # clamped generate to zero new pending (slack left, submit accepted →
    # 2 stored + 1 pending); BayBE returned a fresh point (1 stored +
    # 2 pending = cap, so the overflow guard rejects the import). Both are
    # correct budget enforcement — the hard stop below must hold either way.
    freestanding = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [{"parameter_values": {"x": 0.99}, "objective_values": {"y": 0.42}}]
        ),
        submitted_by=owner_id,
    )

    blocked = await generate_suggestions(campaign_id)
    assert blocked["success"] is False
    assert blocked["error"]["code"] == "E012"  # BUDGET_EXCEEDED
    details = blocked["error"]["details"]
    assert details["stopping_reason"] == "budget_exceeded_observations"
    # The budget invariant is what matters: stored + pending has reached
    # the cap, regardless of how the backend split the reservation.
    assert details["n_pending"] >= 1
    assert details["n_observations"] == (2 if freestanding["success"] else 1)
    assert details["n_observations"] + details["n_pending"] >= 3


@pytest.mark.asyncio
async def test_submit_results_rejects_overflow_in_atomic_mode() -> None:
    """Atomic submissions exceeding ``max_observations`` are rejected wholesale.

    With cap=2 and 1 stored result, submitting two more results in atomic
    mode must reject the whole batch — no row can land or the cap is broken.
    """
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Atomic Submit Budget",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 1,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    # Seed one observation.
    first_gen = await generate_suggestions(campaign_id)
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first_gen["suggestions"][0]["id"],
                    "parameter_values": first_gen["suggestions"][0]["parameter_values"],
                    "objective_values": {
                        "y": float(first_gen["suggestions"][0]["parameter_values"]["x"])
                    },
                }
            ]
        ),
        submitted_by=owner_id,
    )

    # Generate two more suggestions (clamp permits this since pending=0).
    second_gen = await generate_suggestions(campaign_id, batch_size=2)
    # Only 1 slot remains so clamp shrinks the batch to 1; force-construct a
    # second freestanding submission that pushes us over.
    overflow_rows = [
        {
            "suggestion_id": second_gen["suggestions"][0]["id"],
            "parameter_values": second_gen["suggestions"][0]["parameter_values"],
            "objective_values": {"y": 0.5},
        },
        # Second row has no suggestion id (server accepts free-floating rows)
        # and would push the campaign to 3 results.
        {
            "parameter_values": {"x": 0.7},
            "objective_values": {"y": 0.7},
        },
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(overflow_rows),
        submitted_by=owner_id,
        atomic=True,
    )
    assert result["success"] is False
    joined = " ".join(result["errors"]).lower()
    assert "max_observations" in joined
    listed = await list_results_operation(campaign_id=campaign_id)
    assert listed["total_count"] == 1, (
        "Atomic mode must roll back; only the seed result must persist"
    )


@pytest.mark.asyncio
async def test_submit_results_keeps_in_budget_rows_in_non_atomic_mode() -> None:
    """Non-atomic submission accepts rows up to the cap, rejects the rest."""
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Non-Atomic Submit Budget",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 1,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    seed_gen = await generate_suggestions(campaign_id)
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": seed_gen["suggestions"][0]["id"],
                    "parameter_values": seed_gen["suggestions"][0]["parameter_values"],
                    "objective_values": {
                        "y": float(seed_gen["suggestions"][0]["parameter_values"]["x"])
                    },
                }
            ]
        ),
        submitted_by=owner_id,
    )

    follow = await generate_suggestions(campaign_id, batch_size=1)
    rows = [
        {
            "suggestion_id": follow["suggestions"][0]["id"],
            "parameter_values": follow["suggestions"][0]["parameter_values"],
            "objective_values": {"y": 0.5},
        },
        # Overflow row -- must be rejected without taking down the whole batch.
        {"parameter_values": {"x": 0.9}, "objective_values": {"y": 0.9}},
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(rows),
        submitted_by=owner_id,
        atomic=False,
        continue_on_error=True,
        verbosity="detailed",
    )
    assert result["success"] is True
    assert isinstance(result["partial_results"][1], dict)
    assert "max_observations" in result["partial_results"][1]["error"].lower()
    listed = await list_results_operation(campaign_id=campaign_id)
    assert listed["total_count"] == 2  # seed + in-budget row


@pytest.mark.asyncio
async def test_submit_protects_pending_reservation_atomic() -> None:
    """Free-floating rows cannot consume a slot reserved by a pending suggestion.

    Scenario reported by review: ``max_observations=3``, two suggestions
    generated, one submitted (existing=1, pending=1, slack=1). A user then
    submits two free-floating manual rows in atomic mode. The submission
    must reject the batch because the second row would either overshoot the
    cap or evict the legitimate pending experiment's reserved slot.
    """
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Pending Reservation Atomic",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 3,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 7,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, second = gen["suggestions"]

    # Submit only the first; ``second`` stays PENDING and reserves slot #3.
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )

    # Attempt to flood the campaign with two free-floating rows. existing=1,
    # pending_reserved=1, remaining=1 -> second row must be rejected.
    overflow_rows = [
        {"parameter_values": {"x": 0.4}, "objective_values": {"y": 0.4}},
        {"parameter_values": {"x": 0.6}, "objective_values": {"y": 0.6}},
    ]
    blocked = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(overflow_rows),
        submitted_by=owner_id,
        atomic=True,
    )
    assert blocked["success"] is False
    joined = " ".join(blocked["errors"]).lower()
    assert "max_observations" in joined
    assert "pending_reserved=1" in joined

    # Cap is intact -> only the seed result is stored.
    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 1

    # Crucially: the original pending suggestion can still be submitted. The
    # cap calculation accounts for its reserved slot, so this submission
    # consumes the reservation rather than competing with free-floating rows.
    final = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": second["id"],
                    "parameter_values": second["parameter_values"],
                    "objective_values": {"y": float(second["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
        atomic=True,
    )
    assert final["success"] is True


@pytest.mark.asyncio
async def test_submit_protects_pending_reservation_non_atomic() -> None:
    """Non-atomic mode accepts free-floating rows up to unreserved slack only."""
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Pending Reservation Non-Atomic",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 3,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 7,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, _ = gen["suggestions"]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )

    rows = [
        {"parameter_values": {"x": 0.4}, "objective_values": {"y": 0.4}},
        {"parameter_values": {"x": 0.6}, "objective_values": {"y": 0.6}},
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(rows),
        submitted_by=owner_id,
        atomic=False,
        continue_on_error=True,
        verbosity="detailed",
    )
    assert result["success"] is True
    # Index 1 should be rejected with a budget error; index 0 commits.
    assert isinstance(result["partial_results"][1], dict)
    assert "max_observations" in result["partial_results"][1]["error"].lower()
    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 2  # seed + first free-floating row


@pytest.mark.asyncio
async def test_submit_pending_suggestion_consumes_its_reservation() -> None:
    """Submitting a pending suggestion does not double-count its reservation.

    With cap=2, 0 existing, 2 pending, submitting *both* pending rows in a
    single atomic batch must succeed: each consumes the slot it was
    already reserving.
    """
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Pending Self-Consume",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 7,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    rows = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
        }
        for s in gen["suggestions"]
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(rows),
        submitted_by=owner_id,
        atomic=True,
    )
    assert result["success"] is True
    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 2


@pytest.mark.asyncio
async def test_stale_pending_suggestion_does_not_budget_reject_free_floating_submit() -> None:
    """An aged-out PENDING row releases its budget slot for a free-floating submit.

    Submit and generate must agree on which pending rows still hold
    budget slots: generate excludes stale PENDING rows (older than
    ``PENDING_SUGGESTION_MAX_AGE_HOURS`` — the next generate call
    expires them), so submit must not reject a free-floating result for
    a slot that is already logically free.
    """
    from datetime import timedelta

    from sqlalchemy import update

    from bo_engine.constants import PENDING_SUGGESTION_MAX_AGE_HOURS
    from bo_mcp_server.domain.utils import utcnow
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.storage import get_session
    from bo_mcp_server.storage.models import SuggestionModel
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Stale Pending Releases Slot",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 1,
        "initial_design_size": 1,
        "batch_size": 1,
        "random_seed": 7,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    (suggestion,) = gen["suggestions"]

    free_floating = [{"parameter_values": {"x": 0.42}, "objective_values": {"y": 0.42}}]

    # Fresh PENDING row: the reservation is honored, the submit is rejected.
    blocked = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(free_floating),
        submitted_by=owner_id,
        atomic=True,
    )
    assert blocked["success"] is False
    assert "max_observations" in " ".join(blocked["errors"]).lower()

    # Age the pending row past the staleness horizon; the next generate
    # would expire it, so its slot must no longer block the submit.
    aged = utcnow() - timedelta(hours=PENDING_SUGGESTION_MAX_AGE_HOURS + 1)
    async with get_session() as session:
        await session.execute(
            update(SuggestionModel)
            .where(SuggestionModel.id == suggestion["id"])
            .values(created_at=aged)
        )

    accepted = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(free_floating),
        submitted_by=owner_id,
        atomic=True,
    )
    assert accepted["success"] is True
    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 1


@pytest.mark.asyncio
async def test_submit_protects_accepted_reservation() -> None:
    """ACCEPTED suggestions reserve budget just like PENDING ones.

    ``Suggestion.is_actionable`` treats both PENDING and ACCEPTED as live
    reservations: PENDING means "generated, awaiting acknowledgement"; ACCEPTED
    means "user approved, awaiting result". Both states must keep their slot
    so that an accepted-but-not-yet-submitted experiment cannot be evicted by
    free-floating manual submissions.
    """
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.operations.update_suggestion_status import (
        update_suggestion_status_operation,
    )
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Accepted Reservation Atomic",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 3,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 11,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, second = gen["suggestions"]

    # Submit one suggestion; keep ``second`` as ACCEPTED (user committed).
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )
    accept = await update_suggestion_status_operation(suggestion_id=second["id"], status="accepted")
    assert accept["success"] is True

    # Atomic flood attempt: existing=1, accepted_reserved=1, slack=1, so two
    # free-floating rows must be rejected as a batch.
    blocked = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {"parameter_values": {"x": 0.4}, "objective_values": {"y": 0.4}},
                {"parameter_values": {"x": 0.6}, "objective_values": {"y": 0.6}},
            ]
        ),
        submitted_by=owner_id,
        atomic=True,
    )
    assert blocked["success"] is False
    joined = " ".join(blocked["errors"]).lower()
    assert "max_observations" in joined
    assert "pending_reserved=1" in joined

    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 1

    # The ACCEPTED suggestion can still be submitted — its reservation is
    # consumed by this row instead of competing with free-floating slots.
    final = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": second["id"],
                    "parameter_values": second["parameter_values"],
                    "objective_values": {"y": float(second["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
        atomic=True,
    )
    assert final["success"] is True


@pytest.mark.asyncio
async def test_budget_stop_surfaces_pending_vs_accepted_breakdown() -> None:
    """Stop response splits actionable count into pending vs accepted.

    A client reading only ``n_pending`` could misread an ACCEPTED reservation
    as a stale PENDING suggestion. The ``actionable_breakdown`` field exposes
    the split so diagnostics can show which slots are user-committed.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.operations.update_suggestion_status import (
        update_suggestion_status_operation,
    )
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Pending vs Accepted Breakdown",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 17,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, second = gen["suggestions"]
    # Submit ``first`` (existing=1); mark ``second`` ACCEPTED (1 accepted).
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )
    await update_suggestion_status_operation(suggestion_id=second["id"], status="accepted")

    blocked = await generate_suggestions(campaign_id)
    details = blocked["error"]["details"]
    assert details["n_pending"] == 1  # legacy key -- actionable count
    breakdown = details["actionable_breakdown"]
    assert breakdown == {"pending": 0, "accepted": 1, "other": 0}


@pytest.mark.asyncio
async def test_generate_treats_accepted_suggestions_as_reservations() -> None:
    """Generation cannot evict ACCEPTED suggestions from the observation budget."""
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.operations.update_suggestion_status import (
        update_suggestion_status_operation,
    )
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Accepted Reservation Generate",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 11,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, second = gen["suggestions"]
    # Submit ``first`` so existing=1; mark ``second`` ACCEPTED so reserved=1.
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(
            [
                {
                    "suggestion_id": first["id"],
                    "parameter_values": first["parameter_values"],
                    "objective_values": {"y": float(first["parameter_values"]["x"])},
                }
            ]
        ),
        submitted_by=owner_id,
    )
    await update_suggestion_status_operation(suggestion_id=second["id"], status="accepted")

    blocked = await generate_suggestions(campaign_id)
    assert blocked["success"] is False
    assert blocked["error"]["code"] == "E012"  # BUDGET_EXCEEDED
    details = blocked["error"]["details"]
    assert details["stopping_reason"] == "budget_exceeded_observations"
    assert details["n_pending"] == 1  # ACCEPTED counted as actionable


@pytest.mark.asyncio
async def test_mixed_batch_protects_reservation_regardless_of_order_atomic() -> None:
    """A manual row ordered *before* an actionable row cannot steal its slot.

    Reproduces the reviewer's race: cap=2, two PENDING suggestions, no
    stored results. A mixed atomic batch ordered ``[manual, actionable]``
    used to keep the first ``cap - existing - unsubmitted_actionable``
    rows by input order, which let the manual row win and rejected the
    actionable submission. The partitioned budget guard must instead
    classify rows and reject the manual row regardless of position.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Mixed Order Atomic",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 13,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, _ = gen["suggestions"]

    rows = [
        # Manual row FIRST -- this is the order that exposed the bug.
        {"parameter_values": {"x": 0.42}, "objective_values": {"y": 0.42}},
        # Actionable row SECOND.
        {
            "suggestion_id": first["id"],
            "parameter_values": first["parameter_values"],
            "objective_values": {"y": float(first["parameter_values"]["x"])},
        },
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(rows),
        submitted_by=owner_id,
        atomic=True,
    )
    # Atomic mode: the manual row would exceed the cap (cap=2, existing=0,
    # actionable=2 → slack=0) so the whole batch is rejected. The
    # actionable row remains unsubmitted and its reservation intact.
    assert result["success"] is False
    joined = " ".join(result["errors"]).lower()
    assert "max_observations" in joined
    assert "result 0" in joined  # the manual row is index 0


@pytest.mark.asyncio
async def test_mixed_batch_protects_reservation_regardless_of_order_non_atomic() -> None:
    """Non-atomic: actionable row commits, manual row is rejected even if listed first."""
    from bo_mcp_server.operations.list_results import list_results_operation
    from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Mixed Order Non-Atomic",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 2,
        "initial_design_size": 2,
        "batch_size": 2,
        "random_seed": 13,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    gen = await generate_suggestions(campaign_id)
    first, _ = gen["suggestions"]

    rows = [
        {"parameter_values": {"x": 0.42}, "objective_values": {"y": 0.42}},
        {
            "suggestion_id": first["id"],
            "parameter_values": first["parameter_values"],
            "objective_values": {"y": float(first["parameter_values"]["x"])},
        },
    ]
    result = await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(rows),
        submitted_by=owner_id,
        atomic=False,
        continue_on_error=True,
        verbosity="detailed",
    )
    assert result["success"] is True
    # Manual row (index 0) is rejected; actionable row (index 1) commits.
    assert isinstance(result["partial_results"][0], dict)
    assert "max_observations" in result["partial_results"][0]["error"].lower()
    assert isinstance(result["partial_results"][1], str)  # result id

    stored = await list_results_operation(campaign_id=campaign_id)
    assert stored["total_count"] == 1

    # The submitted actionable suggestion is now COMPLETED; the other
    # remains PENDING (its reservation is still protected).
    listed_suggestions = await list_suggestions_operation(
        campaign_id=campaign_id, verbosity="minimal"
    )
    statuses = {s["suggestion_id"]: s["status"] for s in listed_suggestions["suggestions"]}
    assert statuses[first["id"]] == "completed"


@pytest.mark.asyncio
async def test_batch_size_clamps_to_remaining_observation_budget() -> None:
    """When ``remaining_budget < batch_size`` the batch is clamped, not skipped.

    Reproduces the friend's report: campaign with ``max_observations=3`` and
    ``batch_size=2`` previously generated 2 + 2 = 4 observations because the
    cap was only enforced *between* generations.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Clamp Batch",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "max_observations": 3,
        "initial_design_size": 2,
        "random_seed": 42,
        "batch_size": 2,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    # First batch fills two of the three observation slots.
    gen1 = await generate_suggestions(campaign_id)
    assert gen1["success"] is True
    assert len(gen1["suggestions"]) == 2
    results = [
        {
            "suggestion_id": s["id"],
            "parameter_values": s["parameter_values"],
            "objective_values": {"y": float(s["parameter_values"]["x"])},
        }
        for s in gen1["suggestions"]
    ]
    await submit_results_operation(
        campaign_id=campaign_id,
        results=_to_result_inputs(results),
        submitted_by=owner_id,
    )

    # Only one slot remains -- batch_size=2 must be clamped to 1.
    gen2 = await generate_suggestions(campaign_id)
    assert gen2["success"] is True
    assert len(gen2["suggestions"]) == 1, "Batch must be clamped to remaining budget"


@pytest.mark.asyncio
async def test_no_budget_runs_indefinitely() -> None:
    """Without budget fields the campaign behaves like the legacy path.

    Guards against a regression where the stopping check is wired with a
    truthy ``None`` and accidentally short-circuits campaigns that did not
    opt in to automatic stopping.
    """
    from bo_mcp_server.operations.submit_results import submit_results_operation
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "No Budget",
        "parameters": [
            {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
        ],
        "objectives": [{"name": "y", "direction": "minimize"}],
        "initial_design_size": 2,
        "random_seed": 42,
    }
    create_result = await create_campaign(intake, owner_id)
    campaign_id = create_result["campaign_id"]

    for _ in range(3):
        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        results = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"y": float(s["parameter_values"]["x"])},
            }
            for s in gen["suggestions"]
        ]
        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(results),
            submitted_by=owner_id,
        )
