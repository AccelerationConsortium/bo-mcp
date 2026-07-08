"""Integration tests for submit_results atomic-mode pre-validation.

When ``submit_results`` is called with ``atomic=True`` the operation advertises
all-or-nothing semantics: if any result in the batch fails validation, no
result row is persisted AND no suggestion-status update leaks out of the
transaction. The original implementation interleaved suggestion-status
updates with per-result validation inside a single session, so a failure at
result ``k`` still committed the ``COMPLETED`` transitions for results
``0..k-1`` when the function returned through the ``async with`` block
(the session commit ran on normal return).

These tests reproduce the leakage scenario and assert the post-fix
invariants.

References:
    - General database-transaction contract: all-or-nothing semantics require
      that every write participating in the logical unit-of-work either all
      commit or all roll back (C. J. Date, *An Introduction to Database
      Systems*, chapter on transaction management).
"""

from typing import Any

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from tests.factories import seed_owner


def _to_result_inputs(rows: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in rows]


async def _build_campaign_with_suggestions(
    batch_size: int = 3,
) -> tuple[str, list[dict[str, Any]], str]:
    """Create a single-objective campaign and generate ``batch_size`` suggestions.

    Returns (campaign_id, suggestion_payloads, owner_id) where each
    suggestion_payload carries both the suggestion id and the parameter
    values so callers can build matching ``ResultSubmissionInput`` rows.
    """
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = await seed_owner()
    intake = {
        "name": "Atomic Submit Regression",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    generated = await generate_suggestions(campaign_id, batch_size=batch_size)
    return campaign_id, generated["suggestions"], owner_id


@pytest.mark.usefixtures("setup_database")
class TestSubmitResultsAtomicPreValidation:
    """Atomic mode must pre-validate before any DB write leaks out."""

    @pytest.mark.asyncio
    async def test_atomic_failure_does_not_complete_earlier_suggestions(
        self,
    ) -> None:
        """A validation failure on a later result keeps earlier suggestions PENDING.

        In a batch of three results, the first two are valid and reference
        pending suggestions; the third references an unknown objective name.
        Before the fix, the session still committed the COMPLETED transitions
        for the first two suggestions because those writes were staged inside
        the per-result validation loop. After the fix, phase 1 runs
        read-only, so the failure on result 2 short-circuits before any
        suggestion is written.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=3)

        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "suggestion_id": suggestions[0]["id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[1]["id"],
            },
            {
                "parameter_values": suggestions[2]["parameter_values"],
                # Wrong objective name — passes Pydantic, fails operation-level
                # validation: spec requires "y".
                "objective_values": {"wrong_name": 0.3},
                "suggestion_id": suggestions[2]["id"],
            },
        ]

        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )

        # The operation must reject the entire batch.
        assert result["success"] is False
        assert result["result_ids"] == []
        assert any("missing objectives" in e.lower() for e in result["errors"])

        # No result row was persisted.
        listed_results = await list_results_operation(campaign_id=campaign_id)
        assert listed_results["total_count"] == 0

        # No suggestion transitioned to COMPLETED — the write path was aborted
        # before phase 2 ran, so the sibling suggestions that would have been
        # completed ahead of the failing index remain PENDING.
        listed_suggestions = await list_suggestions_operation(
            campaign_id=campaign_id, verbosity="minimal"
        )
        statuses = {s["suggestion_id"]: s["status"] for s in listed_suggestions["suggestions"]}
        for suggestion in suggestions:
            assert statuses[suggestion["id"]] == "pending", (
                "Atomic batch failure must not leak COMPLETED updates for "
                f"suggestion {suggestion['id']}"
            )

    @pytest.mark.asyncio
    async def test_atomic_success_commits_everything(self) -> None:
        """When atomic validation passes, all writes commit together.

        Counter-test to guard against over-rejection: a clean batch should
        persist all result rows and advance the referenced suggestions to
        COMPLETED in a single commit.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)

        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "suggestion_id": suggestions[0]["id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[1]["id"],
            },
        ]

        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )

        assert result["success"] is True
        assert len(result["result_ids"]) == 2

        listed_results = await list_results_operation(campaign_id=campaign_id)
        assert listed_results["total_count"] == 2

        listed_suggestions = await list_suggestions_operation(
            campaign_id=campaign_id, verbosity="minimal"
        )
        statuses = {s["suggestion_id"]: s["status"] for s in listed_suggestions["suggestions"]}
        for suggestion in suggestions:
            assert statuses[suggestion["id"]] == "completed"

    @pytest.mark.asyncio
    async def test_non_atomic_without_continue_on_error_does_not_persist_partial_data(
        self,
    ) -> None:
        """``atomic=False, continue_on_error=False`` must not commit partial rows.

        Reviewer's concern: this combination used to fall through to phase 2
        and persist the valid rows while returning ``success=False`` and an
        empty ``partial_results``. Callers were left with mutated state and
        no way to identify which rows landed. The operation now treats any
        validation error in this mode as all-or-nothing.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "suggestion_id": suggestions[0]["id"],
            },
            # Duplicate suggestion_id reference -- phase-1 validator records
            # a row error. With continue_on_error=False this must abort the
            # whole submission instead of silently keeping the first row.
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[0]["id"],
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=False,
        )
        assert result["success"] is False
        assert result["result_ids"] == []
        listed = await list_results_operation(campaign_id=campaign_id)
        assert listed["total_count"] == 0, (
            "Without continue_on_error the operation must roll back, not commit the surviving row"
        )

    @pytest.mark.asyncio
    async def test_atomic_exact_duplicate_returns_duplicate_envelope(self) -> None:
        """Atomic exact duplicates must return ``DUPLICATE_RESULT`` shape.

        Before the fix, ``_check_duplicates_for_result`` appended a generic
        "Result N is exact duplicate" string to ``tracking.errors`` *and*
        recorded the duplicate in ``tracking.duplicates_detected``. The
        generic-errors branch of ``_check_atomic_failures`` fired first and
        returned a plain ``success=False / errors=[...]`` envelope, so
        clients never saw the standardized ``ErrorCode.DUPLICATE_RESULT``
        response with its ``Use force=True`` recovery_action. The branch
        order was flipped so the duplicate-specific envelope wins.
        """
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)

        # First submission lands cleanly so there is an existing result the
        # duplicate detector can match against.
        seed_rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
        ]
        seed = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(seed_rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert seed["success"] is True

        # Second submission carries the same parameter values -> exact
        # duplicate. ``suggestion_id`` is the OTHER (still PENDING)
        # suggestion so the duplicate detector -- not the duplicate-id
        # validator -- is the rejection path.
        dup_rows = [
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.2},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(dup_rows),
            submitted_by=owner_id,
            atomic=True,
        )

        assert result["success"] is False
        # ``DUPLICATE_RESULT`` = E004 (see bo_mcp_server.errors.ErrorCode).
        assert result["error"]["code"] == "E004"
        assert "force=true" in result["error"]["recovery_action"].lower()
        assert result["error"]["details"]["duplicate_count"] == 1
        assert result["duplicates_detected"], "duplicates_detected must be populated"

    @pytest.mark.asyncio
    async def test_non_atomic_continue_on_error_drops_exact_duplicate(self) -> None:
        """Non-atomic + continue_on_error: duplicate row dropped, others persisted.

        Before the fix, ``_check_duplicates_for_result`` only produced a row
        error in atomic mode. That meant ``atomic=False`` silently accepted
        exact-duplicate parameter values despite the warning saying ``Use
        force=True to override``. Now the duplicate is a hard row error in
        every mode; ``continue_on_error=True`` keeps surviving rows.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=3)

        # Seed the campaign with one accepted result.
        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["id"],
                        "parameter_values": suggestions[0]["parameter_values"],
                        "objective_values": {"y": 0.1},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        # Submit two more rows: one fresh, one exact duplicate of the seed.
        rows = [
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
            },
            {
                "suggestion_id": suggestions[2]["id"],
                "parameter_values": suggestions[0]["parameter_values"],  # duplicate
                "objective_values": {"y": 0.3},
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
        # The duplicate row (index 1) must surface as an error; the fresh
        # row (index 0) must commit.
        assert isinstance(result["partial_results"][1], dict)
        assert "exact duplicate" in result["partial_results"][1]["error"].lower()
        # Total stored = seed (1) + fresh (1). Duplicate did NOT land.
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_non_atomic_no_continue_on_error_rejects_exact_duplicate(
        self,
    ) -> None:
        """Non-atomic + continue_on_error=False: exact duplicate aborts batch.

        The non-atomic-no-continue path is all-or-nothing for any
        validation error. An exact duplicate is now a hard error in every
        mode, so the surviving non-duplicate row must also roll back.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=3)

        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["id"],
                        "parameter_values": suggestions[0]["parameter_values"],
                        "objective_values": {"y": 0.1},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        rows = [
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
            },
            {
                "suggestion_id": suggestions[2]["id"],
                "parameter_values": suggestions[0]["parameter_values"],  # duplicate
                "objective_values": {"y": 0.3},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=False,
        )
        assert result["success"] is False
        # Only the seed row should remain. Neither the fresh row nor the
        # duplicate row landed.
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 1

    @pytest.mark.asyncio
    async def test_non_atomic_force_true_persists_exact_duplicate(self) -> None:
        """``force=True`` is the documented override for exact duplicates.

        The recovery_action surfaced on ``ErrorCode.DUPLICATE_RESULT`` tells
        callers to retry with ``force=True``. That contract must hold across
        modes: with ``force=True`` the duplicate row commits and the
        suggestion transitions to COMPLETED.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)

        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["id"],
                        "parameter_values": suggestions[0]["parameter_values"],
                        "objective_values": {"y": 0.1},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        forced = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[1]["id"],
                        "parameter_values": suggestions[0]["parameter_values"],
                        "objective_values": {"y": 0.4},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=True,
            force=True,
        )
        assert forced["success"] is True
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_atomic_rejects_in_batch_exact_duplicate(self) -> None:
        """Two same-batch rows with identical ``parameter_values`` are rejected.

        The previous duplicate-detection baseline was a snapshot of stored
        results captured before phase 1 ran, so rows could shadow each
        other inside a single submission. The baseline now grows with each
        accepted row, so the second row is flagged as a duplicate of the
        first even though no stored result matches it yet.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            # Same parameter values as row 0 but a different (also valid)
            # suggestion_id. Without the in-batch baseline, both would
            # land in storage.
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.2},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert result["success"] is False
        # Atomic exact-duplicate envelope (E004) wins over the generic
        # validation-errors branch.
        assert result["error"]["code"] == "E004"
        # The duplicate-record must point at the earlier batch row, not a
        # stored result.
        in_flight = [
            d
            for d in result["duplicates_detected"]
            if d.get("duplicate_source") == "in_flight_batch"
        ]
        assert in_flight, "duplicate must be tagged with duplicate_source=in_flight_batch"
        assert in_flight[0]["result_index"] == 1
        assert in_flight[0]["duplicate_of_index"] == 0
        # Nothing landed.
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 0

    @pytest.mark.asyncio
    async def test_non_atomic_continue_drops_in_batch_exact_duplicate(self) -> None:
        """Non-atomic + continue: the in-batch duplicate is dropped, first row commits."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.2},
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
        assert isinstance(result["partial_results"][1], dict)
        assert "exact duplicate" in result["partial_results"][1]["error"].lower()
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 1

    @pytest.mark.asyncio
    async def test_force_true_persists_in_batch_exact_duplicate(self) -> None:
        """``force=True`` overrides in-batch duplicate detection too."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.2},
            },
        ]
        forced = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
            force=True,
        )
        assert forced["success"] is True
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_stale_first_row_does_not_shadow_later_valid_row(self) -> None:
        """A stale-id row must not pollute the duplicate baseline.

        Reviewer's scenario: row 0 has duplicate parameter values *and* a
        stale (REJECTED) suggestion_id; row 1 has the same parameter
        values but a valid (PENDING) suggestion_id. Before the fix, the
        per-row loop added row 0's params to ``batch_so_far_params``
        before the suggestion-reference check fired, so row 1 was
        rejected as a duplicate of row 0 even though row 0 itself was
        going to be dropped for being stale. The reordered phase 1 (shape
        → suggestion-ref → duplicate) drops row 0 first, so its params
        never enter the baseline.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.operations.update_suggestion_status import (
            update_suggestion_status_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        # Mark the first suggestion REJECTED so row 0 will fail the
        # suggestion-reference check.
        await update_suggestion_status_operation(
            suggestion_id=suggestions[0]["id"], status="rejected"
        )

        rows = [
            {
                "suggestion_id": suggestions[0]["id"],  # stale (REJECTED)
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            # Same parameter values as row 0 but a valid PENDING id.
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.2},
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
        # Row 0 fails the suggestion-reference check.
        assert isinstance(result["partial_results"][0], dict)
        assert "not actionable" in result["partial_results"][0]["error"].lower()
        # Row 1 must commit -- the stale row 0 did not shadow it.
        assert isinstance(result["partial_results"][1], str)
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 1

    @pytest.mark.asyncio
    async def test_budget_dropped_row_does_not_shadow_actionable_duplicate(
        self,
    ) -> None:
        """A manual row dropped by the budget guard must not shadow an actionable duplicate.

        Reviewer's scenario: ``max_observations=2`` with 1 stored result
        and 1 actionable suggestion (S1) leaves zero free-floating slack.
        Submit ``[manual row with params P, actionable row (sid=S1, params P)]``
        in non-atomic + continue_on_error mode. The manual row is going
        to be dropped by the budget guard anyway, so it must not enter
        the in-batch duplicate baseline and shadow the actionable row.
        The actionable row should commit; the manual row should fail
        with the budget error.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        # Build a campaign with max_observations=2, batch_size=2 so we can
        # generate two pending suggestions and seed one stored result.
        owner_id = await seed_owner()
        intake = {
            "name": "Budget Drop No Shadow",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "max_observations": 2,
            "initial_design_size": 2,
            "batch_size": 2,
            "random_seed": 21,
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        seed, target = gen["suggestions"]

        # Seed the first result through phase 2 cleanly. After this:
        # stored = 1, actionable = 1 (``target`` still PENDING),
        # slack = 2 - 1 - 1 = 0 free-floating budget.
        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": seed["id"],
                        "parameter_values": seed["parameter_values"],
                        "objective_values": {"y": float(seed["parameter_values"]["x"])},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        rows = [
            # Free-floating row that shares ``target``'s parameter values.
            # The budget guard will drop it (slack=0).
            {
                "parameter_values": target["parameter_values"],
                "objective_values": {"y": 0.99},
            },
            # Actionable row with the same params; consumes ``target``'s
            # reservation.
            {
                "suggestion_id": target["id"],
                "parameter_values": target["parameter_values"],
                "objective_values": {"y": float(target["parameter_values"]["x"])},
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
        # Row 0 fails as a duplicate of the (reservation-prioritized) row 1.
        # The filter processes reservation-consuming rows first so committing
        # the reserved row transitions its suggestion to COMPLETED rather
        # than orphaning it -- the free-floating row 0 then collides with
        # the now-accepted reserved baseline and is rejected. (It would
        # have failed the budget check too; the duplicate error is the
        # more specific signal and wins.)
        assert isinstance(result["partial_results"][0], dict)
        assert "exact duplicate" in result["partial_results"][0]["error"].lower()
        # Row 1 commits -- the dropped free-floating row did not shadow it.
        assert isinstance(result["partial_results"][1], str)
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_duplicate_row_does_not_consume_budget_slack(self) -> None:
        """Duplicate rows do not reduce the slack available to a later unique row.

        Reviewer's scenario: ``max_observations=2``, no reservations, batch
        ``[A, duplicate-of-A, B]``. Previously the budget guard ran first
        and kept rows 0+1 (consuming both slack slots), then the duplicate
        filter dropped row 1 -- leaving the campaign at 1 stored when it
        could have been 2. With the merged filter, the duplicate is
        rejected before it consumes slack, so row 2 (``B``) fits.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = await seed_owner()
        intake = {
            "name": "Dup Does Not Eat Budget",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "max_observations": 2,
        }
        created = await create_campaign(intake, owner_id)
        campaign_id = created["campaign_id"]

        rows = [
            {"parameter_values": {"x": 0.10}, "objective_values": {"y": 0.1}},
            # Same params as row 0 -> in-batch duplicate; must be rejected
            # *without* consuming the slack slot that row 2 needs.
            {"parameter_values": {"x": 0.10}, "objective_values": {"y": 0.2}},
            {"parameter_values": {"x": 0.85}, "objective_values": {"y": 0.3}},
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
        # Row 0: committed (slack consumed by a unique row).
        assert isinstance(result["partial_results"][0], str)
        # Row 1: duplicate-rejected without consuming slack.
        assert isinstance(result["partial_results"][1], dict)
        assert "exact duplicate" in result["partial_results"][1]["error"].lower()
        # Row 2: committed -- there was still budget for it.
        assert isinstance(result["partial_results"][2], str)
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_reservation_wins_duplicate_tie_with_slack(self) -> None:
        """A reserved row beats an earlier free-floating duplicate, even when slack exists.

        Slack=1 means a free-floating row would otherwise be admitted, so
        the question of "which row wins on identical params" is forced
        rather than masked by the budget guard. The filter processes
        reservation-consuming rows first so committing them transitions
        the suggestion to COMPLETED, rather than orphaning the
        reservation in PENDING/ACCEPTED.

        Batch: ``[manual P, reserved(sid=actionable) P]`` with cap=2,
        zero existing, one actionable suggestion (slack=1 for free-
        floating). The reserved row must commit and the manual row must
        be rejected as a duplicate of the reservation.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = await seed_owner()
        intake = {
            "name": "Reservation Wins Duplicate Tie",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "max_observations": 2,
            "initial_design_size": 1,
            "batch_size": 1,
        }
        created = await create_campaign(intake, owner_id)
        campaign_id = created["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        target = gen["suggestions"][0]
        # State now: 0 stored, 1 actionable (``target``), slack = 2 - 0 - 1 = 1.
        # A free-floating row would otherwise fit, so the duplicate tie
        # forces a winner-takes-all decision.

        rows = [
            {
                "parameter_values": target["parameter_values"],
                "objective_values": {"y": 0.42},
            },
            {
                "suggestion_id": target["id"],
                "parameter_values": target["parameter_values"],
                "objective_values": {"y": float(target["parameter_values"]["x"])},
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
        # Row 0 (manual) is the duplicate-loser.
        assert isinstance(result["partial_results"][0], dict)
        assert "exact duplicate" in result["partial_results"][0]["error"].lower()
        # Row 1 (reserved) committed.
        assert isinstance(result["partial_results"][1], str)

        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 1

        # The reservation completed cleanly -- target -> COMPLETED.
        listed = await list_suggestions_operation(campaign_id=campaign_id, verbosity="minimal")
        statuses = {s["suggestion_id"]: s["status"] for s in listed["suggestions"]}
        assert statuses[target["id"]] == "completed"

    @pytest.mark.asyncio
    async def test_non_atomic_continues_on_error(self) -> None:
        """Non-atomic mode persists valid rows and skips invalid ones.

        Guards the fix against the opposite regression: we must preserve the
        partial-success behavior that ``atomic=False`` + ``continue_on_error``
        has always offered. Only the invalid row is dropped; the valid ones
        commit and their suggestions advance to COMPLETED.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=3)

        rows = [
            {
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "suggestion_id": suggestions[0]["id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                # Invalid objective key — only this row must be skipped.
                "objective_values": {"wrong_name": 0.2},
                "suggestion_id": suggestions[1]["id"],
            },
            {
                "parameter_values": suggestions[2]["parameter_values"],
                "objective_values": {"y": 0.3},
                "suggestion_id": suggestions[2]["id"],
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
        assert len(result["result_ids"]) == 2
        assert isinstance(result["partial_results"][1], dict)
        assert "error" in result["partial_results"][1]

        listed_results = await list_results_operation(campaign_id=campaign_id)
        assert listed_results["total_count"] == 2

        listed_suggestions = await list_suggestions_operation(
            campaign_id=campaign_id, verbosity="minimal"
        )
        statuses = {s["suggestion_id"]: s["status"] for s in listed_suggestions["suggestions"]}
        assert statuses[suggestions[0]["id"]] == "completed"
        assert statuses[suggestions[2]["id"]] == "completed"
        # The skipped row's suggestion must remain PENDING because its result
        # was rejected.
        assert statuses[suggestions[1]["id"]] == "pending"
