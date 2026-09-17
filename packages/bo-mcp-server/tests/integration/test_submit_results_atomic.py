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
from uuid import UUID

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
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[1]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[2]["parameter_values"],
                # Wrong objective name — passes Pydantic, fails operation-level
                # validation: spec requires "y".
                "objective_values": {"wrong_name": 0.3},
                "suggestion_id": suggestions[2]["suggestion_id"],
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
            assert statuses[suggestion["suggestion_id"]] == "pending", (
                "Atomic batch failure must not leak COMPLETED updates for "
                f"suggestion {suggestion['suggestion_id']}"
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
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[1]["suggestion_id"],
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
            assert statuses[suggestion["suggestion_id"]] == "completed"

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
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            # Duplicate suggestion_id reference -- phase-1 validator records
            # a row error. With continue_on_error=False this must abort the
            # whole submission instead of silently keeping the first row.
            {
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "suggestion_id": suggestions[0]["suggestion_id"],
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
            suggestion_id=suggestions[0]["suggestion_id"], status="rejected"
        )

        rows = [
            {
                "suggestion_id": suggestions[0]["suggestion_id"],  # stale (REJECTED)
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            # Same parameter values as row 0 but a valid PENDING id.
            {
                "suggestion_id": suggestions[1]["suggestion_id"],
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
                "suggestion_id": suggestions[0]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[1]["parameter_values"],
                # Invalid objective key — only this row must be skipped.
                "objective_values": {"wrong_name": 0.2},
                "suggestion_id": suggestions[1]["suggestion_id"],
            },
            {
                "parameter_values": suggestions[2]["parameter_values"],
                "objective_values": {"y": 0.3},
                "suggestion_id": suggestions[2]["suggestion_id"],
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
        assert statuses[suggestions[0]["suggestion_id"]] == "completed"
        assert statuses[suggestions[2]["suggestion_id"]] == "completed"
        # The skipped row's suggestion must remain PENDING because its result
        # was rejected.
        assert statuses[suggestions[1]["suggestion_id"]] == "pending"


@pytest.mark.usefixtures("setup_database")
class TestSubmitResultsReplicates:
    """Distinct experiments may share parameter settings.

    Running the same conditions twice is a replicate — the standard way to
    estimate measurement noise — not a submission error. Both measurements
    must be stored separately and both must reach the optimizer, because a
    surrogate fitted on one of two disagreeing observations is fitted on a
    fiction. Repeated ingestion is prevented by experiment identity instead:
    a suggestion can be answered once, enforced by the phase-1 checks and
    the ``ix_results_suggestion_id_unique`` partial index.
    """

    @pytest.mark.asyncio
    async def test_replicate_across_requests_is_accepted_without_force(self) -> None:
        """Same parameters, different suggestion, second request: both stored."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        shared_params = suggestions[0]["parameter_values"]

        first = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.1},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )
        assert first["success"] is True

        # Same settings, a different experiment: the replicate. No force.
        second = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[1]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.2},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        assert second["success"] is True
        assert not any("duplicate" in w.lower() for w in second.get("warnings", []))

        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2
        # The two measurements stay distinct: this is the whole point.
        objective_values = sorted(r["objective_values"]["y"] for r in stored["results"])
        assert objective_values == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_replicate_within_one_request_is_accepted(self) -> None:
        """Two rows in one batch may share parameter values."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        shared_params = suggestions[0]["parameter_values"]

        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.1},
                    },
                    {
                        "suggestion_id": suggestions[1]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.3},
                    },
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        assert result["success"] is True
        assert len(result["result_ids"]) == 2
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 2

    @pytest.mark.asyncio
    async def test_both_replicate_measurements_reach_the_backend(self) -> None:
        """Both rows survive into the optimizer's observation set.

        Storing two rows is not enough — a parameter-keyed deduplication
        anywhere between the database and the backend would silently drop
        one, and the surrogate would then be fitted on a single arbitrary
        measurement of a setting that produced two different answers.
        """
        from bo_mcp_server.idempotency import session_scope
        from bo_mcp_server.operations.helpers import results_to_observations
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.storage import ResultRepository
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        shared_params = suggestions[0]["parameter_values"]

        await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.1},
                    },
                    {
                        "suggestion_id": suggestions[1]["suggestion_id"],
                        "parameter_values": shared_params,
                        "objective_values": {"y": 0.9},
                    },
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        # The conversion the optimizer actually consumes must carry both
        # rows, with their differing measurements intact.
        async with session_scope(None) as db:
            stored = await ResultRepository(db).list_by_campaign(UUID(campaign_id))
        observations = results_to_observations(stored)
        assert len(observations) == 2
        assert sorted(o.objective_values["y"] for o in observations) == [0.1, 0.9]

        # Re-deriving campaign state from storage (the reload path) must
        # still ingest both, not collapse them on the way in.
        regenerated = await generate_suggestions(campaign_id, batch_size=1)
        assert regenerated["success"] is True

    @pytest.mark.asyncio
    async def test_each_replicate_consumes_observation_budget(self) -> None:
        """A replicate is a real experiment, so it spends a real budget slot."""
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = await seed_owner()
        intake = {
            "name": "Replicate Budget",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "max_observations": 2,
        }
        created = await create_campaign(intake, owner_id)
        campaign_id = created["campaign_id"]

        rows = [
            {"parameter_values": {"x": 0.5}, "objective_values": {"y": 0.1}},
            {"parameter_values": {"x": 0.5}, "objective_values": {"y": 0.2}},
            {"parameter_values": {"x": 0.5}, "objective_values": {"y": 0.3}},
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=True,
            verbosity="detailed",
        )

        # Two replicates fit the budget; the third overflows it. If
        # replicates were collapsed by parameter equality, rows 1 and 2
        # would have been dropped as duplicates instead.
        assert len(result["result_ids"]) == 2
        assert isinstance(result["partial_results"][2], dict)
        assert "max_observations" in result["partial_results"][2]["error"]

    @pytest.mark.asyncio
    async def test_second_result_for_one_suggestion_is_rejected(self) -> None:
        """One result per suggestion: identity, not parameter values.

        This is what actually prevents repeated ingestion now that
        parameter equality no longer rejects anything.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        target = suggestions[0]

        first = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": target["suggestion_id"],
                        "parameter_values": target["parameter_values"],
                        "objective_values": {"y": 0.1},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )
        assert first["success"] is True

        # Answering the same suggestion again is an identity conflict, not
        # a replicate: the experiment it commissioned already has a result.
        repeat = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": target["suggestion_id"],
                        "parameter_values": target["parameter_values"],
                        "objective_values": {"y": 0.7},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        assert repeat["success"] is False
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 1

    @pytest.mark.asyncio
    async def test_duplicate_suggestion_id_within_one_request_is_still_rejected(
        self,
    ) -> None:
        """Two rows claiming one suggestion stay an error, replicates or not."""
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions(batch_size=2)
        target = suggestions[0]

        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": target["suggestion_id"],
                        "parameter_values": target["parameter_values"],
                        "objective_values": {"y": 0.1},
                    },
                    {
                        "suggestion_id": target["suggestion_id"],
                        "parameter_values": target["parameter_values"],
                        "objective_values": {"y": 0.2},
                    },
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )

        assert result["success"] is False
