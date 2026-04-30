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
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


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

    owner_id = str(uuid4())
    intake = {
        "name": "Atomic Submit Regression",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    generated = await generate_suggestions(campaign_id, batch_size=batch_size)
    return campaign_id, generated["suggestions"], owner_id


class TestSubmitResultsAtomicPreValidation:
    """Atomic mode must pre-validate before any DB write leaks out."""

    @pytest.mark.asyncio
    async def test_atomic_failure_does_not_complete_earlier_suggestions(
        self, setup_database
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
    async def test_atomic_success_commits_everything(self, setup_database) -> None:
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
    async def test_non_atomic_continues_on_error(self, setup_database) -> None:
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
