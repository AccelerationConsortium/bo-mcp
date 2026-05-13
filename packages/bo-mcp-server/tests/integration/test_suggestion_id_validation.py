"""Tests that ``suggestion_id`` references are validated before phase 2.

Submitting against a stale (REJECTED / EXPIRED / COMPLETED) suggestion
used to silently re-complete it, and two rows with the same
``suggestion_id`` could both transition the suggestion to ``COMPLETED``
and attach two ``Result`` rows to a single suggestion. The phase-1
validator (:func:`_validate_suggestion_references`) and the partial
``ix_results_suggestion_id_unique`` index (DB-level safety net) close
both gaps.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(rows: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in rows]


async def _build_campaign_with_suggestions(
    batch_size: int = 2,
) -> tuple[str, list[dict[str, Any]], str]:
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Suggestion-id Validation",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    gen = await generate_suggestions(campaign_id, batch_size=batch_size)
    return campaign_id, gen["suggestions"], owner_id


class TestDuplicateSuggestionIdWithinBatch:
    """Two rows referencing the same actionable suggestion are rejected."""

    @pytest.mark.asyncio
    async def test_atomic_rejects_duplicate_suggestion_id(self, setup_database) -> None:
        """Atomic mode aborts the entire batch on a duplicate reference."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions()
        # Distinct parameter values on each row so the in-batch parameter
        # duplicate detector does not preempt the suggestion-id validator.
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": {"x": 0.97},
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
        joined = " ".join(result["errors"]).lower()
        assert "duplicate suggestion_id" in joined
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 0

        # The suggestion stays PENDING (no row leaked through to phase 2).
        listed = await list_suggestions_operation(campaign_id=campaign_id, verbosity="minimal")
        statuses = {s["suggestion_id"]: s["status"] for s in listed["suggestions"]}
        assert statuses[suggestions[0]["id"]] == "pending"

    @pytest.mark.asyncio
    async def test_non_atomic_keeps_first_drops_duplicate(self, setup_database) -> None:
        """Non-atomic mode keeps the first reference, rejects subsequent ones."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions()
        # Distinct parameter values to isolate the suggestion-id validator.
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": {"x": 0.97},
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
        assert "duplicate suggestion_id" in result["partial_results"][1]["error"].lower()
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 1


class TestNonActionableRepeatedReference:
    """Repeated warning-only ``suggestion_id`` values are not duplicate-id errors.

    The duplicate-id check only protects *actionable* suggestions from
    being claimed twice. Missing UUIDs, invalid formats, and IDs from a
    different campaign are surfaced as warnings only — two rows sharing
    such a value must both fall through as free-floating and be
    subject to the normal parameter-duplicate / budget rules.
    """

    @pytest.mark.asyncio
    async def test_two_rows_with_unknown_uuid_commit_as_free_floating(self, setup_database) -> None:
        """Two rows sharing the same not-found ``suggestion_id`` both persist."""
        from uuid import uuid4 as _uuid4

        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, _suggestions, owner_id = await _build_campaign_with_suggestions()
        stray_id = str(_uuid4())  # not in DB
        rows = [
            {
                "suggestion_id": stray_id,
                "parameter_values": {"x": 0.31},
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": stray_id,
                "parameter_values": {"x": 0.72},
                "objective_values": {"y": 0.2},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert result["success"] is True
        joined = " ".join(result.get("warnings", [])).lower()
        assert "not found" in joined
        # The duplicate-id check must not fire.
        assert not any("duplicate suggestion_id" in e.lower() for e in result.get("errors", []))
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 2

    @pytest.mark.asyncio
    async def test_two_rows_with_invalid_uuid_commit_as_free_floating(self, setup_database) -> None:
        """Invalid-format ``suggestion_id``s are warning-only and may repeat."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, _suggestions, owner_id = await _build_campaign_with_suggestions()
        rows = [
            {
                "suggestion_id": "not-a-uuid",
                "parameter_values": {"x": 0.31},
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": "not-a-uuid",
                "parameter_values": {"x": 0.72},
                "objective_values": {"y": 0.2},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert result["success"] is True
        joined = " ".join(result.get("warnings", [])).lower()
        assert "invalid suggestion_id" in joined
        assert not any("duplicate suggestion_id" in e.lower() for e in result.get("errors", []))
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 2

    @pytest.mark.asyncio
    async def test_two_rows_with_foreign_campaign_id_commit_as_free_floating(
        self, setup_database
    ) -> None:
        """``suggestion_id``s from a different campaign are warning-only and may repeat."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        # Build campaign A and grab its suggestion id.
        _, suggestions_a, _ = await _build_campaign_with_suggestions()
        foreign_id = suggestions_a[0]["id"]

        # Build campaign B; submit two rows referencing campaign A's id.
        campaign_b, _suggestions_b, owner_id_b = await _build_campaign_with_suggestions()
        rows = [
            {
                "suggestion_id": foreign_id,
                "parameter_values": {"x": 0.31},
                "objective_values": {"y": 0.1},
            },
            {
                "suggestion_id": foreign_id,
                "parameter_values": {"x": 0.72},
                "objective_values": {"y": 0.2},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_b,
            results=_to_result_inputs(rows),
            submitted_by=owner_id_b,
            atomic=True,
        )
        assert result["success"] is True
        joined = " ".join(result.get("warnings", [])).lower()
        assert "different campaign" in joined
        assert not any("duplicate suggestion_id" in e.lower() for e in result.get("errors", []))
        # Only campaign B should have the new rows.
        assert (await list_results_operation(campaign_id=campaign_b))["total_count"] == 2


class TestStaleSuggestionReference:
    """Submitting against a non-actionable suggestion is rejected."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stale_status", ["rejected", "expired"])
    async def test_atomic_rejects_stale_status(self, setup_database, stale_status: str) -> None:
        """REJECTED / EXPIRED suggestions cannot be re-completed by a submission."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.list_suggestions import list_suggestions_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.operations.update_suggestion_status import (
            update_suggestion_status_operation,
        )

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions()
        await update_suggestion_status_operation(
            suggestion_id=suggestions[0]["id"], status=stale_status
        )

        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert result["success"] is False
        joined = " ".join(result["errors"]).lower()
        assert "not actionable" in joined
        assert stale_status in joined

        # Suggestion status is preserved.
        listed = await list_suggestions_operation(campaign_id=campaign_id, verbosity="minimal")
        statuses = {s["suggestion_id"]: s["status"] for s in listed["suggestions"]}
        assert statuses[suggestions[0]["id"]] == stale_status
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 0

    @pytest.mark.asyncio
    async def test_atomic_rejects_already_completed(self, setup_database) -> None:
        """A COMPLETED suggestion cannot be re-completed by a second submission."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign_with_suggestions()

        # First submission marks suggestion as COMPLETED.
        first = await submit_results_operation(
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
        assert first["success"] is True

        # A second submission against the same id must be rejected even when
        # parameters differ from the first row (so duplicate-parameter
        # detection cannot mask the stale-id check).
        second = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "suggestion_id": suggestions[0]["id"],
                        "parameter_values": {"x": 0.93},
                        "objective_values": {"y": 0.2},
                    }
                ]
            ),
            submitted_by=owner_id,
            atomic=True,
        )
        assert second["success"] is False
        joined = " ".join(second["errors"]).lower()
        assert "not actionable" in joined
        assert (await list_results_operation(campaign_id=campaign_id))["total_count"] == 1
