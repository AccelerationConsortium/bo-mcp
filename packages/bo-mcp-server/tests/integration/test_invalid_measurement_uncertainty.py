"""Tests that bad ``measurement_uncertainty`` values are rejected, not warned.

Now that uncertainty flows into the bo-engine's ``train_yvar`` and routes
the GP onto a ``FixedNoiseGaussianLikelihood``, a negative stddev would be
silently squared into a positive variance and a NaN/Inf would propagate
into MLL. The submission layer must reject these inputs for declared
objectives instead of merely surfacing a warning.
"""

from __future__ import annotations

import math
from typing import Any
from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(rows: list[dict[str, Any]]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in rows]


async def _build_campaign(batch_size: int = 2) -> tuple[str, list[dict[str, Any]], str]:
    from bo_mcp_server.tools.create_campaign import create_campaign
    from bo_mcp_server.tools.generate_suggestions import generate_suggestions

    owner_id = str(uuid4())
    intake = {
        "name": "Uncertainty Validation",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }
    created = await create_campaign(intake, owner_id)
    campaign_id = created["campaign_id"]
    gen = await generate_suggestions(campaign_id, batch_size=batch_size)
    return campaign_id, gen["suggestions"], owner_id


class TestAtomicRejection:
    """Atomic mode must reject the entire batch on a bad uncertainty value."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_value",
        [-0.1, float("nan"), float("inf"), float("-inf")],
        ids=["negative", "nan", "inf", "-inf"],
    )
    async def test_atomic_rejects_bad_value_for_declared_objective(
        self, setup_database, bad_value: float
    ) -> None:
        """Negative/NaN/inf stddev on a declared objective fails the submission."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign()
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "measurement_uncertainty": {"y": 0.05},
            },
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "measurement_uncertainty": {"y": bad_value},
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
        assert "measurement_uncertainty" in joined
        # No rows leaked into storage.
        stored = await list_results_operation(campaign_id=campaign_id)
        assert stored["total_count"] == 0


class TestNonAtomicRejection:
    """Non-atomic mode drops the bad row but persists valid ones."""

    @pytest.mark.asyncio
    async def test_non_atomic_drops_invalid_row_only(self, setup_database) -> None:
        """A bad uncertainty value blocks only its row; valid rows commit."""
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign()
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                "measurement_uncertainty": {"y": 0.05},
            },
            {
                "suggestion_id": suggestions[1]["id"],
                "parameter_values": suggestions[1]["parameter_values"],
                "objective_values": {"y": 0.2},
                "measurement_uncertainty": {"y": -0.1},
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
        # The bad row left an error in partial_results, the valid row a real id.
        assert isinstance(result["partial_results"][1], dict)
        assert "measurement_uncertainty" in result["partial_results"][1]["error"].lower()
        listed = await list_results_operation(campaign_id=campaign_id)
        assert listed["total_count"] == 1


class TestUnknownObjectiveKeysStillWarn:
    """Unknown objective keys remain warnings (they are dropped before the engine)."""

    @pytest.mark.asyncio
    async def test_unknown_objective_keyed_uncertainty_is_warning_only(
        self, setup_database
    ) -> None:
        """An odd value attached to a non-declared key does not fail submission.

        These keys never reach ``train_yvar`` (the engine indexes by
        ``spec.objectives`` order), so they cannot corrupt the GP. The user
        is told about them via a warning but the row still commits.
        """
        from bo_mcp_server.operations.list_results import list_results_operation
        from bo_mcp_server.operations.submit_results import submit_results_operation

        campaign_id, suggestions, owner_id = await _build_campaign(batch_size=1)
        rows = [
            {
                "suggestion_id": suggestions[0]["id"],
                "parameter_values": suggestions[0]["parameter_values"],
                "objective_values": {"y": 0.1},
                # ``y`` is declared (good value); ``stray`` is unknown.
                "measurement_uncertainty": {"y": 0.05, "stray": math.nan},
            },
        ]
        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(rows),
            submitted_by=owner_id,
            atomic=True,
        )
        assert result["success"] is True
        assert any("unknown" in w.lower() for w in result.get("warnings", []))
        listed = await list_results_operation(campaign_id=campaign_id)
        assert listed["total_count"] == 1
