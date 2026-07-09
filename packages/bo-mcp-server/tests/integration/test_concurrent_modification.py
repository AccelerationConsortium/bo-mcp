"""Integration tests for optimistic-locking error surfacing.

When two state-mutating operations race on the same campaign, the one that
loses the ``UPDATE ... WHERE version = expected_version`` race must see a
structured ``CONCURRENT_MODIFICATION`` envelope with retry guidance, not an
opaque 500 from a raw exception.

Reference:
    - MCP structured-error guidance: each tool error must carry a recovery
      action so agents can retry safely
      (https://modelcontextprotocol.io/specification).

Testing strategy: the race itself is deterministically simulated by patching
``CampaignRepository.save`` to raise ``ConcurrentModificationError`` on the
first update (i.e. the first save with a non-``None`` ``expected_version``).
This mirrors the real lost-race condition — the repository's ``UPDATE ...
WHERE version = expected`` matches zero rows — without depending on the
in-memory SQLite engine's concurrency primitives (which run through a single
connection and therefore cannot race at the SQL layer).
"""

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.storage import repositories as repo_mod
from bo_mcp_server.storage.base import ConcurrentModificationError
from tests.factories import seed_owner


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


def _inject_lost_race(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Patch ``CampaignRepository.save`` so the first update raises.

    Returns a mutable call counter so tests can assert the patch was
    actually exercised (guards against tests that pass for the wrong
    reason — e.g. when ``save`` was never reached).
    """
    original_save = repo_mod.CampaignRepository.save
    state = {"updates_triggered": 0}

    async def lost_race_save(self, campaign, expected_version=None):  # type: ignore[no-untyped-def]
        if expected_version is not None:
            state["updates_triggered"] += 1
            if state["updates_triggered"] == 1:
                msg = "Campaign"
                raise ConcurrentModificationError(msg, campaign.id, expected_version)
        return await original_save(self, campaign, expected_version=expected_version)

    monkeypatch.setattr(repo_mod.CampaignRepository, "save", lost_race_save)
    return state


@pytest.mark.usefixtures("setup_database")
class TestConcurrentModificationEnvelope:
    """Structured CONCURRENT_MODIFICATION envelope on each racing operation."""

    @pytest.mark.asyncio
    async def test_submit_results_lost_race_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lost submit race surfaces CONCURRENT_MODIFICATION with retry hint.

        Two concurrent submits race; one wins the version update, the other
        discovers ``UPDATE ... WHERE version = expected`` matched zero rows.
        The losing caller must see a structured envelope with the code,
        a recovery action, and a ``retry_after_seconds`` hint so it can back
        off and retry against the new version.
        """
        from bo_mcp_server.operations.submit_results import submit_results_operation
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        # Multi-objective so submit_results computes hypervolume and triggers
        # the campaign save in _update_campaign_state — the site that can lose
        # the race in production.
        owner_id = await seed_owner()
        intake = {
            "name": "Concurrent Submit Race",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "y1", "direction": "minimize"},
                {"name": "y2", "direction": "minimize"},
            ],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        race_state = _inject_lost_race(monkeypatch)

        result = await submit_results_operation(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": 0.3},
                        "objective_values": {"y1": 1.0, "y2": 0.5},
                    },
                    {
                        "parameter_values": {"x": 0.7},
                        "objective_values": {"y1": 0.8, "y2": 0.9},
                    },
                ]
            ),
            submitted_by=owner_id,
        )

        assert race_state["updates_triggered"] >= 1, (
            "Test did not exercise the patched save path — the lost race was never simulated."
        )
        assert result["success"] is False
        assert result["error"]["code"] == "E010"
        details = result["error"]["details"]
        assert details["entity_type"] == "Campaign"
        assert details["retry_after_seconds"] > 0
        assert details["campaign_id"] == campaign_id
        recovery = result["error"]["recovery_action"].lower()
        assert "retry" in recovery
        assert "version" in recovery
        # Submit-specific response shape is preserved even on the error path.
        assert result["result_ids"] == []
        assert "duplicates_detected" in result

    @pytest.mark.asyncio
    async def test_campaign_lifecycle_lost_race_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pause/resume/terminate losing the version race surfaces the envelope.

        ``manage_campaign_lifecycle`` saves a status transition under optimistic
        locking; if another writer updates the campaign first, the lifecycle
        call must return a structured error with the transport-specific
        ``campaign_id``/``status``/``previous_status`` fields preserved.
        """
        from bo_mcp_server.operations.campaign_lifecycle import (
            manage_campaign_lifecycle_operation,
        )
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = await seed_owner()
        intake = {
            "name": "Lifecycle Race",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]
        # Move to RUNNING so pause is a valid transition.
        await generate_suggestions(campaign_id)

        race_state = _inject_lost_race(monkeypatch)

        result = await manage_campaign_lifecycle_operation(campaign_id=campaign_id, action="pause")

        assert race_state["updates_triggered"] >= 1
        assert result["success"] is False
        assert result["error"]["code"] == "E010"
        assert result["error"]["details"]["action"] == "pause"
        # Lifecycle response shape is preserved on the error path so that
        # callers can report the previous status and retry against it.
        assert result["campaign_id"] == campaign_id
        assert result["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_generate_suggestions_lost_race_returns_structured_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Suggestion generation losing the version race surfaces the envelope.

        ``generate_suggestions`` advances the campaign iteration and saves
        under optimistic locking at the end of the operation; a lost race
        must convert to the structured error, not an opaque 500, so agents
        can retry without duplicating the suggestion batch they already
        persisted (rollback is handled by the surrounding DB transaction).
        """
        from bo_mcp_server.operations.generate_suggestions import (
            generate_suggestions_operation,
        )
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = await seed_owner()
        intake = {
            "name": "Generate Race",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        race_state = _inject_lost_race(monkeypatch)

        result = await generate_suggestions_operation(campaign_id=campaign_id, batch_size=1)

        assert race_state["updates_triggered"] >= 1
        assert result["success"] is False
        assert result["error"]["code"] == "E010"
        assert result["error"]["details"]["campaign_id"] == campaign_id
        # Generate-suggestions-specific fields are preserved.
        assert result["suggestions"] == []
        assert result["iteration"] is None
