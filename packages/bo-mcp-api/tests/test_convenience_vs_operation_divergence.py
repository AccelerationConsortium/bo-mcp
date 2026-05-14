"""Pin the shape divergence between legacy REST GETs and MCP operations.

Two views of the same data are intentional and first-class:

* MCP / ``POST /query`` callers consume the paginated envelope produced
  by ``list_*_operation`` (summary dicts under verbosity-keyed shapes).
* Legacy REST ``GET /api/{collection}[/{id}]`` callers consume a bare
  array (or single object) of full domain entities.

The convenience facade helpers in :mod:`bo_mcp_server.client` are
documented as serving the second view. These tests answer the same
business question through both transports and assert that each shape
remains stable — so any future drift surfaces immediately instead of
in production.
"""

from __future__ import annotations

import pytest
from bo_mcp_server.client import (
    CampaignIntakeInput,
    ResultSubmissionInput,
    create_campaign_operation,
    generate_suggestions_operation,
    list_campaigns_operation,
    list_owner_campaigns_with_specs,
    list_results_operation,
    list_suggestions_operation,
    submit_results_operation,
)
from bo_mcp_server.client import list_campaign_results as facade_list_results
from bo_mcp_server.client import list_campaign_suggestions as facade_list_suggestions


async def _make_campaign(owner_id: str, name: str = "Divergence Test") -> str:
    intake = CampaignIntakeInput.model_validate(
        {
            "name": name,
            "description": "",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": {"lower": 0.0, "upper": 1.0}}
            ],
            "objectives": [{"name": "y", "direction": "maximize"}],
            "batch_size": 1,
        }
    )
    result = await create_campaign_operation(intake_data=intake, owner_id=owner_id)
    assert result["success"], result
    return result["campaign_id"]


class TestConvenienceVsOperationDivergence:
    """Same business question, two intentional response shapes."""

    @pytest.mark.asyncio
    async def test_list_campaigns_bare_pairs_vs_paginated_summaries(self, persisted_user) -> None:
        owner_id = str(persisted_user.id)
        await _make_campaign(owner_id, "Alpha")
        await _make_campaign(owner_id, "Beta")

        # Legacy convenience: bare list of (Campaign, CampaignSpec).
        pairs = await list_owner_campaigns_with_specs(persisted_user.id)
        assert len(pairs) == 2
        names = sorted(spec.name for _, spec in pairs)
        assert names == ["Alpha", "Beta"]

        # MCP operation: paginated envelope of summary dicts.
        envelope = await list_campaigns_operation(owner_id=persisted_user.id, verbosity="standard")
        assert envelope["success"] is True
        assert envelope["total_count"] == 2
        for key in ("limit", "offset", "next_cursor", "campaigns"):
            assert key in envelope
        # Summaries — not full entities.
        for summary in envelope["campaigns"]:
            assert "campaign_id" in summary
            assert "name" in summary
            assert "status" in summary

    @pytest.mark.asyncio
    async def test_list_suggestions_bare_entities_vs_envelope(self, persisted_user) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _make_campaign(owner_id, "Suggestion Divergence")
        gen = await generate_suggestions_operation(campaign_id=campaign_id, batch_size=2)
        assert gen["success"], gen

        # Legacy convenience: bare list of full Suggestion entities.
        entities = await facade_list_suggestions(campaign_id, persisted_user.id)
        assert len(entities) >= 1
        first = entities[0]
        assert first.id is not None
        assert first.parameter_values  # full domain object, not a dict subset

        # MCP operation: paginated envelope keyed by ``suggestions``.
        envelope = await list_suggestions_operation(campaign_id=campaign_id, verbosity="standard")
        assert envelope["success"] is True
        assert "suggestions" in envelope
        assert "limit" in envelope and "offset" in envelope

    @pytest.mark.asyncio
    async def test_list_results_bare_entities_vs_envelope(self, persisted_user) -> None:
        owner_id = str(persisted_user.id)
        campaign_id = await _make_campaign(owner_id, "Result Divergence")
        await generate_suggestions_operation(campaign_id=campaign_id, batch_size=1)
        submit = await submit_results_operation(
            campaign_id=campaign_id,
            results=[
                ResultSubmissionInput(
                    parameter_values={"x": 0.5},
                    objective_values={"y": 1.0},
                )
            ],
            submitted_by=owner_id,
            source="api",
        )
        assert submit["success"], submit

        # Legacy convenience: bare list of full Result entities.
        entities = await facade_list_results(campaign_id, persisted_user.id)
        assert len(entities) == 1
        first = entities[0]
        assert first.objective_values == {"y": 1.0}

        # MCP operation: paginated envelope keyed by ``results``.
        envelope = await list_results_operation(campaign_id=campaign_id, verbosity="standard")
        assert envelope["success"] is True
        assert "results" in envelope
        assert envelope["total_count"] == 1
