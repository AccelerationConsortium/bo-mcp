"""Tests for transport-neutral operations shared by MCP and HTTP."""

from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput
from bo_mcp_server.operations.batch_status import batch_get_status_operation
from bo_mcp_server.operations.campaign_lifecycle import manage_campaign_lifecycle_operation
from bo_mcp_server.operations.compare_campaigns import compare_campaigns_operation
from bo_mcp_server.operations.suggestion_explanation import (
    get_suggestion_explanation_operation,
)
from bo_mcp_server.operations.transfer_candidates import (
    discover_transfer_candidates_operation,
)
from bo_mcp_server.tools.batch_operations import batch_get_status
from bo_mcp_server.tools.campaign_lifecycle import pause_campaign
from bo_mcp_server.tools.compare_campaigns import compare_campaigns
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.discover_transfer_candidates import discover_transfer_candidates
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_suggestion_explanation import get_suggestion_explanation
from bo_mcp_server.tools.submit_results import submit_results


def _to_result_inputs(rows: list[dict]) -> list[ResultSubmissionInput]:
    return [
        ResultSubmissionInput(
            parameter_values=row["parameter_values"],
            objective_values=row["objective_values"],
            suggestion_id=row.get("suggestion_id"),
            metadata=row.get("metadata", {}),
        )
        for row in rows
    ]


async def _create_single_objective_campaign(owner_id: str, name: str) -> str:
    result = await create_campaign(
        {
            "name": name,
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        },
        owner_id,
    )
    return result["campaign_id"]


class TestSharedOperations:
    @pytest.mark.asyncio
    async def test_lifecycle_operation_matches_tool(self, setup_database):
        owner_id = str(uuid4())
        operation_campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Lifecycle Shared Op Test",
        )
        tool_campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Lifecycle Tool Test",
        )
        await generate_suggestions(operation_campaign_id)
        await generate_suggestions(tool_campaign_id)

        operation_result = await manage_campaign_lifecycle_operation(
            operation_campaign_id,
            "pause",
        )
        tool_result = await pause_campaign(tool_campaign_id)

        assert operation_result["success"] is True
        assert tool_result["success"] is True
        assert tool_result["status"] == "paused"
        assert operation_result["status"] == "paused"
        assert tool_result["previous_status"] == operation_result["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_suggestion_explanation_operation_matches_tool(self, setup_database):
        owner_id = str(uuid4())
        campaign_id = await _create_single_objective_campaign(
            owner_id,
            "Explanation Shared Op Test",
        )
        generated = await generate_suggestions(campaign_id)
        suggestion_id = generated["suggestions"][0]["id"]

        operation_result = await get_suggestion_explanation_operation(suggestion_id)
        tool_result = await get_suggestion_explanation(suggestion_id)

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_batch_status_operation_matches_tool(self, setup_database):
        owner_id = str(uuid4())
        campaign_ids = [
            await _create_single_objective_campaign(owner_id, "Batch Shared Op Test A"),
            await _create_single_objective_campaign(owner_id, "Batch Shared Op Test B"),
        ]

        operation_result = await batch_get_status_operation(campaign_ids, verbosity="standard")
        tool_result = await batch_get_status(campaign_ids, verbosity="standard")

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_compare_operation_matches_tool(self, setup_database):
        owner_id = str(uuid4())
        campaign_a = await _create_single_objective_campaign(owner_id, "Compare Shared Op Test A")
        campaign_b = await _create_single_objective_campaign(owner_id, "Compare Shared Op Test B")

        await generate_suggestions(campaign_a)
        await generate_suggestions(campaign_b)
        await submit_results(
            campaign_a,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )
        await submit_results(
            campaign_b,
            _to_result_inputs([{"parameter_values": {"x": 0.3}, "objective_values": {"y": 0.8}}]),
            owner_id,
        )

        operation_result = await compare_campaigns_operation(
            [campaign_a, campaign_b],
            verbosity="standard",
        )
        tool_result = await compare_campaigns([campaign_a, campaign_b], verbosity="standard")

        assert operation_result == tool_result

    @pytest.mark.asyncio
    async def test_transfer_candidates_operation_matches_tool(self, setup_database):
        owner_id = str(uuid4())
        source = await create_campaign(
            {
                "name": "Shared Op Source",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )
        target = await create_campaign(
            {
                "name": "Shared Op Target",
                "parameters": [
                    {"name": "temperature", "type": "continuous", "bounds": [30.0, 90.0]},
                    {"name": "pressure", "type": "continuous", "bounds": [2.0, 8.0]},
                ],
                "objectives": [{"name": "yield", "direction": "maximize"}],
            },
            owner_id,
        )

        await generate_suggestions(source["campaign_id"])
        await submit_results(
            source["campaign_id"],
            _to_result_inputs(
                [
                    {
                        "parameter_values": {"temperature": 50.0, "pressure": 5.0},
                        "objective_values": {"yield": 0.8},
                    },
                    {
                        "parameter_values": {"temperature": 60.0, "pressure": 6.0},
                        "objective_values": {"yield": 0.85},
                    },
                    {
                        "parameter_values": {"temperature": 70.0, "pressure": 7.0},
                        "objective_values": {"yield": 0.9},
                    },
                ]
            ),
            owner_id,
        )

        operation_result = await discover_transfer_candidates_operation(
            target["campaign_id"],
            similarity_threshold=0.3,
            verbosity="standard",
        )
        tool_result = await discover_transfer_candidates(
            target["campaign_id"],
            similarity_threshold=0.3,
            verbosity="standard",
        )

        assert operation_result == tool_result
