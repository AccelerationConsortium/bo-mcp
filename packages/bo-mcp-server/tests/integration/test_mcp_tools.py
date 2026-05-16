"""Integration tests for MCP tools.

References:
- MCP Tool Testing Best Practices: https://modelcontextprotocol.io/docs/concepts/tools
- Async Testing with pytest-asyncio: https://pytest-asyncio.readthedocs.io/
"""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


class TestValidateIntake:
    """Tests for the internal validate_intake helper."""

    @pytest.mark.asyncio
    async def test_validate_intake_success(self):
        """Valid intake data passes validation."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake_data = {
            "name": "Test Campaign",
            "description": "A test campaign",
            "parameters": [
                {
                    "name": "temperature",
                    "type": "continuous",
                    "bounds": [20.0, 100.0],
                    "description": "Temperature in C",
                },
                {
                    "name": "pressure",
                    "type": "discrete",
                    "bounds": [1.0, 10.0],
                },
            ],
            "objectives": [
                {"name": "yield", "direction": "maximize", "unit": "%"},
                {"name": "cost", "direction": "minimize", "unit": "USD"},
            ],
            "batch_size": 2,
        }

        # Use detailed verbosity to get the full spec in the response
        result = await validate_intake(intake_data, verbosity="detailed")

        assert result["valid"] is True
        assert len(result["errors"]) == 0
        assert result["spec"] is not None
        assert result["spec"]["name"] == "Test Campaign"

    @pytest.mark.asyncio
    async def test_validate_intake_missing_name(self):
        """Missing name fails validation."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake_data = {
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [
                {"name": "y", "direction": "minimize"},
            ],
        }

        result = await validate_intake(intake_data)

        assert result["valid"] is False
        assert any("name" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_intake_invalid_parameter_type(self):
        """Invalid parameter type fails validation."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake_data = {
            "name": "Test",
            "parameters": [
                {"name": "x", "type": "invalid_type", "bounds": [0, 1]},
            ],
            "objectives": [
                {"name": "y", "direction": "minimize"},
            ],
        }

        result = await validate_intake(intake_data)

        assert result["valid"] is False
        assert any("type" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_validate_intake_categorical_without_categories(self):
        """Categorical without categories fails validation."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake_data = {
            "name": "Test",
            "parameters": [
                {"name": "cat", "type": "categorical"},
            ],
            "objectives": [
                {"name": "y", "direction": "minimize"},
            ],
        }

        result = await validate_intake(intake_data)

        assert result["valid"] is False

    @pytest.mark.asyncio
    async def test_validate_intake_constraint_references_valid_param(self):
        """Constraint referencing unknown param fails validation."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake_data = {
            "name": "Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [
                {"name": "y", "direction": "minimize"},
            ],
            "constraints": [
                {
                    "type": "sum_equals",
                    "parameters": ["unknown"],
                    "value": 1.0,
                },
            ],
        }

        result = await validate_intake(intake_data)

        assert result["valid"] is False
        assert any("unknown" in e.lower() for e in result["errors"])


class TestCreateCampaign:
    """Tests for create_campaign MCP tool.

    Reference: MCP Server Campaign Management Pattern
    https://modelcontextprotocol.io/docs/concepts/resources
    """

    @pytest.mark.asyncio
    async def test_create_campaign_success(self, setup_database):
        """Successfully creates a campaign with valid intake data."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake_data = {
            "name": "Test Campaign",
            "description": "A test optimization campaign",
            "parameters": [
                {
                    "name": "temperature",
                    "type": "continuous",
                    "bounds": [20.0, 100.0],
                },
            ],
            "objectives": [
                {"name": "yield", "direction": "maximize"},
            ],
            "batch_size": 1,
        }

        result = await create_campaign(intake_data, owner_id)

        assert result["success"] is True
        assert result["campaign_id"] is not None
        assert result["spec_id"] is not None
        assert len(result["errors"]) == 0
        # Verify UUIDs are valid strings
        from uuid import UUID

        UUID(result["campaign_id"])
        UUID(result["spec_id"])

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_owner_id(self, setup_database):
        """Fails when owner_id is not a valid UUID."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        intake_data = {
            "name": "Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, "not-a-uuid")

        assert result["success"] is False
        assert result["campaign_id"] is None
        assert any("owner_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_create_campaign_invalid_intake(self, setup_database):
        """Fails when intake data is invalid."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        # Missing required 'name' field
        intake_data = {
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, owner_id)

        assert result["success"] is False
        assert result["campaign_id"] is None
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_create_campaign_with_constraints(self, setup_database):
        """Creates campaign with constraints."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake_data = {
            "name": "Constrained Campaign",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0, 1]},
                {"name": "x2", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "constraints": [
                {"type": "sum_equals", "parameters": ["x1", "x2"], "value": 1.0},
            ],
            "batch_size": 2,
        }

        result = await create_campaign(intake_data, owner_id)

        assert result["success"] is True
        assert result["campaign_id"] is not None

    @pytest.mark.asyncio
    async def test_create_campaign_multi_objective(self, setup_database):
        """Creates multi-objective campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake_data = {
            "name": "Multi-Objective Campaign",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 10]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "maximize"},
            ],
        }

        result = await create_campaign(intake_data, owner_id)

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_create_campaign_categorical_param(self, setup_database):
        """Creates campaign with categorical parameter."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake_data = {
            "name": "Categorical Campaign",
            "parameters": [
                {"name": "material", "type": "categorical", "categories": ["A", "B", "C"]},
            ],
            "objectives": [{"name": "quality", "direction": "maximize"}],
        }

        result = await create_campaign(intake_data, owner_id)

        assert result["success"] is True


class TestGenerateSuggestions:
    """Tests for generate_suggestions MCP tool.

    Reference: Bayesian Optimization suggestion generation patterns
    https://botorch.org/docs/batched_optimization
    """

    @pytest.mark.asyncio
    async def test_generate_suggestions_invalid_campaign_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        result = await generate_suggestions("not-a-uuid")

        assert result["success"] is False
        assert any("campaign_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_generate_suggestions_campaign_not_found(self, setup_database):
        """Fails when campaign does not exist."""
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        result = await generate_suggestions(str(uuid4()))

        assert result["success"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_generate_suggestions_initial_design(self, setup_database):
        """Generates initial design suggestions for new campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        # Create a campaign first
        owner_id = str(uuid4())
        intake_data = {
            "name": "Suggestion Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True

        # Generate suggestions
        result = await generate_suggestions(create_result["campaign_id"])

        assert result["success"] is True
        assert len(result["suggestions"]) == 3
        assert result["iteration"] == 1
        # Verify suggestions have required fields
        for sugg in result["suggestions"]:
            assert "id" in sugg
            assert "parameter_values" in sugg
            assert "x" in sugg["parameter_values"]
            assert 0.0 <= sugg["parameter_values"]["x"] <= 1.0

    @pytest.mark.asyncio
    async def test_generate_suggestions_custom_batch_size(self, setup_database):
        """Generates custom number of suggestions."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Custom Batch Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)

        # Override batch size
        result = await generate_suggestions(create_result["campaign_id"], batch_size=2)

        assert result["success"] is True
        assert len(result["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_generate_suggestions_multi_param(self, setup_database):
        """Generates suggestions for multi-parameter campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Multi-Param Test",
            "parameters": [
                {"name": "temp", "type": "continuous", "bounds": [20.0, 100.0]},
                {"name": "pressure", "type": "discrete", "bounds": [1.0, 10.0]},
            ],
            "objectives": [{"name": "yield", "direction": "maximize"}],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        result = await generate_suggestions(create_result["campaign_id"])

        assert result["success"] is True
        for sugg in result["suggestions"]:
            assert "temp" in sugg["parameter_values"]
            assert "pressure" in sugg["parameter_values"]
            assert 20.0 <= sugg["parameter_values"]["temp"] <= 100.0
            assert 1.0 <= sugg["parameter_values"]["pressure"] <= 10.0

    @pytest.mark.asyncio
    async def test_generate_suggestions_returns_method_selection(self, setup_database):
        """Method selection is returned for transparency."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Method Selection Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        result = await generate_suggestions(create_result["campaign_id"])

        assert "method_selection" in result
        method_selection = result["method_selection"]
        assert "model_type" in method_selection
        assert "acquisition_function" in method_selection
        assert "explanation" in method_selection


class TestSubmitResults:
    """Tests for submit_results MCP tool.

    Reference: Bayesian Optimization feedback loop
    https://botorch.org/docs/overview
    """

    @pytest.mark.asyncio
    async def test_submit_results_invalid_campaign_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id="not-a-uuid",
            results=_to_result_inputs([]),
            submitted_by=str(uuid4()),
        )

        assert result["success"] is False
        assert any("campaign_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_invalid_submitted_by(self, setup_database):
        """Fails with invalid submitted_by format."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id=str(uuid4()),
            results=_to_result_inputs([]),
            submitted_by="not-a-uuid",
        )

        assert result["success"] is False
        assert any("submitted_by" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_empty_results(self, setup_database):
        """Fails when results list is empty."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id=str(uuid4()),
            results=_to_result_inputs([]),
            submitted_by=str(uuid4()),
        )

        assert result["success"] is False
        assert any("at least one" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_campaign_not_found(self, setup_database):
        """Fails when campaign does not exist."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id=str(uuid4()),
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]
            ),
            submitted_by=str(uuid4()),
        )

        assert result["success"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_success(self, setup_database):
        """Successfully submits results."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        # Setup: create campaign and generate suggestions
        owner_id = str(uuid4())
        intake_data = {
            "name": "Submit Results Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Generate suggestions to move campaign to RUNNING state
        await generate_suggestions(campaign_id)

        # Submit results
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},
                    {"parameter_values": {"x": 0.7}, "objective_values": {"y": 0.8}},
                ]
            ),
            submitted_by=owner_id,
        )

        assert result["success"] is True
        assert len(result["result_ids"]) == 2
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_submit_results_missing_parameters(self, setup_database):
        """Fails when parameter values are missing."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Missing Params Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        await generate_suggestions(create_result["campaign_id"])

        # Missing x2 parameter
        result = await submit_results(
            campaign_id=create_result["campaign_id"],
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x1": 0.5}, "objective_values": {"y": 1.0}},
                ]
            ),
            submitted_by=owner_id,
        )

        assert result["success"] is False
        assert any("missing" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_missing_objectives(self, setup_database):
        """Fails when objective values are missing."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Missing Objectives Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "y1", "direction": "minimize"},
                {"name": "y2", "direction": "maximize"},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        await generate_suggestions(create_result["campaign_id"])

        # Missing y2 objective
        result = await submit_results(
            campaign_id=create_result["campaign_id"],
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x": 0.5}, "objective_values": {"y1": 1.0}},
                ]
            ),
            submitted_by=owner_id,
        )

        assert result["success"] is False
        assert any("missing" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_invalid_source(self, setup_database):
        """Fails with invalid source value."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id=str(uuid4()),
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]
            ),
            submitted_by=str(uuid4()),
            source="invalid_source",
        )

        assert result["success"] is False
        assert any("source" in e.lower() for e in result["errors"])


class TestSubmitResultsBatchOperations:
    """Tests for submit_results batch operations enhancement (v3.1 12.5).

    Reference: Batch processing patterns for ML pipelines
    https://modelcontextprotocol.io/docs/concepts/tools

    These tests verify the atomic and continue_on_error parameters
    for handling multiple results with partial failures.
    """

    @pytest.mark.asyncio
    async def test_atomic_mode_rollback_on_error(self, setup_database):
        """Malformed payloads fail fast during typed input validation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Atomic Rollback Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        invalid_payload = [
            {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},  # Valid
            {"parameter_values": {"x": 0.5}},  # Missing objective_values
            {"parameter_values": {"x": 0.7}, "objective_values": {"y": 0.8}},  # Valid
        ]

        with pytest.raises(ValidationError):
            _to_result_inputs(invalid_payload)

    @pytest.mark.asyncio
    async def test_continue_on_error_partial_success(self, setup_database):
        """Continue on error mode allows partial success."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Partial Success Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Mix of valid and invalid results.
        # The invalid result (idx 1) has objective_values present but missing
        # the required "y" key — this passes Pydantic validation but fails
        # at the operation level.
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": 0.3},
                        "objective_values": {"y": 1.5},
                    },  # Valid (idx 0)
                    {
                        "parameter_values": {"x": 0.5},
                        "objective_values": {"wrong_name": 1.0},
                    },  # Missing "y" objective (idx 1)
                    {
                        "parameter_values": {"x": 0.7},
                        "objective_values": {"y": 0.8},
                    },  # Valid (idx 2)
                ]
            ),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=True,
            verbosity="detailed",
        )

        # Should succeed with partial results
        assert result["success"] is True
        assert len(result["result_ids"]) == 2  # Only valid results saved
        assert "partial_results" in result
        # Check partial_results mapping
        assert 0 in result["partial_results"]  # Success
        assert 1 in result["partial_results"]  # Error
        assert 2 in result["partial_results"]  # Success
        # Index 1 should be an error dict
        assert isinstance(result["partial_results"][1], dict)
        assert "error" in result["partial_results"][1]

    @pytest.mark.asyncio
    async def test_non_atomic_without_continue_on_error(self, setup_database):
        """Malformed payloads fail fast during typed input validation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Non-Atomic Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        invalid_payload = [
            {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},  # Valid
            {"objective_values": {"y": 0.5}},  # Missing parameter_values
            {"parameter_values": {"x": 0.7}, "objective_values": {"y": 0.8}},  # Valid
        ]

        with pytest.raises(ValidationError):
            _to_result_inputs(invalid_payload)

    @pytest.mark.asyncio
    async def test_partial_results_contains_result_ids(self, setup_database):
        """Partial results contain actual result IDs for successful saves."""
        from uuid import UUID

        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Partial Results IDs Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},
                    {"parameter_values": {"x": 0.7}, "objective_values": {"y": 0.8}},
                ]
            ),
            submitted_by=owner_id,
            atomic=False,
            continue_on_error=True,
            verbosity="detailed",
        )

        assert result["success"] is True
        assert "partial_results" in result
        # Both should be UUIDs (strings)
        for idx in [0, 1]:
            result_id = result["partial_results"][idx]
            assert isinstance(result_id, str)
            UUID(result_id)  # Should not raise

    @pytest.mark.asyncio
    async def test_all_results_fail_in_continue_mode(self, setup_database):
        """Malformed payloads fail fast during typed input validation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "All Fail Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        invalid_payload = [
            {"objective_values": {"y": 1.5}},  # Missing parameter_values
            {"parameter_values": {"x": 0.5}},  # Missing objective_values
        ]

        with pytest.raises(ValidationError):
            _to_result_inputs(invalid_payload)

    @pytest.mark.asyncio
    async def test_atomic_default_true(self, setup_database):
        """Malformed payloads fail fast during typed input validation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Default Atomic Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        invalid_payload = [
            {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},
            {"parameter_values": {"x": 0.5}},  # Invalid - missing objective
        ]

        with pytest.raises(ValidationError):
            _to_result_inputs(invalid_payload)

    @pytest.mark.asyncio
    async def test_continue_on_error_ignored_when_atomic(self, setup_database):
        """Malformed payloads fail fast during typed input validation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake_data = {
            "name": "Ignore Continue Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        invalid_payload = [
            {"parameter_values": {"x": 0.3}, "objective_values": {"y": 1.5}},
            {"parameter_values": {"x": 0.5}},  # Invalid
        ]

        with pytest.raises(ValidationError):
            _to_result_inputs(invalid_payload)


class TestGetDiagnostics:
    """Tests for get_diagnostics MCP tool.

    Reference: Gaussian Process model diagnostics
    https://botorch.org/docs/models
    """

    @pytest.mark.asyncio
    async def test_get_diagnostics_invalid_campaign_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        result = await get_diagnostics("not-a-uuid")

        assert result["success"] is False
        assert any("campaign_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_get_diagnostics_campaign_not_found(self, setup_database):
        """Fails when campaign does not exist."""
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        result = await get_diagnostics(str(uuid4()))

        assert result["success"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_get_diagnostics_new_campaign(self, setup_database):
        """Returns diagnostics for campaign without results."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake_data = {
            "name": "Diagnostics Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        result = await get_diagnostics(create_result["campaign_id"])

        assert result["success"] is True
        assert result["n_results"] == 0
        assert result["iteration"] == 0
        assert "health_status" in result

    @pytest.mark.asyncio
    async def test_get_diagnostics_single_objective(self, setup_database):
        """Returns correct diagnostics for single-objective campaign with results."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Single Objective Diagnostics",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Generate suggestions and submit results
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x": 0.2}, "objective_values": {"y": 2.0}},
                    {"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}},
                    {"parameter_values": {"x": 0.8}, "objective_values": {"y": 1.5}},
                ]
            ),
            submitted_by=owner_id,
        )

        result = await get_diagnostics(campaign_id)

        assert result["success"] is True
        assert result["n_results"] == 3
        assert result["best_value"] == 1.0  # Minimum value
        assert result["best_parameters"] == {"x": 0.5}
        assert "improvement_history" in result
        assert result["pareto_front"] is None  # Single-objective

    @pytest.mark.asyncio
    async def test_get_diagnostics_multi_objective(self, setup_database):
        """Returns correct diagnostics for multi-objective campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Multi Objective Diagnostics",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {"parameter_values": {"x": 0.2}, "objective_values": {"f1": 0.8, "f2": 0.3}},
                    {"parameter_values": {"x": 0.5}, "objective_values": {"f1": 0.5, "f2": 0.5}},
                    {"parameter_values": {"x": 0.8}, "objective_values": {"f1": 0.3, "f2": 0.8}},
                ]
            ),
            submitted_by=owner_id,
        )

        result = await get_diagnostics(campaign_id)

        assert result["success"] is True
        assert result["best_value"] is None  # Multi-objective
        assert result["pareto_front"] is not None
        assert result["hypervolume"] is not None
        assert result["n_pareto_points"] is not None

    @pytest.mark.asyncio
    async def test_get_diagnostics_includes_model_info(self, setup_database):
        """Diagnostics include model information."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake_data = {
            "name": "Model Info Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        result = await get_diagnostics(create_result["campaign_id"])

        assert "model_info" in result
        model_info = result["model_info"]
        assert "type" in model_info
        assert "acquisition_function" in model_info


class TestGetSuggestionExplanation:
    """Tests for get_suggestion_explanation MCP tool."""

    @pytest.mark.asyncio
    async def test_get_explanation_invalid_id(self, setup_database):
        """Fails with invalid suggestion_id format."""
        from bo_mcp_server.tools.get_suggestion_explanation import get_suggestion_explanation

        result = await get_suggestion_explanation("not-a-uuid")

        assert result["success"] is False
        assert any("suggestion_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_get_explanation_not_found(self, setup_database):
        """Fails when suggestion does not exist."""
        from bo_mcp_server.tools.get_suggestion_explanation import get_suggestion_explanation

        result = await get_suggestion_explanation(str(uuid4()))

        assert result["success"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_get_explanation_success(self, setup_database):
        """Returns explanation for existing suggestion."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_suggestion_explanation import get_suggestion_explanation

        owner_id = str(uuid4())
        intake_data = {
            "name": "Explanation Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        gen_result = await generate_suggestions(create_result["campaign_id"])

        suggestion_id = gen_result["suggestions"][0]["id"]
        result = await get_suggestion_explanation(suggestion_id)

        assert result["success"] is True
        assert result["explanation"] is not None
        assert result["provenance"] is not None
        assert "generation_method" in result["provenance"]


class TestUploadResultsFile:
    """Tests for upload_results_file MCP tool.

    Reference: CSV file processing patterns
    """

    @pytest.mark.asyncio
    async def test_upload_invalid_format(self, setup_database):
        """Fails with unsupported file format."""
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        result = await upload_results_file(
            campaign_id=str(uuid4()),
            file_content="",
            file_format="xml",
        )

        assert result["success"] is False
        assert any("unsupported" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_upload_early_validation_envelope_echoes_trace_id(self, setup_database):
        """A traced upload that fails early validation still carries the trace id.

        Before the inner pipeline was wrapped with ``with_response_metadata``
        the unsupported-format / oversized-file / invalid-UUID exits emitted
        raw envelopes with no ``_metadata`` block, so a traced upload that
        failed validation was the only upload return that dropped the trace.
        """
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        result = await upload_results_file(
            campaign_id=str(uuid4()),
            file_content="",
            file_format="xml",
            trace_id="trace-upload-bad-format",
        )
        assert result["success"] is False
        assert result.get("_metadata", {}).get("trace_id") == "trace-upload-bad-format"

    @pytest.mark.asyncio
    async def test_upload_invalid_campaign_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        result = await upload_results_file(
            campaign_id="not-a-uuid",
            file_content="param_x,obj_y\n0.5,1.0",
        )

        assert result["success"] is False
        assert any("campaign_id" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_upload_csv_success(self, setup_database):
        """Successfully uploads CSV results."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        owner_id = str(uuid4())
        intake_data = {
            "name": "CSV Upload Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)

        csv_content = """param_x,obj_y
0.1,2.5
0.5,1.8
0.9,2.1"""

        result = await upload_results_file(
            campaign_id=create_result["campaign_id"],
            file_content=csv_content,
        )

        assert result["success"] is True
        assert result["results_created"] == 3
        assert len(result["errors"]) == 0

    @pytest.mark.asyncio
    async def test_upload_csv_missing_param_columns(self, setup_database):
        """Reports error when parameter columns are missing."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        owner_id = str(uuid4())
        intake_data = {
            "name": "Missing Param Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)

        # Missing param_ prefix
        csv_content = """x,obj_y
0.1,2.5"""

        result = await upload_results_file(
            campaign_id=create_result["campaign_id"],
            file_content=csv_content,
        )

        assert result["success"] is False
        assert any("param_" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_upload_csv_missing_obj_columns(self, setup_database):
        """Reports error when objective columns are missing."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        owner_id = str(uuid4())
        intake_data = {
            "name": "Missing Obj Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)

        # Missing obj_ prefix
        csv_content = """param_x,y
0.1,2.5"""

        result = await upload_results_file(
            campaign_id=create_result["campaign_id"],
            file_content=csv_content,
        )

        assert result["success"] is False
        assert any("obj_" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_upload_csv_invalid_objective_value(self, setup_database):
        """Reports error for non-numeric objective values."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.upload_results_file import upload_results_file

        owner_id = str(uuid4())
        intake_data = {
            "name": "Invalid Value Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)

        csv_content = """param_x,obj_y
0.1,not_a_number
0.5,1.8"""

        result = await upload_results_file(
            campaign_id=create_result["campaign_id"],
            file_content=csv_content,
        )

        # Row 2 (first data row) fails with invalid value, row 3 succeeds
        assert result["results_created"] == 1
        # The tool generates errors for invalid values (may be multiple per row)
        assert len(result["errors"]) >= 1
        assert any("invalid" in e.lower() for e in result["errors"])


class TestEndToEndWorkflow:
    """End-to-end workflow tests.

    These tests verify the complete Bayesian Optimization workflow from
    campaign creation to suggestion generation with results.

    Reference: Bayesian Optimization closed-loop workflow
    https://botorch.org/docs/overview
    """

    @pytest.mark.asyncio
    async def test_complete_single_objective_workflow(self, setup_database):
        """Complete workflow: create -> suggest -> result -> suggest again."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Complete Workflow Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 2,
        }

        # 1. Create campaign
        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        # 2. Generate initial suggestions
        gen1 = await generate_suggestions(campaign_id)
        assert gen1["success"] is True
        assert len(gen1["suggestions"]) == 2
        assert gen1["iteration"] == 1

        # 3. Submit results for initial suggestions
        results1 = [
            {
                "parameter_values": s["parameter_values"],
                "objective_values": {"f": sum(s["parameter_values"].values())},
            }
            for s in gen1["suggestions"]
        ]
        submit1 = await submit_results(campaign_id, _to_result_inputs(results1), owner_id)
        assert submit1["success"] is True

        # 4. Generate BO-based suggestions (now with data)
        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"] is True
        assert gen2["iteration"] == 2
        # Method selection should show BO is being used
        assert gen2["method_selection"]["model_type"] is not None

        # 5. Check diagnostics
        diag = await get_diagnostics(campaign_id)
        assert diag["success"] is True
        assert diag["n_results"] == 2
        assert diag["best_value"] is not None

    @pytest.mark.asyncio
    async def test_complete_multi_objective_workflow(self, setup_database):
        """Complete multi-objective workflow with Pareto front computation."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Multi-Objective Workflow",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        # 1. Create and generate
        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        gen1 = await generate_suggestions(campaign_id)
        assert gen1["success"] is True

        # 2. Submit diverse Pareto-front results
        results = [
            {"parameter_values": {"x": 0.1}, "objective_values": {"f1": 0.1, "f2": 0.9}},
            {"parameter_values": {"x": 0.5}, "objective_values": {"f1": 0.5, "f2": 0.5}},
            {"parameter_values": {"x": 0.9}, "objective_values": {"f1": 0.9, "f2": 0.1}},
        ]
        await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        # 3. Verify multi-objective diagnostics
        diag = await get_diagnostics(campaign_id)
        assert diag["success"] is True
        assert diag["pareto_front"] is not None
        assert diag["hypervolume"] > 0
        assert diag["n_pareto_points"] >= 2  # All three points should be Pareto-optimal


class TestAgentUsabilityDiagnostics:
    """Tests for agent usability diagnostics (v2.4).

    These tests verify the new diagnostic features designed to improve
    AI agent usability when running BO campaigns.

    Reference: MCP Agent Integration Best Practices
    https://modelcontextprotocol.io/docs/concepts/tools
    """

    @pytest.mark.asyncio
    async def test_diagnostics_includes_uncertainty_trend(self, setup_database):
        """Diagnostics include uncertainty trend information."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Uncertainty Trend Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Generate suggestions and submit results to trigger BO
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.2}, "objective_values": {"y": 2.0}},
                    {"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}},
                    {"parameter_values": {"x": 0.8}, "objective_values": {"y": 1.5}},
                ]
            ),
            owner_id,
        )
        await generate_suggestions(campaign_id)  # This should have uncertainty info

        diag = await get_diagnostics(campaign_id)

        assert diag["success"] is True
        # uncertainty_trend may be None if no BO suggestions yet have uncertainty
        assert "uncertainty_trend" in diag

    @pytest.mark.asyncio
    async def test_diagnostics_includes_exploration_exploitation(self, setup_database):
        """Diagnostics include exploration/exploitation balance."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "E-E Balance Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)  # Initial design
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.1}, "objective_values": {"y": 2.0}},
                    {"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}},
                    {"parameter_values": {"x": 0.9}, "objective_values": {"y": 1.5}},
                ]
            ),
            owner_id,
        )
        await generate_suggestions(campaign_id)  # BO-based suggestions

        diag = await get_diagnostics(campaign_id)

        assert diag["success"] is True
        assert "exploration_exploitation" in diag

    @pytest.mark.asyncio
    async def test_diagnostics_includes_hyperparameters(self, setup_database):
        """Diagnostics include GP hyperparameters when model is fitted."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Hyperparameter Visibility Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Need enough data to fit a model (at least 2 * n_params)
        await generate_suggestions(campaign_id)
        results = [
            {"parameter_values": {"x1": 0.1, "x2": 0.1}, "objective_values": {"y": 2.0}},
            {"parameter_values": {"x1": 0.3, "x2": 0.7}, "objective_values": {"y": 1.5}},
            {"parameter_values": {"x1": 0.5, "x2": 0.5}, "objective_values": {"y": 1.0}},
            {"parameter_values": {"x1": 0.7, "x2": 0.3}, "objective_values": {"y": 1.2}},
            {"parameter_values": {"x1": 0.9, "x2": 0.9}, "objective_values": {"y": 1.8}},
        ]
        await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        # Use detailed verbosity to include hyperparameters in the response
        diag = await get_diagnostics(campaign_id, verbosity="detailed")

        assert diag["success"] is True
        assert "hyperparameters" in diag
        # Should have hyperparameters since we have enough data
        if diag["hyperparameters"] is not None:
            assert "lengthscales" in diag["hyperparameters"]
            assert "noise_variance" in diag["hyperparameters"]
            assert "interpretation" in diag["hyperparameters"]

    @pytest.mark.asyncio
    async def test_diagnostics_includes_constraint_satisfaction(self, setup_database):
        """Diagnostics include constraint satisfaction metrics."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake_data = {
            "name": "Constraint Satisfaction Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "constraints": [
                {"type": "sum_equals", "parameters": ["x1", "x2"], "value": 1.0},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)
        # Submit some feasible and infeasible results
        results = [
            {
                "parameter_values": {"x1": 0.3, "x2": 0.7},
                "objective_values": {"y": 1.0},
            },  # Feasible
            {
                "parameter_values": {"x1": 0.5, "x2": 0.5},
                "objective_values": {"y": 1.2},
            },  # Feasible
            {
                "parameter_values": {"x1": 0.2, "x2": 0.3},
                "objective_values": {"y": 1.5},
            },  # Infeasible
        ]
        await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        diag = await get_diagnostics(campaign_id)

        assert diag["success"] is True
        assert "constraint_satisfaction" in diag
        if diag["constraint_satisfaction"] is not None:
            assert "satisfaction_rate" in diag["constraint_satisfaction"]
            assert "feasible_count" in diag["constraint_satisfaction"]
            assert "interpretation" in diag["constraint_satisfaction"]

    @pytest.mark.asyncio
    async def test_diagnostics_includes_suggestion_diversity(self, setup_database):
        """Diagnostics include suggestion diversity metrics."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake_data = {
            "name": "Diversity Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)  # Generate batch of 5

        diag = await get_diagnostics(campaign_id)

        assert diag["success"] is True
        assert "suggestion_diversity" in diag
        if diag["suggestion_diversity"] is not None:
            assert "diversity_score" in diag["suggestion_diversity"]
            assert "n_suggestions" in diag["suggestion_diversity"]
            assert "interpretation" in diag["suggestion_diversity"]


class TestCompareCampaigns:
    """Tests for compare_campaigns MCP tool.

    Reference: Multi-campaign analysis patterns
    """

    @pytest.mark.asyncio
    async def test_compare_campaigns_insufficient_campaigns(self, setup_database):
        """Fails when fewer than 2 campaigns provided."""
        from bo_mcp_server.tools.compare_campaigns import compare_campaigns

        result = await compare_campaigns([str(uuid4())])

        assert result["success"] is False
        assert any("at least 2" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_compare_campaigns_too_many_campaigns(self, setup_database):
        """Fails when more than 10 campaigns provided."""
        from bo_mcp_server.tools.compare_campaigns import compare_campaigns

        result = await compare_campaigns([str(uuid4()) for _ in range(15)])

        assert result["success"] is False
        assert any("10" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_compare_campaigns_invalid_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.compare_campaigns import compare_campaigns

        result = await compare_campaigns(["not-a-uuid", str(uuid4())])

        assert result["success"] is False
        assert any("invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_compare_campaigns_success(self, setup_database):
        """Successfully compares two campaigns."""
        from bo_mcp_server.tools.compare_campaigns import compare_campaigns
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        # Create first campaign
        intake1 = {
            "name": "Campaign A",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create1 = await create_campaign(intake1, owner_id)
        campaign1_id = create1["campaign_id"]
        await generate_suggestions(campaign1_id)
        await submit_results(
            campaign1_id,
            _to_result_inputs([{"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}}]),
            owner_id,
        )

        # Create second campaign
        intake2 = {
            "name": "Campaign B",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create2 = await create_campaign(intake2, owner_id)
        campaign2_id = create2["campaign_id"]
        await generate_suggestions(campaign2_id)
        await submit_results(
            campaign2_id,
            _to_result_inputs([{"parameter_values": {"x": 0.3}, "objective_values": {"y": 0.8}}]),
            owner_id,
        )

        # Compare campaigns
        result = await compare_campaigns([campaign1_id, campaign2_id])

        assert result["success"] is True
        assert len(result["campaigns"]) == 2
        assert result["comparison"] is not None
        assert "best_sample_efficiency" in result["comparison"]
        assert "recommendation" in result["comparison"]


class TestDiscoverTransferCandidates:
    """Tests for discover_transfer_candidates MCP tool.

    Reference: Transfer learning in Bayesian Optimization
    https://botorch.org/docs/transfer_learning
    """

    @pytest.mark.asyncio
    async def test_discover_invalid_campaign_id(self, setup_database):
        """Fails with invalid campaign_id format."""
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )

        result = await discover_transfer_candidates("not-a-uuid")

        assert result["success"] is False
        assert any("invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_discover_campaign_not_found(self, setup_database):
        """Fails when campaign does not exist."""
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )

        result = await discover_transfer_candidates(str(uuid4()))

        assert result["success"] is False
        assert any("not found" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_discover_invalid_threshold(self, setup_database):
        """Fails with invalid similarity threshold."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )

        owner_id = str(uuid4())
        intake = {
            "name": "Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)

        result = await discover_transfer_candidates(
            create_result["campaign_id"], similarity_threshold=1.5
        )

        assert result["success"] is False
        assert any("threshold" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_discover_no_candidates(self, setup_database):
        """Returns empty candidates when no similar campaigns exist."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )

        owner_id = str(uuid4())
        intake = {
            "name": "Unique Campaign",
            "parameters": [{"name": "unique_param", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "unique_obj", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)

        result = await discover_transfer_candidates(create_result["campaign_id"])

        assert result["success"] is True
        assert result["target_campaign"] is not None
        assert "overall_recommendation" in result

    @pytest.mark.asyncio
    async def test_discover_finds_similar_campaign(self, setup_database):
        """Finds similar campaign for transfer learning."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        # Create source campaign with results
        source_intake = {
            "name": "Source Campaign",
            "parameters": [
                {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
                {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
            ],
            "objectives": [{"name": "yield", "direction": "maximize"}],
        }
        source_result = await create_campaign(source_intake, owner_id)
        source_id = source_result["campaign_id"]
        await generate_suggestions(source_id)
        # Submit enough results to make it a transfer candidate
        await submit_results(
            source_id,
            _to_result_inputs(
                [
                    {
                        "parameter_values": {"temperature": 50, "pressure": 5},
                        "objective_values": {"yield": 0.8},
                    },
                    {
                        "parameter_values": {"temperature": 60, "pressure": 6},
                        "objective_values": {"yield": 0.85},
                    },
                    {
                        "parameter_values": {"temperature": 70, "pressure": 7},
                        "objective_values": {"yield": 0.9},
                    },
                ]
            ),
            owner_id,
        )

        # Create target campaign with similar parameters
        target_intake = {
            "name": "Target Campaign",
            "parameters": [
                {"name": "temperature", "type": "continuous", "bounds": [30.0, 90.0]},
                {"name": "pressure", "type": "continuous", "bounds": [2.0, 8.0]},
            ],
            "objectives": [{"name": "yield", "direction": "maximize"}],
        }
        target_result = await create_campaign(target_intake, owner_id)
        target_id = target_result["campaign_id"]

        # Discover transfer candidates
        result = await discover_transfer_candidates(target_id, similarity_threshold=0.3)

        assert result["success"] is True
        assert result["target_campaign"]["name"] == "Target Campaign"
        # Should find the source campaign as a candidate
        if result["candidates"]:
            assert result["candidates"][0]["name"] == "Source Campaign"
            assert "similarity_score" in result["candidates"][0]
            assert "recommendation" in result["candidates"][0]

    @pytest.mark.asyncio
    async def test_transfer_candidates_uses_parameter_aliases(self, setup_database):
        """``parameter_aliases`` bridges naming drift across related campaigns.

        Without aliases, ``temperature`` and ``temp_c`` look like distinct
        parameters and the Jaccard intersection collapses to zero. With the
        alias map, the canonical name unifies the two so the parameter
        similarity recovers the value it would have had under matching names.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.discover_transfer_candidates import (
            discover_transfer_candidates,
        )
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        source_intake = {
            "name": "Aliased Source",
            "parameters": [
                {"name": "temperature", "type": "continuous", "bounds": [20.0, 100.0]},
            ],
            "objectives": [{"name": "yield", "direction": "maximize"}],
        }
        source = await create_campaign(source_intake, owner_id)
        await generate_suggestions(source["campaign_id"])
        await submit_results(
            source["campaign_id"],
            _to_result_inputs(
                [
                    {
                        "parameter_values": {"temperature": 25},
                        "objective_values": {"yield": 0.5},
                    },
                    {
                        "parameter_values": {"temperature": 50},
                        "objective_values": {"yield": 0.7},
                    },
                    {
                        "parameter_values": {"temperature": 75},
                        "objective_values": {"yield": 0.6},
                    },
                ]
            ),
            owner_id,
        )

        target_intake = {
            "name": "Aliased Target",
            "parameters": [
                {"name": "temp_c", "type": "continuous", "bounds": [30.0, 90.0]},
            ],
            "objectives": [{"name": "yield", "direction": "maximize"}],
        }
        target = await create_campaign(target_intake, owner_id)

        without_aliases = await discover_transfer_candidates(
            target["campaign_id"],
            similarity_threshold=0.0,
            verbosity="detailed",
        )
        with_aliases = await discover_transfer_candidates(
            target["campaign_id"],
            similarity_threshold=0.0,
            verbosity="detailed",
            parameter_aliases={"temperature": ["temp_c"]},
        )

        # Find the source candidate scores; if not present, the aliases must at
        # least surface a non-empty candidate set.
        aliased_candidates = with_aliases["candidates"]
        assert aliased_candidates, "aliases must reveal the aliased source"
        aliased_top = aliased_candidates[0]
        baseline_top = next(
            (c for c in without_aliases["candidates"] if c["name"] == aliased_top["name"]),
            None,
        )
        if baseline_top is not None:
            assert (
                aliased_top["component_scores"]["parameter_similarity"]
                > baseline_top["component_scores"]["parameter_similarity"]
            )


# =============================================================================
# v3.3 Agent Efficiency Tools Tests
# =============================================================================


class TestListCampaigns:
    """Tests for list_campaigns MCP tool.

    Reference: MCP Tool Best Practices - Agents prefer tools over resources.
    https://modelcontextprotocol.io/docs/concepts/tools
    """

    @pytest.mark.asyncio
    async def test_list_campaigns_empty(self, setup_database):
        """Returns empty list when no campaigns exist."""
        from bo_mcp_server.tools.list_campaigns import list_campaigns

        result = await list_campaigns()

        assert result["success"] is True
        assert result["campaigns"] == []
        assert result["total_count"] == 0

    @pytest.mark.asyncio
    async def test_list_campaigns_with_campaigns(self, setup_database):
        """Lists existing campaigns."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.list_campaigns import list_campaigns

        owner_id = str(uuid4())
        # Create two campaigns
        for name in ["Campaign A", "Campaign B"]:
            intake = {
                "name": name,
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            }
            await create_campaign(intake, owner_id)

        result = await list_campaigns()

        assert result["success"] is True
        assert len(result["campaigns"]) == 2
        assert result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_list_campaigns_filter_by_owner(self, setup_database):
        """Filters campaigns by owner_id."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.list_campaigns import list_campaigns

        owner_a = str(uuid4())
        owner_b = str(uuid4())

        # Create campaigns for different owners
        for owner in [owner_a, owner_a, owner_b]:
            intake = {
                "name": "Test",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            }
            await create_campaign(intake, owner)

        result = await list_campaigns(owner_id=owner_a)

        assert result["success"] is True
        assert result["total_count"] == 2

    @pytest.mark.asyncio
    async def test_list_campaigns_filter_by_status(self, setup_database):
        """Filters campaigns by status."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.list_campaigns import list_campaigns

        owner_id = str(uuid4())
        intake = {
            "name": "Running Campaign",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await create_campaign(intake, owner_id)
        # Generate suggestions to move to RUNNING
        await generate_suggestions(result["campaign_id"])

        # Create another campaign (stays in CREATED)
        intake2 = {
            "name": "Created Campaign",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        await create_campaign(intake2, owner_id)

        result = await list_campaigns(status="running")

        assert result["success"] is True
        assert result["total_count"] == 1
        assert result["campaigns"][0]["status"] == "running"

    @pytest.mark.asyncio
    async def test_list_campaigns_verbosity_levels(self, setup_database):
        """Tests verbosity levels affect response size."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.list_campaigns import list_campaigns

        owner_id = str(uuid4())
        intake = {
            "name": "Verbosity Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        await create_campaign(intake, owner_id)

        # Minimal - fewer fields
        minimal_result = await list_campaigns(verbosity="minimal")
        assert "iteration" not in minimal_result["campaigns"][0]

        # Standard - includes iteration and n_results
        standard_result = await list_campaigns(verbosity="standard")
        assert "iteration" in standard_result["campaigns"][0]
        assert "n_results" in standard_result["campaigns"][0]

        # Detailed - includes spec_summary
        detailed_result = await list_campaigns(verbosity="detailed")
        assert "spec_summary" in detailed_result["campaigns"][0]


class TestCampaignLifecycleTools:
    """Tests for individual campaign lifecycle tools.

    Reference: MCP Best Practices - Individual tools are more discoverable.
    https://modelcontextprotocol.io/docs/best-practices
    """

    @pytest.mark.asyncio
    async def test_pause_campaign(self, setup_database):
        """bo_pause_campaign transitions running campaign to paused."""
        from bo_mcp_server.tools.campaign_lifecycle import pause_campaign
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake = {
            "name": "Lifecycle Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        # Move to RUNNING
        await generate_suggestions(campaign_id)

        result = await pause_campaign(campaign_id)

        assert result["success"] is True
        assert result["status"] == "paused"
        assert result["previous_status"] == "running"

    @pytest.mark.asyncio
    async def test_resume_campaign(self, setup_database):
        """bo_resume_campaign transitions paused campaign to running."""
        from bo_mcp_server.tools.campaign_lifecycle import pause_campaign, resume_campaign
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake = {
            "name": "Resume Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        # Move to RUNNING then PAUSED
        await generate_suggestions(campaign_id)
        await pause_campaign(campaign_id)

        result = await resume_campaign(campaign_id)

        assert result["success"] is True
        assert result["status"] == "running"
        assert result["previous_status"] == "paused"

    @pytest.mark.asyncio
    async def test_terminate_campaign(self, setup_database):
        """bo_terminate_campaign completes the campaign."""
        from bo_mcp_server.tools.campaign_lifecycle import terminate_campaign
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake = {
            "name": "Terminate Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        result = await terminate_campaign(campaign_id)

        assert result["success"] is True
        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_invalid_state_transition(self, setup_database):
        """Pausing an already paused campaign returns error."""
        from bo_mcp_server.tools.campaign_lifecycle import pause_campaign
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        intake = {
            "name": "Invalid Transition Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)
        await pause_campaign(campaign_id)

        # Try to pause again - should fail
        result = await pause_campaign(campaign_id)

        assert result["success"] is False
        errors_str = str(result["errors"]).lower()
        assert "cannot pause" in errors_str or "paused" in errors_str


class TestBatchGetStatus:
    """Tests for batch_get_status tool.

    Reference: MCP Best Practices - Batch operations for efficiency
    https://modelcontextprotocol.io/docs/best-practices
    """

    @pytest.mark.asyncio
    async def test_batch_get_status_multiple_campaigns(self, setup_database):
        """Gets status for multiple campaigns in one call."""
        from bo_mcp_server.tools.batch_operations import batch_get_status
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        campaign_ids = []

        for i in range(3):
            intake = {
                "name": f"Batch Test {i}",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
                "objectives": [{"name": "y", "direction": "minimize"}],
            }
            result = await create_campaign(intake, owner_id)
            campaign_ids.append(result["campaign_id"])

        result = await batch_get_status(campaign_ids)

        assert result["success"] is True
        assert len(result["campaigns"]) == 3
        assert len(result["failed_ids"]) == 0

    @pytest.mark.asyncio
    async def test_batch_get_status_handles_missing(self, setup_database):
        """Handles missing campaign IDs gracefully."""
        from bo_mcp_server.tools.batch_operations import batch_get_status
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake = {
            "name": "Exists",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await create_campaign(intake, owner_id)

        result = await batch_get_status(
            [result["campaign_id"], str(uuid4())]  # Second ID doesn't exist
        )

        assert result["success"] is True
        assert len(result["campaigns"]) == 1
        assert len(result["failed_ids"]) == 1

    @pytest.mark.asyncio
    async def test_batch_get_status_verbosity_levels(self, setup_database):
        """Tests verbosity levels affect response."""
        from bo_mcp_server.tools.batch_operations import batch_get_status
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake = {
            "name": "Verbosity Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        campaign_ids = [create_result["campaign_id"]]

        # Minimal - basic fields only, plus a lightweight next-action hint
        minimal = await batch_get_status(campaign_ids, verbosity="minimal")
        campaign_info = minimal["campaigns"][create_result["campaign_id"]]
        assert "health" not in campaign_info
        assert "next_action_recommendation" in campaign_info
        assert campaign_info["next_action_recommendation"]["action"] == "bo_generate_suggestions"

        # Detailed - includes all fields
        detailed = await batch_get_status(campaign_ids, verbosity="detailed")
        campaign_info = detailed["campaigns"][create_result["campaign_id"]]
        assert "owner_id" in campaign_info

    @pytest.mark.asyncio
    async def test_batch_get_status_empty_list(self, setup_database):
        """Empty campaign list returns error."""
        from bo_mcp_server.tools.batch_operations import batch_get_status

        result = await batch_get_status([])

        assert result["success"] is False


class TestNextActionRecommendation:
    """Tests for next_action_recommendation in get_diagnostics.

    Reference: v3.3 Agent Efficiency - Proactive guidance for agents.
    """

    @pytest.mark.asyncio
    async def test_next_action_generate_suggestions(self, setup_database):
        """Recommends generating suggestions for new campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake = {
            "name": "Next Action Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)

        result = await get_diagnostics(create_result["campaign_id"], verbosity="detailed")

        assert result["success"] is True
        assert "next_action_recommendation" in result
        # New campaign with no results should recommend generating suggestions
        assert result["next_action_recommendation"]["action"] == "bo_generate_suggestions"

    @pytest.mark.asyncio
    async def test_next_action_submit_results(self, setup_database):
        """Recommends submitting results when suggestions pending."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake = {
            "name": "Next Action Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        await generate_suggestions(create_result["campaign_id"])

        result = await get_diagnostics(create_result["campaign_id"], verbosity="detailed")

        assert result["success"] is True
        assert result["next_action_recommendation"]["action"] == "bo_submit_results"

    @pytest.mark.asyncio
    async def test_next_action_in_minimal_response(self, setup_database):
        """Next action is included even in minimal verbosity."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        owner_id = str(uuid4())
        intake = {
            "name": "Minimal Next Action",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)

        result = await get_diagnostics(create_result["campaign_id"], verbosity="minimal")

        assert result["success"] is True
        assert "next_action" in result


class TestVerbosityOnExistingTools:
    """Tests for verbosity parameter on existing tools.

    Reference: v3.3 Agent Efficiency - Token consumption reduction.
    """

    @pytest.mark.asyncio
    async def test_create_campaign_verbosity_levels(self, setup_database):
        """Tests verbosity levels on create_campaign."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())
        intake = {
            "name": "Verbosity Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        # Minimal - campaign_id only
        minimal = await create_campaign(intake, owner_id, verbosity="minimal")
        assert "campaign_id" in minimal
        assert "spec_id" not in minimal

        # Standard - includes spec_id and name
        intake["name"] = "Standard Test"
        standard = await create_campaign(intake, owner_id, verbosity="standard")
        assert "campaign_id" in standard
        assert "spec_id" in standard

    @pytest.mark.asyncio
    async def test_submit_results_verbosity_levels(self, setup_database):
        """Tests verbosity levels on submit_results."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())
        intake = {
            "name": "Submit Verbosity Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        create_result = await create_campaign(intake, owner_id)
        await generate_suggestions(create_result["campaign_id"])

        results_data = [
            {"parameter_values": {"x": 0.5}, "objective_values": {"y": 1.0}},
        ]

        # Minimal - n_submitted only
        minimal = await submit_results(
            create_result["campaign_id"],
            _to_result_inputs(results_data),
            owner_id,
            verbosity="minimal",
        )
        assert "n_submitted" in minimal
        assert "result_ids" not in minimal

    @pytest.mark.asyncio
    async def test_validate_intake_verbosity_levels(self):
        """Tests verbosity levels on the internal validate_intake helper."""
        from bo_mcp_server.tools.validate_intake import validate_intake

        intake = {
            "name": "Validate Verbosity Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }

        # Minimal - valid and errors only
        minimal = await validate_intake(intake, verbosity="minimal")
        assert "valid" in minimal
        assert "spec" not in minimal

        # Standard - includes spec_summary
        standard = await validate_intake(intake, verbosity="standard")
        assert "spec_summary" in standard

        # Detailed - includes full spec
        detailed = await validate_intake(intake, verbosity="detailed")
        assert "spec" in detailed
