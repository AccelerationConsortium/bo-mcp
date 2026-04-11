"""Unit tests for the validate_intake operation.

Tests cover the protocol-neutral validation logic independently of MCP/HTTP
transport. The operation accepts CampaignIntakeInput or raw dicts and returns
the full canonical response (no verbosity filtering — that's the transport
layer's job).

Reference: Pydantic validation patterns for scientific experiment specs.
"""

import pytest

from bo_mcp_server.domain import CampaignIntakeInput
from bo_mcp_server.operations.validate_intake import validate_intake_operation


def _make_valid_intake(
    name: str = "Test Campaign",
    n_params: int = 2,
    n_objectives: int = 1,
) -> dict:
    """Build a valid intake dict with configurable size."""
    parameters = [
        {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(n_params)
    ]
    objectives = [{"name": f"y{i}", "direction": "minimize"} for i in range(n_objectives)]
    return {
        "name": name,
        "description": "A test campaign",
        "parameters": parameters,
        "objectives": objectives,
    }


class TestValidateIntakeOperation:
    """Tests for validate_intake_operation."""

    @pytest.mark.asyncio
    async def test_valid_intake_dict(self):
        """Valid intake dict passes validation and returns full spec."""
        intake = _make_valid_intake()
        result = await validate_intake_operation(intake)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["warnings"] == []
        assert result["spec"] is not None
        assert result["spec"]["name"] == "Test Campaign"

    @pytest.mark.asyncio
    async def test_valid_intake_model(self):
        """Valid CampaignIntakeInput instance passes validation."""
        intake = CampaignIntakeInput.model_validate(_make_valid_intake())
        result = await validate_intake_operation(intake)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["spec"]["name"] == "Test Campaign"

    @pytest.mark.asyncio
    async def test_returns_full_response_always(self):
        """Operation always returns the full response (no verbosity filtering)."""
        result = await validate_intake_operation(_make_valid_intake())

        assert "valid" in result
        assert "errors" in result
        assert "warnings" in result
        assert "spec" in result

    @pytest.mark.asyncio
    async def test_invalid_missing_name(self):
        """Missing required name field fails validation."""
        intake = {
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await validate_intake_operation(intake)

        assert result["valid"] is False
        assert len(result["errors"]) > 0
        assert result["spec"] is None
        assert result["warnings"] == []

    @pytest.mark.asyncio
    async def test_invalid_parameter_type(self):
        """Invalid parameter type fails Pydantic validation."""
        intake = {
            "name": "Test",
            "parameters": [
                {"name": "x", "type": "invalid_type", "bounds": [0, 1]},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await validate_intake_operation(intake)

        assert result["valid"] is False
        assert any("type" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_invalid_missing_objectives(self):
        """Missing objectives field fails validation."""
        intake = {
            "name": "Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0, 1]},
            ],
        }
        result = await validate_intake_operation(intake)

        assert result["valid"] is False
        assert len(result["errors"]) > 0

    @pytest.mark.asyncio
    async def test_warning_many_objectives(self):
        """More than 4 objectives triggers a warning."""
        intake = _make_valid_intake(n_objectives=5)
        result = await validate_intake_operation(intake)

        assert result["valid"] is True
        assert any("objectives" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_warning_many_parameters(self):
        """More than 20 parameters triggers a warning."""
        intake = _make_valid_intake(n_params=21)
        result = await validate_intake_operation(intake)

        assert result["valid"] is True
        assert any("parameters" in w.lower() for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_no_warnings_under_thresholds(self):
        """No warnings when under both thresholds."""
        intake = _make_valid_intake(n_params=2, n_objectives=2)
        result = await validate_intake_operation(intake)

        assert result["valid"] is True
        assert result["warnings"] == []

    @pytest.mark.asyncio
    async def test_spec_contains_expected_fields(self):
        """Returned spec dict has the expected structure."""
        intake = _make_valid_intake()
        result = await validate_intake_operation(intake)

        spec = result["spec"]
        assert "name" in spec
        assert "parameters" in spec
        assert "objectives" in spec
        assert "batch_size" in spec

    @pytest.mark.asyncio
    async def test_categorical_without_categories_fails(self):
        """Categorical parameter without categories list fails validation."""
        intake = {
            "name": "Test",
            "parameters": [
                {"name": "cat", "type": "categorical"},
            ],
            "objectives": [{"name": "y", "direction": "minimize"}],
        }
        result = await validate_intake_operation(intake)

        assert result["valid"] is False
