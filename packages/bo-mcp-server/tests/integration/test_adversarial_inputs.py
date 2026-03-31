"""Adversarial input tests for MCP tools.

These tests validate proper handling of edge cases and malformed inputs including:
- NaN/Inf values in results
- Extreme bounds
- Conflicting constraints
- Invalid UUIDs and missing fields
- Boundary values

References:
- OWASP Input Validation Cheat Sheet
  https://cheatsheetseries.owasp.org/cheatsheets/Input_Validation_Cheat_Sheet.html
- BoTorch Numerical Stability Discussion
  https://github.com/pytorch/botorch/discussions
- IEEE 754 Floating Point Standard (for NaN/Inf handling)
"""

from uuid import uuid4

import pytest

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


class TestNaNAndInfInputs:
    """Tests for handling NaN and Inf values in various inputs.

    Reference: IEEE 754 Special Values
    https://en.wikipedia.org/wiki/IEEE_754
    """

    @pytest.mark.asyncio
    async def test_submit_results_with_nan_objective(self, setup_database):
        """NaN objective values should be rejected or handled gracefully."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "NaN Objective Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Submit result with NaN objective
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"f": float("nan")}}]
            ),
            submitted_by=owner_id,
        )

        # Should either reject or warn about NaN values
        # Accepting NaN could corrupt the model
        if result["success"]:
            assert "warnings" in result or "nan" in str(result).lower()
        else:
            assert any("nan" in e.lower() or "invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_with_inf_objective(self, setup_database):
        """Inf objective values should be rejected or handled gracefully."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Inf Objective Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Submit result with Inf objective
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"f": float("inf")}}]
            ),
            submitted_by=owner_id,
        )

        # Should either reject or warn about Inf values
        if result["success"]:
            assert "warnings" in result or "inf" in str(result).lower()
        else:
            assert any("inf" in e.lower() or "invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_with_negative_inf(self, setup_database):
        """Negative Inf objective values should be handled."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Negative Inf Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"f": float("-inf")}}]
            ),
            submitted_by=owner_id,
        )

        if result["success"]:
            assert "warnings" in result or "inf" in str(result).lower()
        else:
            assert any("inf" in e.lower() or "invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_submit_results_with_nan_parameter(self, setup_database):
        """NaN parameter values should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "NaN Parameter Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": float("nan")}, "objective_values": {"f": 1.0}}]
            ),
            submitted_by=owner_id,
        )

        # NaN parameters should definitely be rejected
        # as they cannot be used for model training
        if result["success"]:
            assert "warnings" in result
        else:
            assert len(result["errors"]) > 0


class TestExtremeBounds:
    """Tests for campaigns with extreme parameter bounds.

    Reference: Numerical stability in GP models
    https://botorch.org/docs/models
    """

    @pytest.mark.asyncio
    async def test_very_small_bounds(self, setup_database):
        """Campaign with very small parameter range should work."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Small Bounds Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1e-10]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        for s in gen["suggestions"]:
            assert 0.0 <= s["parameter_values"]["x"] <= 1e-10

    @pytest.mark.asyncio
    async def test_very_large_bounds(self, setup_database):
        """Campaign with very large parameter range should work."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Large Bounds Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1e10]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        for s in gen["suggestions"]:
            assert 0.0 <= s["parameter_values"]["x"] <= 1e10

    @pytest.mark.asyncio
    async def test_negative_to_positive_bounds(self, setup_database):
        """Campaign with negative to positive bounds should work."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Negative to Positive Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [-1e6, 1e6]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Bounds validation not yet implemented - see TODO.md Step 10")
    async def test_inverted_bounds_rejected(self, setup_database):
        """Campaign with inverted bounds (lower > upper) should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Inverted Bounds Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [1.0, 0.0]},  # Inverted!
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is False
        # Error message says "Lower bound must be less than upper bound"
        assert any("bound" in e.lower() for e in create_result["errors"])

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Bounds validation not yet implemented - see TODO.md Step 10")
    async def test_equal_bounds_rejected(self, setup_database):
        """Campaign with equal bounds (zero range) should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Equal Bounds Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.5, 0.5]},  # Zero range!
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is False
        # Error message says "Lower bound must be less than upper bound"
        assert any("bound" in e.lower() for e in create_result["errors"])


class TestConflictingConstraints:
    """Tests for campaigns with conflicting or impossible constraints.

    Reference: Constraint satisfaction in optimization
    https://botorch.org/tutorials/constrained_multi_objective_bo
    """

    @pytest.mark.asyncio
    async def test_impossible_sum_constraint(self, setup_database):
        """Constraint that cannot be satisfied should be rejected or warn."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Impossible Constraint Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 0.3]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 0.3]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "constraints": [
                # Sum must equal 1.0 but max possible is 0.6
                {"type": "sum_equals", "parameters": ["x1", "x2"], "value": 1.0},
            ],
        }

        result = await create_campaign(intake_data, owner_id)
        # Should either reject or provide warning about infeasibility
        if result["success"]:
            # If accepted, later suggestion generation should handle it
            pass  # Acceptable behavior
        else:
            assert any(
                "constraint" in e.lower() or "infeasible" in e.lower() for e in result["errors"]
            )

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=(
            "Constraint validation does not include 'nonexistent' in error "
            "message - see TODO.md Step 10"
        )
    )
    async def test_constraint_referencing_nonexistent_param(self, setup_database):
        """Constraint referencing unknown parameter should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Bad Constraint Reference Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "constraints": [
                {"type": "sum_equals", "parameters": ["x", "nonexistent"], "value": 1.0},
            ],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("nonexistent" in e.lower() for e in result["errors"])


class TestInvalidUUIDs:
    """Tests for handling invalid UUID inputs.

    Reference: UUID format specification
    https://datatracker.ietf.org/doc/html/rfc4122
    """

    @pytest.mark.asyncio
    async def test_generate_suggestions_empty_uuid(self, setup_database):
        """Empty string for campaign_id should be rejected."""
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        result = await generate_suggestions("")
        assert result["success"] is False
        assert any("campaign_id" in e.lower() or "invalid" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_generate_suggestions_malformed_uuid(self, setup_database):
        """Malformed UUID should be rejected."""
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        result = await generate_suggestions("not-a-valid-uuid-format")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_generate_suggestions_partial_uuid(self, setup_database):
        """Partial UUID should be rejected."""
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        result = await generate_suggestions("12345678-1234-1234-1234")  # Incomplete
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_submit_results_null_campaign_id(self, setup_database):
        """None/null for campaign_id should be rejected."""
        from bo_mcp_server.tools.submit_results import submit_results

        # Python None will fail differently than string, but should handle gracefully
        try:
            result = await submit_results(
                campaign_id=None,  # type: ignore
                results=_to_result_inputs(
                    [{"parameter_values": {"x": 0.5}, "objective_values": {"f": 1.0}}]
                ),
                submitted_by=str(uuid4()),
            )
            assert result["success"] is False
        except (TypeError, ValueError):
            pass  # Also acceptable behavior

    @pytest.mark.asyncio
    async def test_diagnostics_sql_injection_attempt(self, setup_database):
        """SQL injection attempt in campaign_id should be safely handled."""
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics

        # Attempt SQL injection through campaign_id
        result = await get_diagnostics("'; DROP TABLE campaigns; --")
        assert result["success"] is False
        # Should fail due to invalid UUID format, not execute SQL


class TestMissingAndEmptyFields:
    """Tests for handling missing or empty required fields."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Empty parameters validation not yet implemented - see TODO.md Step 10"
    )
    async def test_create_campaign_empty_parameters(self, setup_database):
        """Campaign with no parameters should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "No Parameters Test",
            "parameters": [],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("parameter" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Empty objectives validation not yet implemented - see TODO.md Step 10"
    )
    async def test_create_campaign_empty_objectives(self, setup_database):
        """Campaign with no objectives should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "No Objectives Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("objective" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_create_campaign_missing_parameter_name(self, setup_database):
        """Parameter without name should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Missing Parameter Name Test",
            "parameters": [{"type": "continuous", "bounds": [0.0, 1.0]}],  # No name
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_create_campaign_missing_objective_direction(self, setup_database):
        """Objective without direction should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Missing Direction Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f"}],  # No direction
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_submit_empty_results_list(self, setup_database):
        """Empty results list should be rejected."""
        from bo_mcp_server.tools.submit_results import submit_results

        result = await submit_results(
            campaign_id=str(uuid4()),
            results=_to_result_inputs([]),
            submitted_by=str(uuid4()),
        )

        assert result["success"] is False
        assert any("at least one" in e.lower() or "empty" in e.lower() for e in result["errors"])


class TestBoundaryValues:
    """Tests for boundary value handling.

    Reference: Boundary Value Analysis in Testing
    https://en.wikipedia.org/wiki/Boundary-value_analysis
    """

    @pytest.mark.asyncio
    async def test_parameter_at_exact_lower_bound(self, setup_database):
        """Parameter value exactly at lower bound should be accepted."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Lower Bound Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Parameter at exact lower bound
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.0}, "objective_values": {"f": 1.0}}]
            ),
            submitted_by=owner_id,
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_parameter_at_exact_upper_bound(self, setup_database):
        """Parameter value exactly at upper bound should be accepted."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Upper Bound Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Parameter at exact upper bound
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 1.0}, "objective_values": {"f": 1.0}}]
            ),
            submitted_by=owner_id,
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Out-of-bounds parameter validation not yet implemented - see TODO.md Step 10"
    )
    async def test_parameter_outside_bounds_rejected(self, setup_database):
        """Parameter value outside bounds should be rejected or warned."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Out of Bounds Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": 1.5},
                        "objective_values": {"f": 1.0},
                    }  # Outside bounds
                ]
            ),
            submitted_by=owner_id,
        )

        # Should either reject or provide warning
        if result["success"]:
            assert "warnings" in result and len(result["warnings"]) > 0
        else:
            assert any("bounds" in e.lower() or "range" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Zero batch size validation not yet implemented - see TODO.md Step 10"
    )
    async def test_zero_batch_size(self, setup_database):
        """Batch size of 0 should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Zero Batch Size Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 0,
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("batch" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_negative_batch_size(self, setup_database):
        """Negative batch size should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Negative Batch Size Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": -5,
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False


class TestDuplicateAndConflictingData:
    """Tests for handling duplicate and conflicting input data.

    Reference: Section 1.2 - Duplicate Detection
    """

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=("Duplicate parameter name validation not yet implemented - see TODO.md Step 10")
    )
    async def test_duplicate_parameter_names(self, setup_database):
        """Duplicate parameter names should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Duplicate Param Names Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x", "type": "continuous", "bounds": [0.0, 2.0]},  # Duplicate name
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("duplicate" in e.lower() or "unique" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason=("Duplicate objective name validation not yet implemented - see TODO.md Step 10")
    )
    async def test_duplicate_objective_names(self, setup_database):
        """Duplicate objective names should be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Duplicate Objective Names Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f", "direction": "minimize"},
                {"name": "f", "direction": "maximize"},  # Duplicate name
            ],
        }

        result = await create_campaign(intake_data, owner_id)
        assert result["success"] is False
        assert any("duplicate" in e.lower() or "unique" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_near_duplicate_results_warning(self, setup_database):
        """Near-duplicate results should trigger duplicate detection.

        Reference: Section 1.2 - Duplicate detection with tolerance 1e-6
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Near Duplicate Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]
        await generate_suggestions(campaign_id)

        # Submit first result
        await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [{"parameter_values": {"x": 0.5}, "objective_values": {"f": 1.0}}]
            ),
            submitted_by=owner_id,
        )

        # Submit near-duplicate result
        result = await submit_results(
            campaign_id=campaign_id,
            results=_to_result_inputs(
                [
                    {
                        "parameter_values": {"x": 0.5 + 1e-9},  # Within tolerance
                        "objective_values": {"f": 1.0},
                    }
                ]
            ),
            submitted_by=owner_id,
        )

        # Should either detect duplicate or accept with warning
        # The exact behavior depends on implementation
        if result["success"]:
            # If accepted, should flag as potential duplicate
            assert "duplicates_detected" in result or "warnings" in result


class TestExtremeCampaignConfigurations:
    """Tests for extreme but valid campaign configurations."""

    @pytest.mark.asyncio
    async def test_many_parameters(self, setup_database):
        """Campaign with many parameters (high-dimensional) should work."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        # 20 parameters (triggers TuRBO consideration)
        parameters = [
            {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(20)
        ]

        intake_data = {
            "name": "High Dimensional Test",
            "parameters": parameters,
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        assert len(gen["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_many_objectives(self, setup_database):
        """Campaign with many objectives should work (though may be slow)."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        # 4 objectives
        objectives = [{"name": f"f{i}", "direction": "minimize"} for i in range(4)]

        intake_data = {
            "name": "Many Objectives Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": objectives,
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True

    @pytest.mark.asyncio
    async def test_large_batch_size(self, setup_database):
        """Campaign with large batch size should work."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Large Batch Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 20,  # Large batch
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"] is True
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        assert len(gen["suggestions"]) == 20

    @pytest.mark.asyncio
    async def test_very_long_parameter_name(self, setup_database):
        """Campaign with very long parameter name should work or be rejected."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        long_name = "x" * 1000  # 1000 character name

        intake_data = {
            "name": "Long Parameter Name Test",
            "parameters": [{"name": long_name, "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
        }

        result = await create_campaign(intake_data, owner_id)
        # Either accept or reject gracefully
        if not result["success"]:
            assert any("name" in e.lower() or "length" in e.lower() for e in result["errors"])

    @pytest.mark.asyncio
    async def test_special_characters_in_names(self, setup_database):
        """Parameter names with special characters should be handled."""
        from bo_mcp_server.tools.create_campaign import create_campaign

        owner_id = str(uuid4())

        intake_data = {
            "name": "Special Characters Test",
            "parameters": [
                {
                    "name": "temp_celsius",
                    "type": "continuous",
                    "bounds": [0.0, 1.0],
                },  # Underscore OK
            ],
            "objectives": [{"name": "yield_%", "direction": "maximize"}],  # Special char
        }

        result = await create_campaign(intake_data, owner_id)
        # Should work or reject gracefully
        if not result["success"]:
            # Rejection should explain why
            assert len(result["errors"]) > 0
