"""Tests for response_formatter module.

These tests verify that the verbosity-based response formatting correctly
filters responses at each verbosity level.

References:
- Implementation Plan Section 12.1: Response Payload Optimization
- MCP Production Best Practices: https://thinhdanggroup.github.io/mcp-production-ready/
"""

import pytest

from bo_mcp_server.response_formatter import (
    VerbosityLevel,
    format_compare_campaigns_response,
    format_create_campaign_response,
    format_diagnostics_response,
    format_submit_results_response,
    format_suggestions_response,
    format_transfer_candidates_response,
    format_validate_intake_response,
)


class TestVerbosityLevel:
    """Tests for VerbosityLevel enum."""

    def test_verbosity_level_values(self) -> None:
        """Verify VerbosityLevel enum has expected values."""
        assert VerbosityLevel.MINIMAL.value == "minimal"
        assert VerbosityLevel.STANDARD.value == "standard"
        assert VerbosityLevel.DETAILED.value == "detailed"

    def test_verbosity_level_from_string(self) -> None:
        """Verify VerbosityLevel can be created from string."""
        assert VerbosityLevel("minimal") == VerbosityLevel.MINIMAL
        assert VerbosityLevel("standard") == VerbosityLevel.STANDARD
        assert VerbosityLevel("detailed") == VerbosityLevel.DETAILED

    def test_verbosity_level_invalid_string(self) -> None:
        """Verify VerbosityLevel raises ValueError for invalid string."""
        with pytest.raises(ValueError, match="'invalid'"):
            VerbosityLevel("invalid")


class TestFormatDiagnosticsResponse:
    """Tests for format_diagnostics_response function."""

    @pytest.fixture
    def full_diagnostics_response(self) -> dict:
        """Create a complete diagnostics response for testing."""
        return {
            "success": True,
            "campaign_status": "running",
            "iteration": 5,
            "n_results": 10,
            "n_pending_suggestions": 3,
            "errors": [],
            "pareto_front": None,  # Single-objective
            "hypervolume": None,
            "best_value": 0.123,
            "best_parameters": {"x": 1.0, "y": 2.0},
            "improvement_history": [0.5, 0.3, 0.2, 0.15, 0.123],
            "improvement_rate": 0.05,
            "health_status": "healthy",
            "progress_status": "improving",
            "warnings": [],
            "objective_ranges": {"objective": {"min": 0.0, "max": 1.0, "direction": "minimize"}},
            "model_info": {
                "type": "SingleTaskGP",
                "acquisition_function": "noisy_expected_improvement",
            },
            "feature_importance": {"x": 0.6, "y": 0.4},
            "model_correlation": 0.95,
            "hyperparameters": {"lengthscales": {"x": 0.1, "y": 0.2}, "noise_variance": 0.01},
            "loo_cv_metrics": {"objective": {"rmse": 0.05, "mae": 0.04, "r_squared": 0.9}},
            "hypervolume_history": [0.1, 0.2, 0.3, 0.4, 0.5],
            "outliers": {"count": 0, "outlier_results": []},
            "convergence": {"converged": False, "convergence_score": 0.3},
            "uncertainty_trend": {"trend": "decreasing", "slope": -0.01},
            "exploration_exploitation": {"balance_assessment": "balanced"},
            "constraint_satisfaction": None,
            "suggestion_diversity": {"diversity_score": 0.8},
        }

    def test_minimal_verbosity_single_objective(self, full_diagnostics_response: dict) -> None:
        """Verify minimal verbosity returns only key fields for single-objective."""
        result = format_diagnostics_response(full_diagnostics_response, VerbosityLevel.MINIMAL)

        # Should have these fields
        assert result["success"] is True
        assert result["status"] == "running"
        assert result["iteration"] == 5
        assert result["n_results"] == 10
        assert result["health"] == "healthy"
        assert result["progress"] == "improving"
        assert result["key_metric"] == {"best_value": 0.123}
        assert result["converged"] is False
        assert result["errors"] == []

        # Should NOT have these fields
        assert "hyperparameters" not in result
        assert "loo_cv_metrics" not in result
        assert "pareto_front" not in result
        assert "model_info" not in result
        assert "feature_importance" not in result

    def test_minimal_verbosity_multi_objective(self) -> None:
        """Verify minimal verbosity returns hypervolume for multi-objective."""
        multi_obj_response = {
            "success": True,
            "campaign_status": "running",
            "iteration": 3,
            "n_results": 8,
            "health_status": "healthy",
            "progress_status": "stable",
            "pareto_front": [{"x": 0.1, "y": 0.2}],  # Multi-objective indicator
            "hypervolume": 0.456,
            "best_value": None,
            "convergence": {"converged": False},
            "errors": [],
        }

        result = format_diagnostics_response(multi_obj_response, VerbosityLevel.MINIMAL)

        assert result["key_metric"] == {"hypervolume": 0.456}

    def test_standard_verbosity_excludes_debug_fields(
        self, full_diagnostics_response: dict
    ) -> None:
        """Verify standard verbosity excludes hyperparameters and loo_cv."""
        result = format_diagnostics_response(full_diagnostics_response, VerbosityLevel.STANDARD)

        # Should NOT have debug fields
        assert "hyperparameters" not in result
        assert "loo_cv_metrics" not in result
        assert "hypervolume_history" not in result
        assert "outliers" not in result

        # Should have other fields
        assert "success" in result
        assert "feature_importance" in result
        assert "convergence" in result

    def test_detailed_verbosity_returns_all_fields(self, full_diagnostics_response: dict) -> None:
        """Verify detailed verbosity returns complete response.

        The formatter now validates the payload through a Pydantic model
        and returns a fresh dict so the input is not mutated; the result
        therefore contains ``_metadata`` in addition to every field
        present in the input. The key-set assertion also pins that no
        spurious keys leak through the passthrough projection.
        """
        result = format_diagnostics_response(full_diagnostics_response, VerbosityLevel.DETAILED)

        # All input fields survive the round trip, exactly — same keys
        # plus ``_metadata``, nothing else.
        assert set(result) == set(full_diagnostics_response) | {"_metadata", "schema_version"}
        for key, value in full_diagnostics_response.items():
            assert result[key] == value


class TestFormatSuggestionsResponse:
    """Tests for format_suggestions_response function."""

    @pytest.fixture
    def full_suggestions_response(self) -> dict:
        """Create a complete suggestions response for testing."""
        return {
            "success": True,
            "suggestions": [
                {
                    "id": "uuid-1",
                    "parameter_values": {"x": 0.5, "y": 0.3},
                    "provenance": {"iteration": 3, "method": "hypervolume_improvement"},
                    "created_at": "2025-01-01T00:00:00Z",
                },
                {
                    "id": "uuid-2",
                    "parameter_values": {"x": 0.7, "y": 0.9},
                    "provenance": {"iteration": 3, "method": "hypervolume_improvement"},
                    "created_at": "2025-01-01T00:00:01Z",
                },
            ],
            "iteration": 3,
            "errors": [],
            "warnings": [],
            "method_selection": {
                "model_type": "ModelListGP",
                "acquisition_function": "hypervolume_improvement",
                "optimization_strategy": "sequential_greedy",
                "explanation": "Using multi-objective acquisition.",
            },
            "pending_points": {
                "total_pending": 5,
                "valid_pending": 3,
                "stale_expired": 2,
                "note": "3 pending experiments considered.",
            },
            "batch_diversity": {
                "min_pairwise_distance": 0.15,
                "mean_pairwise_distance": 0.25,
                "diversity_score": 0.8,
                "is_diverse": True,
            },
        }

    def test_minimal_verbosity_returns_ids_only(self, full_suggestions_response: dict) -> None:
        """Verify minimal verbosity returns only suggestion IDs."""
        result = format_suggestions_response(full_suggestions_response, VerbosityLevel.MINIMAL)

        assert result["success"] is True
        assert result["iteration"] == 3
        assert result["suggestion_ids"] == ["uuid-1", "uuid-2"]
        assert result["method"] == "hypervolume_improvement"
        assert result["errors"] == []

        # Should NOT have full suggestion details
        assert "suggestions" not in result
        assert "pending_points" not in result
        assert "batch_diversity" not in result

    def test_standard_verbosity_excludes_pending_points(
        self, full_suggestions_response: dict
    ) -> None:
        """Verify standard verbosity excludes pending_points."""
        result = format_suggestions_response(full_suggestions_response, VerbosityLevel.STANDARD)

        assert "pending_points" not in result
        assert "suggestions" in result
        assert "batch_diversity" in result

    def test_detailed_verbosity_returns_all_fields(self, full_suggestions_response: dict) -> None:
        """Verify detailed verbosity returns complete response.

        The formatter validates the payload through Pydantic and returns
        a fresh dict, so the result must carry exactly the input keys
        plus the ``_metadata`` envelope — no spurious leaks.
        """
        result = format_suggestions_response(full_suggestions_response, VerbosityLevel.DETAILED)

        assert set(result) == set(full_suggestions_response) | {"_metadata", "schema_version"}
        for key, value in full_suggestions_response.items():
            assert result[key] == value


class TestFormatCompareCampaignsResponse:
    """Tests for format_compare_campaigns_response function."""

    @pytest.fixture
    def full_compare_response(self) -> dict:
        """Create a complete compare campaigns response for testing."""
        return {
            "success": True,
            "campaigns": [
                {
                    "campaign_id": "uuid-a",
                    "campaign_name": "Campaign A",
                    "status": "running",
                    "n_results": 20,
                    "iteration": 5,
                    "n_parameters": 3,
                    "n_objectives": 1,
                    "is_multi_objective": False,
                    "best_value": 0.1,
                    "improvement_rate": 0.05,
                    "sample_efficiency": 0.002,
                    "hypervolume": None,
                    "n_pareto_points": None,
                },
                {
                    "campaign_id": "uuid-b",
                    "campaign_name": "Campaign B",
                    "status": "completed",
                    "n_results": 30,
                    "iteration": 8,
                    "n_parameters": 3,
                    "n_objectives": 1,
                    "is_multi_objective": False,
                    "best_value": 0.05,
                    "improvement_rate": 0.03,
                    "sample_efficiency": 0.003,
                    "hypervolume": None,
                    "n_pareto_points": None,
                },
            ],
            "comparison": {
                "best_sample_efficiency": "Campaign B",
                "best_single_objective": "Campaign B",
                "best_multi_objective": None,
                "total_campaigns_compared": 2,
                "recommendation": "Campaign B has best efficiency.",
            },
            "errors": [],
        }

    def test_minimal_verbosity_returns_summary(self, full_compare_response: dict) -> None:
        """Verify minimal verbosity returns only summary."""
        result = format_compare_campaigns_response(full_compare_response, VerbosityLevel.MINIMAL)

        assert result["success"] is True
        assert result["n_campaigns"] == 2
        assert result["best_performer"] == "Campaign B"
        assert result["recommendation"] == "Campaign B has best efficiency."
        assert result["errors"] == []

        # Should NOT have full campaign details
        assert "campaigns" not in result
        assert "comparison" not in result

    def test_standard_verbosity_simplifies_campaigns(self, full_compare_response: dict) -> None:
        """Verify standard verbosity simplifies campaign metrics."""
        result = format_compare_campaigns_response(full_compare_response, VerbosityLevel.STANDARD)

        # Should have campaigns but simplified
        assert len(result["campaigns"]) == 2
        first_campaign = result["campaigns"][0]
        assert "campaign_id" in first_campaign
        assert "campaign_name" in first_campaign
        assert "best_value" in first_campaign

        # Should NOT have detailed metrics
        assert "improvement_rate" not in first_campaign
        assert "is_multi_objective" not in first_campaign

    def test_detailed_verbosity_returns_all_fields(self, full_compare_response: dict) -> None:
        """Verify detailed verbosity returns complete response (plus ``_metadata``)."""
        result = format_compare_campaigns_response(full_compare_response, VerbosityLevel.DETAILED)

        assert set(result) == set(full_compare_response) | {"_metadata", "schema_version"}
        for key, value in full_compare_response.items():
            assert result[key] == value


class TestFormatTransferCandidatesResponse:
    """Tests for format_transfer_candidates_response function."""

    @pytest.fixture
    def full_transfer_response(self) -> dict:
        """Create a complete transfer candidates response for testing."""
        return {
            "success": True,
            "target_campaign": {
                "campaign_id": "uuid-target",
                "name": "Target Campaign",
                "n_parameters": 5,
                "n_objectives": 2,
                "parameter_names": ["a", "b", "c", "d", "e"],
                "objective_names": ["obj1", "obj2"],
            },
            "candidates": [
                {
                    "campaign_id": "uuid-source-1",
                    "name": "Source Campaign 1",
                    "status": "completed",
                    "n_results": 50,
                    "iteration": 10,
                    "similarity_score": 0.85,
                    "component_scores": {
                        "parameter_similarity": 0.9,
                        "objective_similarity": 0.8,
                        "bounds_overlap": 0.7,
                        "data_richness": 1.0,
                    },
                    "recommendation": "Highly recommended for transfer learning.",
                },
                {
                    "campaign_id": "uuid-source-2",
                    "name": "Source Campaign 2",
                    "status": "running",
                    "n_results": 20,
                    "iteration": 4,
                    "similarity_score": 0.6,
                    "component_scores": {
                        "parameter_similarity": 0.7,
                        "objective_similarity": 0.5,
                        "bounds_overlap": 0.5,
                        "data_richness": 0.5,
                    },
                    "recommendation": "Recommended for transfer learning.",
                },
            ],
            "overall_recommendation": "Recommend transferring from 'Source Campaign 1'.",
            "errors": [],
        }

    def test_minimal_verbosity_returns_top_candidate(self, full_transfer_response: dict) -> None:
        """Verify minimal verbosity returns only top candidate info."""
        result = format_transfer_candidates_response(full_transfer_response, VerbosityLevel.MINIMAL)

        assert result["success"] is True
        assert result["n_candidates"] == 2
        assert result["top_candidate_id"] == "uuid-source-1"
        assert result["top_similarity"] == 0.85
        assert result["recommendation"] == "Recommend transferring from 'Source Campaign 1'."
        assert result["errors"] == []

        # Should NOT have full candidate details
        assert "candidates" not in result
        assert "target_campaign" not in result

    def test_minimal_verbosity_no_candidates(self) -> None:
        """Verify minimal verbosity handles no candidates."""
        response = {
            "success": True,
            "target_campaign": {"campaign_id": "uuid-target"},
            "candidates": [],
            "overall_recommendation": "No candidates found.",
            "errors": [],
        }

        result = format_transfer_candidates_response(response, VerbosityLevel.MINIMAL)

        assert result["n_candidates"] == 0
        assert result["top_candidate_id"] is None
        assert result["top_similarity"] is None

    def test_standard_verbosity_simplifies_candidates(self, full_transfer_response: dict) -> None:
        """Verify standard verbosity simplifies candidate info."""
        result = format_transfer_candidates_response(
            full_transfer_response, VerbosityLevel.STANDARD
        )

        assert len(result["candidates"]) == 2
        first_candidate = result["candidates"][0]

        # Should have key fields
        assert "campaign_id" in first_candidate
        assert "name" in first_candidate
        assert "similarity_score" in first_candidate
        assert "recommendation" in first_candidate

        # Should NOT have detailed component scores
        assert "component_scores" not in first_candidate
        assert "status" not in first_candidate
        assert "iteration" not in first_candidate

    def test_detailed_verbosity_returns_all_fields(self, full_transfer_response: dict) -> None:
        """Verify detailed verbosity returns complete response (plus ``_metadata``)."""
        result = format_transfer_candidates_response(
            full_transfer_response, VerbosityLevel.DETAILED
        )

        assert set(result) == set(full_transfer_response) | {"_metadata", "schema_version"}
        for key, value in full_transfer_response.items():
            assert result[key] == value


class TestTokenEstimation:
    """Tests verifying approximate token reduction at different verbosity levels.

    These tests use character count as a proxy for token count, validating
    that minimal responses are significantly smaller than detailed responses.
    """

    def test_diagnostics_minimal_is_significantly_smaller(self) -> None:
        """Verify minimal diagnostics response is much smaller than detailed."""
        full_response = {
            "success": True,
            "campaign_status": "running",
            "iteration": 5,
            "n_results": 10,
            "n_pending_suggestions": 3,
            "errors": [],
            "pareto_front": [{"obj1": 0.1, "obj2": 0.2}] * 10,
            "hypervolume": 0.456,
            "best_value": None,
            "health_status": "healthy",
            "progress_status": "improving",
            "hyperparameters": {"lengthscales": {f"x{i}": 0.1 for i in range(10)}},
            "loo_cv_metrics": {"obj1": {"rmse": 0.05}},
            "hypervolume_history": [0.1, 0.2, 0.3] * 10,
            "outliers": {"count": 0, "outlier_results": []},
            "convergence": {"converged": False},
        }

        minimal = format_diagnostics_response(full_response, VerbosityLevel.MINIMAL)
        detailed = format_diagnostics_response(full_response, VerbosityLevel.DETAILED)

        # Minimal should be significantly smaller (at least 50% reduction)
        minimal_size = len(str(minimal))
        detailed_size = len(str(detailed))
        assert minimal_size < detailed_size * 0.5

    def test_suggestions_minimal_is_significantly_smaller(self) -> None:
        """Verify minimal suggestions response is much smaller than detailed."""
        full_response = {
            "success": True,
            "suggestions": [
                {"id": f"uuid-{i}", "parameter_values": {"x": i}, "provenance": {}}
                for i in range(10)
            ],
            "iteration": 5,
            "errors": [],
            "method_selection": {"acquisition_function": "hypervolume_improvement"},
            "pending_points": {"total_pending": 20, "details": "..." * 100},
            "batch_diversity": {"score": 0.8},
        }

        minimal = format_suggestions_response(full_response, VerbosityLevel.MINIMAL)
        detailed = format_suggestions_response(full_response, VerbosityLevel.DETAILED)

        minimal_size = len(str(minimal))
        detailed_size = len(str(detailed))
        assert minimal_size < detailed_size * 0.5


class TestResponseContractValidation:
    """Format outputs validate against their declared Pydantic contracts.

    The formatter now routes every dict through a per-(operation,
    verbosity) Pydantic model, so the dict served to the agent is
    guaranteed to (a) carry the documented keys, (b) carry no spurious
    extras at MINIMAL / STANDARD verbosity, and (c) coerce the simple
    field types (bool / int / str / list[str]) at the transport
    boundary instead of silently passing typos through ``cast`` as the
    previous ``TypedDict`` implementation did.

    Reference: Pydantic v2 ``ConfigDict(extra="forbid")`` and
    ``model_validate`` semantics.
    """

    def test_diagnostics_minimal_rejects_unknown_keys(self) -> None:
        """The strict Minimal model rejects extras when constructed directly."""
        from pydantic import ValidationError

        from bo_mcp_server.response_formatter import DiagnosticsMinimalResponse

        with pytest.raises(ValidationError):
            DiagnosticsMinimalResponse(success=True, mystery_key="oops")  # ty: ignore[unknown-argument]

    def test_create_campaign_minimal_round_trip(self) -> None:
        """A MINIMAL create_campaign response carries exactly the documented keys."""
        result = format_create_campaign_response(
            {
                "success": True,
                "campaign_id": "cmp-123",
                "spec_id": "spec-456",  # excluded at MINIMAL
                "campaign_name": "ignored",  # excluded at MINIMAL
                "warnings": ["w"],
                "errors": [],
            },
            VerbosityLevel.MINIMAL,
        )
        assert set(result) == {
            "success",
            "campaign_id",
            "warnings",
            "errors",
            "field_errors",
            "_metadata",
            "schema_version",
        }
        assert result["campaign_id"] == "cmp-123"

    def test_validate_intake_standard_validates_spec_summary(self) -> None:
        """The STANDARD validate-intake projection materializes a typed spec_summary."""
        full = {
            "valid": True,
            "errors": [],
            "warnings": ["w1"],
            "spec": {
                "name": "campaign",
                "parameters": [{"name": "x"}, {"name": "y"}],
                "objectives": [{"name": "yld"}],
                "constraints": [],
                "batch_size": 4,
            },
        }
        result = format_validate_intake_response(full, VerbosityLevel.STANDARD)
        summary = result["spec_summary"]
        assert summary["name"] == "campaign"
        assert summary["n_parameters"] == 2
        assert summary["n_objectives"] == 1
        assert summary["n_constraints"] == 0
        assert summary["batch_size"] == 4

    def test_submit_results_standard_does_not_leak_duplicates_detail(self) -> None:
        """STANDARD submit_results exposes count, never the full duplicates list."""
        full = {
            "success": True,
            "result_ids": ["r1", "r2"],
            "errors": [],
            "warnings": [],
            "duplicates_detected": [{"index": 1, "matches_existing": "abc"}],
        }
        result = format_submit_results_response(full, VerbosityLevel.STANDARD)
        assert result["n_duplicates_detected"] == 1
        assert "duplicates_detected" not in result


class TestSchemaVersionContract:
    """Every tool envelope advertises a stable ``schema_version``.

    Reference: this is the "single integer the client compares against"
    pattern Stripe documents at https://stripe.com/docs/api/versioning;
    it lets a caller dispatch on the contract without parsing the body
    shape. Bump rules are documented in ``response_formatter.py``.
    """

    def test_success_envelope_includes_schema_version(self) -> None:
        from bo_mcp_server.response_formatter import RESPONSE_SCHEMA_VERSION

        result = format_create_campaign_response(
            {
                "success": True,
                "campaign_id": "cmp-123",
                "warnings": [],
                "errors": [],
            },
            VerbosityLevel.STANDARD,
        )
        assert result["schema_version"] == RESPONSE_SCHEMA_VERSION

    def test_error_envelope_includes_schema_version(self) -> None:
        from bo_mcp_server.errors import ErrorCode, make_error_response
        from bo_mcp_server.response_formatter import RESPONSE_SCHEMA_VERSION

        envelope = make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)
        assert envelope["schema_version"] == RESPONSE_SCHEMA_VERSION
        assert envelope["success"] is False

    def test_schema_version_is_a_positive_integer(self) -> None:
        """The version is a small positive int — clients ``int()``-cast safely."""
        from bo_mcp_server.response_formatter import RESPONSE_SCHEMA_VERSION

        assert isinstance(RESPONSE_SCHEMA_VERSION, int)
        assert RESPONSE_SCHEMA_VERSION >= 1
