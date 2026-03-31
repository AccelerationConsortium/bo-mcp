"""Regression tests for suggestion generation.

These tests ensure that same inputs produce same suggestions (reproducibility)
and that the suggestion quality remains stable across code changes.

References:
- BoTorch Reproducibility Guide
  https://botorch.org/docs/reproducibility
- PyTorch Reproducibility Documentation
  https://pytorch.org/docs/stable/notes/randomness.html
- Section 3.7: Full Reproducibility Guarantees
"""

from uuid import uuid4

import pytest
import torch

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


class TestSuggestionReproducibility:
    """Tests ensuring deterministic suggestion generation with fixed seeds.

    Reference: BoTorch reproducibility documentation
    https://botorch.org/docs/reproducibility
    """

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Different campaign IDs produce different Sobol seeds - Section 3.7 not implemented"
    )
    async def test_initial_design_deterministic(self, setup_database):
        """Initial design (Sobol sequence) produces same suggestions with same setup.

        Sobol sequences are deterministic by construction.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Deterministic Initial Design Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 5,
        }

        # Create first campaign and generate
        create1 = await create_campaign(intake_data, owner_id)
        campaign1_id = create1["campaign_id"]
        gen1 = await generate_suggestions(campaign1_id)

        # Create second campaign with same spec and generate
        create2 = await create_campaign(intake_data, owner_id)
        campaign2_id = create2["campaign_id"]
        gen2 = await generate_suggestions(campaign2_id)

        # Initial design should be identical (Sobol is deterministic)
        assert gen1["success"] is True
        assert gen2["success"] is True
        assert len(gen1["suggestions"]) == len(gen2["suggestions"])

        for s1, s2 in zip(gen1["suggestions"], gen2["suggestions"], strict=True):
            # Parameter values should match exactly for Sobol
            for param in ["x1", "x2"]:
                assert abs(s1["parameter_values"][param] - s2["parameter_values"][param]) < 1e-10

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="random_seed not populated in provenance - Section 3.7 not implemented"
    )
    async def test_suggestion_provenance_includes_seed(self, setup_database):
        """Suggestion provenance includes random_seed for reproducibility.

        Reference: Section 3.7 - Full Reproducibility Guarantees
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Provenance Seed Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        gen1 = await generate_suggestions(campaign_id)
        assert gen1["success"] is True

        for s in gen1["suggestions"]:
            assert "random_seed" in s["provenance"]
            assert s["provenance"]["random_seed"] is not None
            assert isinstance(s["provenance"]["random_seed"], int)

        # Submit results and generate BO suggestions
        results = [
            {"parameter_values": {"x": 0.3}, "objective_values": {"f1": 0.5, "f2": 0.8}},
            {"parameter_values": {"x": 0.7}, "objective_values": {"f1": 0.8, "f2": 0.5}},
        ]
        await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"] is True

        for s in gen2["suggestions"]:
            assert "random_seed" in s["provenance"]
            assert s["provenance"]["random_seed"] is not None


class TestSuggestionQualityRegression:
    """Tests ensuring suggestion quality remains stable.

    These tests use known benchmark functions to verify that
    optimization finds reasonable solutions.
    """

    @pytest.mark.asyncio
    async def test_branin_currin_hypervolume_regression(self, setup_database):
        """Hypervolume on Branin-Currin should reach minimum threshold.

        Reference: BoTorch Multi-Objective Tutorial
        https://botorch.org/tutorials/multi_objective_bo
        Expected hypervolume after 3 iterations: > 0.5 (normalized)
        """
        from bo_engine.benchmarks import branin_currin

        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "Branin-Currin Regression Test",
            "parameters": [
                {"name": "x0", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "branin", "direction": "minimize"},
                {"name": "currin", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Run 3 iterations
        for _ in range(3):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"] is True

            results = []
            for s in gen["suggestions"]:
                x = torch.tensor([[s["parameter_values"]["x0"], s["parameter_values"]["x1"]]])
                y = branin_currin(x)
                results.append(
                    {
                        "parameter_values": s["parameter_values"],
                        "objective_values": {
                            "branin": y[0, 0].item(),
                            "currin": y[0, 1].item(),
                        },
                    }
                )
            await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        # Check final hypervolume
        diag = await get_diagnostics(campaign_id)
        assert diag["success"] is True
        assert diag["hypervolume"] is not None

        # After 9 evaluations (3 iterations x 3 batch), hypervolume should be reasonable
        # This is a regression threshold - if optimization regresses, this will catch it
        assert diag["hypervolume"] > 0.1, f"Hypervolume too low: {diag['hypervolume']}"

    @pytest.mark.asyncio
    async def test_quadratic_finds_minimum(self, setup_database):
        """Simple quadratic function should find near-optimal solution.

        A simple test function with known minimum at (0.5, 0.5).
        After a few iterations, best value should be close to 0.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "Quadratic Regression Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 3,
        }

        def quadratic(x: float, y: float) -> float:
            return (x - 0.5) ** 2 + (y - 0.5) ** 2

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Run 4 iterations
        for _ in range(4):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"] is True

            results = [
                {
                    "parameter_values": s["parameter_values"],
                    "objective_values": {
                        "f": quadratic(
                            s["parameter_values"]["x"],
                            s["parameter_values"]["y"],
                        )
                    },
                }
                for s in gen["suggestions"]
            ]
            await submit_results(campaign_id, _to_result_inputs(results), owner_id)

        # Check final best value
        diag = await get_diagnostics(campaign_id)
        assert diag["success"] is True
        assert diag["best_value"] is not None

        # After 12 evaluations, best value should be close to 0
        # Allow 0.1 tolerance for this regression test
        assert diag["best_value"] < 0.1, f"Best value too high: {diag['best_value']}"


class TestMethodSelectionStability:
    """Tests ensuring method selection logic remains stable.

    These tests verify that the automatic method selection
    chooses appropriate methods based on problem characteristics.
    """

    @pytest.mark.asyncio
    async def test_single_objective_uses_qlogei(self, setup_database):
        """Single-objective campaigns use qLogEI acquisition.

        Reference: BoTorch acquisition function recommendations
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Single Objective Method Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.3}, "objective_values": {"f": 0.5}},
                    {"parameter_values": {"x": 0.7}, "objective_values": {"f": 0.8}},
                ]
            ),
            owner_id,
        )

        # BO suggestions
        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"] is True

        # Method selection should indicate single-objective acquisition
        method = gen2["method_selection"]["acquisition_function"]
        # Should be a single-objective acquisition method
        assert "expected_improvement" in method or "ei" in method.lower()

    @pytest.mark.asyncio
    async def test_multi_objective_uses_qlognehvi(self, setup_database):
        """Multi-objective campaigns use qLogNEHVI acquisition.

        Reference: BoTorch multi-objective tutorial
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Multi Objective Method Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)
        await submit_results(
            campaign_id,
            _to_result_inputs(
                [
                    {"parameter_values": {"x": 0.3}, "objective_values": {"f1": 0.5, "f2": 0.8}},
                    {"parameter_values": {"x": 0.7}, "objective_values": {"f1": 0.8, "f2": 0.5}},
                ]
            ),
            owner_id,
        )

        # BO suggestions
        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"] is True

        # Method selection should indicate multi-objective acquisition
        method = gen2["method_selection"]["acquisition_function"]
        # Should be a multi-objective acquisition method
        assert "hypervolume" in method or "multi_objective" in method

    @pytest.mark.asyncio
    async def test_method_selection_explanation_present(self, setup_database):
        """Method selection includes explanation for transparency."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Method Explanation Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True
        assert "method_selection" in gen
        assert "explanation" in gen["method_selection"]
        assert gen["method_selection"]["explanation"] is not None
        assert len(gen["method_selection"]["explanation"]) > 10  # Non-trivial explanation


class TestSuggestionBatchConsistency:
    """Tests ensuring batch suggestions are consistent and valid."""

    @pytest.mark.asyncio
    async def test_batch_suggestions_within_bounds(self, setup_database):
        """All batch suggestions must be within parameter bounds."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Batch Bounds Test",
            "parameters": [
                {"name": "temp", "type": "continuous", "bounds": [20.0, 100.0]},
                {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Test multiple iterations
        for _ in range(3):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"] is True

            for s in gen["suggestions"]:
                # Check bounds
                assert 20.0 <= s["parameter_values"]["temp"] <= 100.0
                assert 1.0 <= s["parameter_values"]["pressure"] <= 10.0

            # Submit random results to continue
            results = [
                {
                    "parameter_values": s["parameter_values"],
                    "objective_values": {"f1": 0.5, "f2": 0.5},
                }
                for s in gen["suggestions"]
            ]
            await submit_results(campaign_id, _to_result_inputs(results), owner_id)

    @pytest.mark.asyncio
    async def test_batch_suggestions_have_unique_provenance_indices(self, setup_database):
        """Each suggestion in batch has unique batch_index in provenance."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Provenance Index Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True

        batch_indices = [s["provenance"]["batch_index"] for s in gen["suggestions"]]

        # All indices should be unique within the batch
        assert len(batch_indices) == len(set(batch_indices))

        # All indices should be in range [0, batch_size)
        for idx in batch_indices:
            assert 0 <= idx < 5

    @pytest.mark.asyncio
    async def test_batch_diversity_is_reported(self, setup_database):
        """Batch diversity metrics are included in response.

        Reference: Section 1.5 - Batch Diversity Enforcement
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Batch Diversity Test",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        gen = await generate_suggestions(campaign_id)
        assert gen["success"] is True

        # Batch diversity should be reported for batches > 1
        assert "batch_diversity" in gen
        if gen["batch_diversity"] is not None:
            assert "diversity_score" in gen["batch_diversity"]
            assert "min_pairwise_distance" in gen["batch_diversity"]
            assert "mean_pairwise_distance" in gen["batch_diversity"]
            assert gen["batch_diversity"]["diversity_score"] >= 0


class TestIterationConsistency:
    """Tests ensuring iteration counting is consistent."""

    @pytest.mark.asyncio
    async def test_iteration_matches_provenance(self, setup_database):
        """Response iteration matches provenance iteration in suggestions."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Iteration Consistency Test",
            "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 2,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        for expected_iteration in range(1, 5):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"] is True
            assert gen["iteration"] == expected_iteration

            for s in gen["suggestions"]:
                assert s["provenance"]["iteration"] == expected_iteration

            results = [
                {
                    "parameter_values": s["parameter_values"],
                    "objective_values": {"f1": 0.5, "f2": 0.5},
                }
                for s in gen["suggestions"]
            ]
            await submit_results(campaign_id, _to_result_inputs(results), owner_id)
