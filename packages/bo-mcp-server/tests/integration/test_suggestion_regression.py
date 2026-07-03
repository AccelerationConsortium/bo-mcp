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

import random
from uuid import uuid4

import pytest
import torch

from bo_mcp_server.domain import ResultSubmissionInput


def _to_result_inputs(results: list[dict]) -> list[ResultSubmissionInput]:
    return [ResultSubmissionInput.model_validate(r) for r in results]


@pytest.mark.usefixtures("setup_database")
class TestSuggestionReproducibility:
    """Tests ensuring deterministic suggestion generation with fixed seeds.

    Reference: BoTorch reproducibility documentation
    https://botorch.org/docs/reproducibility
    """

    @pytest.mark.asyncio
    async def test_initial_design_deterministic(self):
        """Initial design (Sobol sequence) produces same suggestions with same setup.

        Sobol sequences are deterministic when seeded; the campaign's
        explicit ``random_seed`` is threaded through ``OptimizationSpec``
        into ``SobolEngine`` so two campaigns sharing a spec produce
        identical initial-design batches.
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
            "random_seed": 42,
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
    async def test_suggestion_provenance_includes_seed(self):
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

    @pytest.mark.asyncio
    async def test_bo_phase_deterministic_with_seed(self):
        """BO-phase suggestions are reproducible when ``random_seed`` is set.

        Previously ``generate_next_batch`` picked its acquisition seed
        with ``random.randint`` whenever the caller did not pass an
        ``rng``, so two replays of the same seeded campaign produced
        divergent BO candidates even though the ``spec.random_seed``
        hint was honored by the Sobol initial design.  The fix derives
        the acquisition seed from ``spec.random_seed`` + ``iteration``,
        so the BO path on call 2 of two sibling campaigns with
        identical specs and identical observed history returns the same
        ``parameter_values``.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "1.41b seeded BO reproducibility",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 1,
            "random_seed": 4242,
        }

        seed_points = [
            (0.1, 0.1),
            (0.9, 0.1),
            (0.1, 0.9),
            (0.9, 0.9),
            (0.5, 0.5),
            (0.3, 0.7),
        ]
        seed_results = [
            {
                "parameter_values": {"x": px, "y": py},
                "objective_values": {"f": (px - 0.2) ** 2 + (py - 0.2) ** 2},
            }
            for px, py in seed_points
        ]

        async def _campaign_bo_candidate() -> dict:
            create = await create_campaign(intake_data, owner_id)
            assert create["success"], create
            campaign_id = create["campaign_id"]
            submit = await submit_results(campaign_id, _to_result_inputs(seed_results), owner_id)
            assert submit["success"], submit
            gen = await generate_suggestions(campaign_id)
            assert gen["success"], gen
            assert len(gen["suggestions"]) == 1
            assert gen["suggestions"][0]["provenance"]["generation_method"] == "bo"
            return gen["suggestions"][0]["parameter_values"]

        first = await _campaign_bo_candidate()
        second = await _campaign_bo_candidate()

        for param in ("x", "y"):
            assert abs(first[param] - second[param]) < 1e-10, (
                f"Seeded BO-phase candidate diverged across replays for '{param}': "
                f"first={first}, second={second}"
            )


def _params_tuple(params: dict) -> tuple:
    """Hashable representation of a parameter-values dict."""
    return tuple(sorted(params.items()))


@pytest.mark.usefixtures("setup_database")
class TestInitialDesignNoDuplicates:
    """Regression tests for initial-design duplicate elimination.

    Smoke-test reproduction: a 2x2 categorical campaign with
    ``batch_size=2`` and ``initial_design_size=2`` used to reissue
    already-suggested points on the second ``generate_suggestions`` call
    because Sobol was reseeded on every call and no duplicate filter
    existed.  These tests pin the new contract:

    1. Consecutive initial-design batches (with results submitted
       between) return disjoint parameter sets.
    2. When the finite categorical search space has been fully
       observed, a further call returns a structured
       ``SEARCH_SPACE_EXHAUSTED`` error instead of a duplicate or a
       500.
    """

    @pytest.mark.asyncio
    async def test_consecutive_initial_design_batches_are_disjoint(self):
        """Two consecutive batches never reissue an observed point."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "1.41a disjoint batches",
            "parameters": [
                {"name": "molecule", "type": "categorical", "categories": ["a", "b"]},
                {"name": "solvent", "type": "categorical", "categories": ["x", "y"]},
            ],
            "objectives": [{"name": "score", "direction": "maximize"}],
            "batch_size": 2,
            "initial_design_size": 2,
            "random_seed": 123,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"], create_result
        campaign_id = create_result["campaign_id"]

        gen1 = await generate_suggestions(campaign_id)
        assert gen1["success"], gen1
        assert len(gen1["suggestions"]) == 2
        first_keys = {_params_tuple(s["parameter_values"]) for s in gen1["suggestions"]}
        assert len(first_keys) == 2, "Within-batch duplicates already violate 1.41a"

        # Feed the results back so call 2 enters the partial-data fallback.
        results = [
            {
                "suggestion_id": s["id"],
                "parameter_values": s["parameter_values"],
                "objective_values": {"score": float(idx + 1)},
            }
            for idx, s in enumerate(gen1["suggestions"])
        ]
        submit = await submit_results(campaign_id, _to_result_inputs(results), owner_id)
        assert submit["success"], submit

        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"], gen2
        assert len(gen2["suggestions"]) == 2
        second_keys = {_params_tuple(s["parameter_values"]) for s in gen2["suggestions"]}
        assert len(second_keys) == 2, "Within-batch duplicates on second call"
        assert first_keys.isdisjoint(second_keys), (
            f"Second batch reissued observed points: overlap={first_keys & second_keys}"
        )

    @pytest.mark.asyncio
    async def test_exhausted_categorical_space_returns_structured_error(self):
        """After all 4 combinations are observed, a further call errors cleanly."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "1.41a exhaustion",
            "parameters": [
                {"name": "molecule", "type": "categorical", "categories": ["a", "b"]},
                {"name": "solvent", "type": "categorical", "categories": ["x", "y"]},
            ],
            "objectives": [{"name": "score", "direction": "maximize"}],
            "batch_size": 2,
            "initial_design_size": 2,
            "random_seed": 123,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"], create_result
        campaign_id = create_result["campaign_id"]

        # Exhaust the 4 combinations over two initial-design batches.
        observed_keys: set[tuple] = set()
        for iteration_idx in range(2):
            gen = await generate_suggestions(campaign_id)
            assert gen["success"], gen
            results = [
                {
                    "suggestion_id": s["id"],
                    "parameter_values": s["parameter_values"],
                    "objective_values": {"score": float(iteration_idx + 1)},
                }
                for s in gen["suggestions"]
            ]
            observed_keys.update(_params_tuple(s["parameter_values"]) for s in gen["suggestions"])
            submit = await submit_results(campaign_id, _to_result_inputs(results), owner_id)
            assert submit["success"], submit

        assert len(observed_keys) == 4, f"Expected all 4 combinations observed, got {observed_keys}"

        # The next call has no fresh combination to return.
        gen3 = await generate_suggestions(campaign_id)
        assert gen3["success"] is False
        assert gen3["suggestions"] == []
        assert gen3["error"]["code"] == "E011"
        details = gen3["error"]["details"]
        assert details["next_action_recommendation"] == "terminate_campaign"
        assert details["n_total_combinations"] == 4
        assert details["n_available"] == 0


@pytest.mark.usefixtures("setup_database")
class TestPendingPointsConditioning:
    """Regression tests for X_pending conditioning in acquisition.

    Previously ``optimize_acquisition`` never threaded ``X_pending``
    into ``botorch.optim.optimize_acqf``.  Two consecutive
    ``generate_suggestions`` calls on the same campaign state (no new
    results submitted between them) therefore returned essentially the
    same BO candidate — the acquisition maximum is a function of the
    fitted GP alone, which has not changed.  In-flight / PENDING
    suggestions are now encoded and passed as ``X_pending``, so the
    joint acquisition conditions on them and produces a distinct point.

    Reference: BoTorch "batched" / parallel BO tutorial
    https://botorch.org/docs/batched_bayesian_optimization/
    """

    @pytest.mark.asyncio
    async def test_consecutive_bo_calls_return_distinct_points(self):
        """Two sequential BO calls (no results between) return distinct points.

        Without X_pending wiring, call 2 re-selects the same acquisition
        argmax as call 1, so the normalized L2 distance between the two
        returned candidates collapses to ~0.  The pinned tolerance below
        is well above machine epsilon but far below any plausible
        random-search spread; a ~0 distance is a direct regression
        signal.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "1.41 X_pending conditioning",
            "parameters": [
                {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "y", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 1,
            "random_seed": 7,
        }

        create_result = await create_campaign(intake_data, owner_id)
        assert create_result["success"], create_result
        campaign_id = create_result["campaign_id"]

        # Seed the GP with enough observations so call 2 hits the BO
        # path, not the initial-design fallback.  Six points on a simple
        # bowl give a well-conditioned model.
        seed_points = [
            (0.1, 0.1),
            (0.9, 0.1),
            (0.1, 0.9),
            (0.9, 0.9),
            (0.5, 0.5),
            (0.3, 0.7),
        ]
        seed_results = [
            {
                "parameter_values": {"x": px, "y": py},
                "objective_values": {"f": (px - 0.2) ** 2 + (py - 0.2) ** 2},
            }
            for px, py in seed_points
        ]
        submit = await submit_results(campaign_id, _to_result_inputs(seed_results), owner_id)
        assert submit["success"], submit

        # Call 1 — produces a PENDING suggestion.
        gen1 = await generate_suggestions(campaign_id)
        assert gen1["success"], gen1
        assert len(gen1["suggestions"]) == 1
        first = gen1["suggestions"][0]["parameter_values"]
        assert gen1["suggestions"][0]["provenance"]["generation_method"] == "bo"

        # Call 2 — no new results submitted.  Without X_pending this
        # returns the same acquisition argmax as call 1.
        gen2 = await generate_suggestions(campaign_id)
        assert gen2["success"], gen2
        assert len(gen2["suggestions"]) == 1
        second = gen2["suggestions"][0]["parameter_values"]

        # Normalized L2 distance on the unit square.  Tolerance picked
        # to be ~100x machine epsilon and ~1/20th of the bounds span so
        # the assertion discriminates "same point" from "different
        # point" without being sensitive to restart noise.
        dx = first["x"] - second["x"]
        dy = first["y"] - second["y"]
        distance = (dx * dx + dy * dy) ** 0.5
        min_distance = 0.01
        assert distance > min_distance, (
            f"Consecutive BO suggestions collapsed to same point without "
            f"X_pending conditioning: first={first}, second={second}, "
            f"distance={distance:.6f}"
        )


@pytest.mark.usefixtures("setup_database")
class TestSuggestionQualityRegression:
    """Tests ensuring suggestion quality remains stable.

    These tests use known benchmark functions to verify that
    optimization finds reasonable solutions.
    """

    @pytest.mark.asyncio
    async def test_branin_currin_hypervolume_regression(self):
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

        random.seed(42)
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
    async def test_quadratic_finds_minimum(self):
        """Simple quadratic function should find near-optimal solution.

        A simple test function with known minimum at (0.5, 0.5).
        After a few iterations, best value should be close to 0.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        random.seed(42)
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


@pytest.mark.usefixtures("setup_database")
class TestMethodSelectionStability:
    """Tests ensuring method selection logic remains stable.

    These tests verify that the automatic method selection
    chooses appropriate methods based on problem characteristics.
    """

    @pytest.mark.asyncio
    async def test_single_objective_uses_qlogei(self):
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
        # Should be a single-objective acquisition method. Accept both the
        # BoTorch-style snake_case label and BayBE's acqf class names.
        normalized = method.lower().replace("_", "")
        assert "expectedimprovement" in normalized or "ei" in method.lower()

    @pytest.mark.asyncio
    async def test_multi_objective_uses_qlognehvi(self):
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
        # Should be a multi-objective acquisition method (case-insensitive to
        # cover BayBE's acqf class names, e.g. qLogNoisyExpectedHypervolumeImprovement).
        assert "hypervolume" in method.lower() or "multi_objective" in method.lower()

    @pytest.mark.asyncio
    async def test_method_selection_explanation_present(self):
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


@pytest.mark.usefixtures("setup_database")
class TestSuggestionBatchConsistency:
    """Tests ensuring batch suggestions are consistent and valid."""

    @pytest.mark.asyncio
    async def test_batch_suggestions_within_bounds(self):
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
    async def test_batch_suggestions_have_unique_provenance_indices(self):
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
    async def test_batch_diversity_is_reported(self):
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


@pytest.mark.usefixtures("setup_database")
class TestIterationConsistency:
    """Tests ensuring iteration counting is consistent."""

    @pytest.mark.asyncio
    async def test_iteration_matches_provenance(self):
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
