"""Performance benchmarks for MCP tools.

These tests track suggestion generation time and diagnostics computation
time versus dataset size to ensure performance doesn't regress.

References:
- BoTorch Performance Considerations
  https://botorch.org/docs/optimize_stochastic
- Gaussian Process Scaling: O(n^3) for standard GPs
  Rasmussen & Williams "Gaussian Processes for Machine Learning", 2006
- pytest-benchmark Documentation
  https://pytest-benchmark.readthedocs.io/
"""

import time
from uuid import uuid4

import pytest
import torch

# Performance thresholds (in seconds)
# These are conservative bounds to catch major regressions
# while allowing for variance in CI environments
INITIAL_DESIGN_MAX_TIME = 5.0  # Sobol sequence generation should be fast
BO_SUGGESTION_BASE_TIME = 30.0  # Base time for BO with small dataset
BO_SUGGESTION_PER_POINT = 1.0  # Additional time per observation (rough)
DIAGNOSTICS_BASE_TIME = 10.0  # Base time for diagnostics
DIAGNOSTICS_PER_POINT = 0.5  # Additional time per observation


class TestSuggestionGenerationPerformance:
    """Performance tests for suggestion generation.

    Reference: GP fitting scales as O(n^3), acquisition optimization O(n^2)
    """

    @pytest.mark.asyncio
    async def test_initial_design_performance(self, setup_database):
        """Initial design generation should be fast (< 5 seconds).

        Sobol sequence generation is O(n*d) where n=points, d=dimensions.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        intake_data = {
            "name": "Initial Design Performance Test",
            "parameters": [
                {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(10)
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 10,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Measure initial design generation time
        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True
        assert len(gen["suggestions"]) == 10

        # Initial design should be fast
        assert elapsed_time < INITIAL_DESIGN_MAX_TIME, (
            f"Initial design took {elapsed_time:.2f}s, expected < {INITIAL_DESIGN_MAX_TIME}s"
        )

        # Record performance for tracking
        print(f"\nInitial design (10 params, 10 points): {elapsed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_bo_suggestion_performance_small_dataset(self, setup_database):
        """BO with small dataset (10 observations) should be reasonably fast."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "BO Small Dataset Performance Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)

        # Submit 10 results
        results = [
            {
                "parameter_values": {"x1": i / 10, "x2": (10 - i) / 10},
                "objective_values": {"f1": i / 10, "f2": (10 - i) / 10},
            }
            for i in range(10)
        ]
        await submit_results(campaign_id, results, owner_id)

        # Measure BO suggestion time
        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        max_time = BO_SUGGESTION_BASE_TIME + 10 * BO_SUGGESTION_PER_POINT
        assert elapsed_time < max_time, (
            f"BO with 10 observations took {elapsed_time:.2f}s, expected < {max_time:.2f}s"
        )

        print(f"\nBO suggestions (2 params, 10 obs): {elapsed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_bo_suggestion_performance_medium_dataset(self, setup_database):
        """BO with medium dataset (30 observations) performance tracking."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "BO Medium Dataset Performance Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)

        # Submit 30 results
        results = [
            {
                "parameter_values": {"x1": (i % 10) / 10, "x2": (i // 10) / 10},
                "objective_values": {"f1": (i % 10) / 10, "f2": (i // 10) / 10},
            }
            for i in range(30)
        ]
        await submit_results(campaign_id, results, owner_id)

        # Measure BO suggestion time
        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        max_time = BO_SUGGESTION_BASE_TIME + 30 * BO_SUGGESTION_PER_POINT
        assert elapsed_time < max_time, (
            f"BO with 30 observations took {elapsed_time:.2f}s, expected < {max_time:.2f}s"
        )

        print(f"\nBO suggestions (2 params, 30 obs): {elapsed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_bo_suggestion_scaling_with_observations(self, setup_database):
        """Track how suggestion time scales with number of observations.

        Expected scaling: Roughly O(n^3) for GP fitting, but BoTorch has
        optimizations that may change this in practice.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "BO Scaling Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)

        timing_results = []
        observation_counts = [5, 10, 20]

        cumulative_results = []

        for target_count in observation_counts:
            # Add results to reach target count
            new_results = [
                {
                    "parameter_values": {"x1": (i % 10) / 10, "x2": (i // 10) / 10},
                    "objective_values": {"f1": (i % 10) / 10, "f2": (i // 10) / 10},
                }
                for i in range(len(cumulative_results), target_count)
            ]
            cumulative_results.extend(new_results)

            if new_results:
                await submit_results(campaign_id, new_results, owner_id)

            # Measure suggestion time
            start_time = time.perf_counter()
            gen = await generate_suggestions(campaign_id)
            elapsed_time = time.perf_counter() - start_time

            assert gen["success"] is True
            timing_results.append((target_count, elapsed_time))

        # Report scaling behavior
        print("\nBO Suggestion Time Scaling:")
        print("-" * 40)
        for count, elapsed in timing_results:
            print(f"  {count:3d} observations: {elapsed:.3f}s")

        # Verify time doesn't grow unreasonably
        # Time with 20 obs should be < 10x time with 5 obs (allowing for O(n^3))
        if timing_results[0][1] > 0.1:  # Only check if initial time is measurable
            ratio = timing_results[-1][1] / timing_results[0][1]
            # 20/5 = 4, so O(n^3) would give 64x, but with optimizations expect < 30x
            assert ratio < 50, f"Time scaling ratio {ratio:.1f}x seems too high"


class TestDiagnosticsPerformance:
    """Performance tests for diagnostics computation."""

    @pytest.mark.asyncio
    async def test_diagnostics_performance_small_dataset(self, setup_database):
        """Diagnostics with small dataset should be fast."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Diagnostics Small Dataset Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Generate and submit 10 results
        await generate_suggestions(campaign_id)
        results = [
            {
                "parameter_values": {"x1": i / 10, "x2": (10 - i) / 10},
                "objective_values": {"f1": i / 10, "f2": (10 - i) / 10},
            }
            for i in range(10)
        ]
        await submit_results(campaign_id, results, owner_id)

        # Measure diagnostics time
        start_time = time.perf_counter()
        diag = await get_diagnostics(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert diag["success"] is True

        max_time = DIAGNOSTICS_BASE_TIME + 10 * DIAGNOSTICS_PER_POINT
        assert elapsed_time < max_time, (
            f"Diagnostics with 10 observations took {elapsed_time:.2f}s, expected < {max_time:.2f}s"
        )

        print(f"\nDiagnostics (2 params, 10 obs): {elapsed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_diagnostics_performance_medium_dataset(self, setup_database):
        """Diagnostics with medium dataset performance tracking."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.get_diagnostics import get_diagnostics
        from bo_mcp_server.tools.submit_results import submit_results

        owner_id = str(uuid4())

        intake_data = {
            "name": "Diagnostics Medium Dataset Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Generate and submit 30 results
        await generate_suggestions(campaign_id)
        results = [
            {
                "parameter_values": {"x1": (i % 10) / 10, "x2": (i // 10) / 10},
                "objective_values": {"f1": (i % 10) / 10, "f2": (i // 10) / 10},
            }
            for i in range(30)
        ]
        await submit_results(campaign_id, results, owner_id)

        # Measure diagnostics time
        start_time = time.perf_counter()
        diag = await get_diagnostics(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert diag["success"] is True

        max_time = DIAGNOSTICS_BASE_TIME + 30 * DIAGNOSTICS_PER_POINT
        assert elapsed_time < max_time, (
            f"Diagnostics with 30 observations took {elapsed_time:.2f}s, expected < {max_time:.2f}s"
        )

        print(f"\nDiagnostics (2 params, 30 obs): {elapsed_time:.3f}s")


class TestHighDimensionalPerformance:
    """Performance tests for high-dimensional optimization.

    Reference: TuRBO for high-dimensional optimization
    https://botorch.org/tutorials/turbo_1
    """

    @pytest.mark.asyncio
    async def test_high_dimensional_initial_design(self, setup_database):
        """Initial design with many parameters should still be fast."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())

        # 20 parameters (high-dimensional)
        intake_data = {
            "name": "High-Dim Initial Design Test",
            "parameters": [
                {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(20)
            ],
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 5,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        # Initial design should still be fast for high-dim
        assert elapsed_time < INITIAL_DESIGN_MAX_TIME * 2, (
            f"High-dim initial design took {elapsed_time:.2f}s"
        )

        print(f"\nHigh-dim initial design (20 params, 5 points): {elapsed_time:.3f}s")

    @pytest.mark.asyncio
    async def test_high_dimensional_bo_performance(self, setup_database):
        """BO with many parameters performance tracking.

        High-dimensional BO is expected to be slower but should trigger
        TuRBO for efficiency.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        # 15 parameters
        n_params = 15
        params = [
            {"name": f"x{i}", "type": "continuous", "bounds": [0.0, 1.0]} for i in range(n_params)
        ]
        intake_data = {
            "name": "High-Dim BO Test",
            "parameters": params,
            "objectives": [{"name": "f", "direction": "minimize"}],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        # Initial design
        await generate_suggestions(campaign_id)

        # Submit 15 results (n_params observations)
        results = [
            {
                "parameter_values": {f"x{j}": (i + j) % 10 / 10 for j in range(n_params)},
                "objective_values": {"f": i / 10},
            }
            for i in range(15)
        ]
        await submit_results(campaign_id, results, owner_id)

        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        # High-dim BO is slower, allow 60 seconds
        max_time = 60.0
        assert elapsed_time < max_time, (
            f"High-dim BO took {elapsed_time:.2f}s, expected < {max_time}s"
        )

        print(f"\nHigh-dim BO ({n_params} params, 15 obs): {elapsed_time:.3f}s")


class TestMultiObjectivePerformance:
    """Performance tests for multi-objective optimization.

    Reference: qLogNEHVI complexity grows with number of objectives
    """

    @pytest.mark.asyncio
    async def test_two_objective_performance(self, setup_database):
        """Two-objective optimization performance baseline."""
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "Two-Objective Performance Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)

        results = [
            {
                "parameter_values": {"x1": i / 10, "x2": (10 - i) / 10},
                "objective_values": {"f1": i / 10, "f2": (10 - i) / 10},
            }
            for i in range(15)
        ]
        await submit_results(campaign_id, results, owner_id)

        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        print(f"\nTwo-objective BO (2 params, 15 obs): {elapsed_time:.3f}s")

        return elapsed_time  # For comparison

    @pytest.mark.asyncio
    async def test_three_objective_performance(self, setup_database):
        """Three-objective optimization performance tracking.

        Expected to be slower than two-objective due to Pareto front computation.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "Three-Objective Performance Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
                {"name": "f3", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)

        results = [
            {
                "parameter_values": {"x1": i / 10, "x2": (10 - i) / 10},
                "objective_values": {
                    "f1": i / 10,
                    "f2": (10 - i) / 10,
                    "f3": abs(5 - i) / 10,
                },
            }
            for i in range(15)
        ]
        await submit_results(campaign_id, results, owner_id)

        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True

        # Three objectives expected to be slower, allow 60 seconds
        max_time = 60.0
        assert elapsed_time < max_time, (
            f"Three-objective BO took {elapsed_time:.2f}s, expected < {max_time}s"
        )

        print(f"\nThree-objective BO (2 params, 15 obs): {elapsed_time:.3f}s")


class TestConcurrentOperationsPerformance:
    """Performance tests for concurrent operations.

    These tests ensure the system handles multiple simultaneous operations.
    """

    @pytest.mark.asyncio
    async def test_multiple_campaigns_concurrent_suggestions(self, setup_database):
        """Multiple campaigns can generate suggestions concurrently."""
        import asyncio

        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions

        owner_id = str(uuid4())
        n_campaigns = 3

        # Create multiple campaigns
        campaign_ids = []
        for i in range(n_campaigns):
            intake_data = {
                "name": f"Concurrent Test Campaign {i}",
                "parameters": [{"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}],
                "objectives": [
                    {"name": "f1", "direction": "minimize"},
                    {"name": "f2", "direction": "minimize"},
                ],
                "batch_size": 2,
            }
            create_result = await create_campaign(intake_data, owner_id)
            campaign_ids.append(create_result["campaign_id"])

        # Generate suggestions concurrently
        start_time = time.perf_counter()
        tasks = [generate_suggestions(cid) for cid in campaign_ids]
        results = await asyncio.gather(*tasks)
        elapsed_time = time.perf_counter() - start_time

        # All should succeed
        for result in results:
            assert result["success"] is True

        # Concurrent execution should be faster than sequential
        # (Or at least not much slower than single execution * n)
        print(f"\nConcurrent suggestions ({n_campaigns} campaigns): {elapsed_time:.3f}s")

        # Verify it completed in reasonable time
        assert elapsed_time < INITIAL_DESIGN_MAX_TIME * n_campaigns


class TestPerformanceRegression:
    """Meta-tests to catch performance regressions.

    These tests fail if performance degrades significantly from baseline.
    """

    @pytest.mark.asyncio
    async def test_suggestion_performance_regression(self, setup_database):
        """Catch regressions in suggestion generation performance.

        Baseline: Small dataset BO suggestions should complete in < 30 seconds.
        """
        from bo_mcp_server.tools.create_campaign import create_campaign
        from bo_mcp_server.tools.generate_suggestions import generate_suggestions
        from bo_mcp_server.tools.submit_results import submit_results

        torch.manual_seed(42)
        owner_id = str(uuid4())

        intake_data = {
            "name": "Performance Regression Test",
            "parameters": [
                {"name": "x1", "type": "continuous", "bounds": [0.0, 1.0]},
                {"name": "x2", "type": "continuous", "bounds": [0.0, 1.0]},
            ],
            "objectives": [
                {"name": "f1", "direction": "minimize"},
                {"name": "f2", "direction": "minimize"},
            ],
            "batch_size": 3,
        }

        create_result = await create_campaign(intake_data, owner_id)
        campaign_id = create_result["campaign_id"]

        await generate_suggestions(campaign_id)

        results = [
            {
                "parameter_values": {"x1": i / 10, "x2": (10 - i) / 10},
                "objective_values": {"f1": i / 10, "f2": (10 - i) / 10},
            }
            for i in range(10)
        ]
        await submit_results(campaign_id, results, owner_id)

        # Regression threshold: 30 seconds for small dataset BO
        regression_threshold = 30.0

        start_time = time.perf_counter()
        gen = await generate_suggestions(campaign_id)
        elapsed_time = time.perf_counter() - start_time

        assert gen["success"] is True
        assert elapsed_time < regression_threshold, (
            f"PERFORMANCE REGRESSION: Suggestion generation took {elapsed_time:.2f}s, "
            f"baseline is < {regression_threshold}s"
        )
