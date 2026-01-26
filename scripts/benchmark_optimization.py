#!/usr/bin/env python3
"""Benchmark optimization tests using official BoTorch benchmarks.

This script runs optimization workflows on standard benchmark functions
to validate the BO engine implementation.

Usage:
    uv run python scripts/benchmark_optimization.py
    uv run python scripts/benchmark_optimization.py --benchmark branin_currin
    uv run python scripts/benchmark_optimization.py --all

References:
    - https://botorch.org/docs/tutorials/multi_objective_bo/
"""

import argparse
import time
from dataclasses import dataclass

import torch
from bo_engine import (
    compute_hypervolume,
    compute_pareto_front,
    generate_next_batch,
)
from bo_engine.benchmarks import (
    get_benchmark_spec,
    list_benchmarks,
)
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark optimization run."""

    benchmark_name: str
    n_iterations: int
    total_evaluations: int
    best_value: float | None
    final_hypervolume: float | None
    final_pareto_size: int
    elapsed_time: float
    success: bool
    message: str


def create_spec_from_benchmark(
    benchmark_name: str,
    batch_size: int = 3,
) -> tuple[OptimizationSpec, callable]:
    """Create OptimizationSpec from benchmark specification.

    Args:
        benchmark_name: Name of the benchmark
        batch_size: Suggestions per iteration

    Returns:
        Tuple of (OptimizationSpec, objective_function)
    """
    spec_info = get_benchmark_spec(benchmark_name)
    n_dims = spec_info["n_dims"]
    n_objectives = spec_info["n_objectives"]
    bounds = spec_info["bounds"]
    func = spec_info["function"]

    # Create parameter specs
    parameters = []
    for i in range(n_dims):
        parameters.append(
            ParameterSpec(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                bounds=(bounds[0, i].item(), bounds[1, i].item()),
            )
        )

    # Create objective specs (minimize by default)
    objectives = []
    for i in range(n_objectives):
        objectives.append(ObjectiveSpec(name=f"f{i}", minimize=True))

    opt_spec = OptimizationSpec(
        parameters=parameters,
        objectives=objectives,
        batch_size=batch_size,
    )

    return opt_spec, func


def evaluate_batch(
    spec: OptimizationSpec,
    suggestions: list,
    func: callable,
) -> list[ObservationData]:
    """Evaluate a batch of suggestions on the benchmark function.

    Args:
        spec: Optimization specification
        suggestions: List of SuggestionResult objects
        func: Benchmark function

    Returns:
        List of ObservationData with evaluated objectives
    """
    observations = []

    for sugg in suggestions:
        # Convert parameter values to tensor
        x = torch.tensor(
            [sugg.parameter_values[f"x{i}"] for i in range(spec.n_parameters)],
            dtype=torch.double,
        ).unsqueeze(0)

        # Evaluate function
        y = func(x)

        # Handle multi-objective vs single-objective
        if y.dim() == 1:
            obj_values = {f"f{i}": y[0, i].item() for i in range(y.shape[-1])}
        else:
            obj_values = {"f0": y.item()}

        observations.append(
            ObservationData(
                parameter_values=sugg.parameter_values,
                objective_values=obj_values,
            )
        )

    return observations


def run_benchmark(
    benchmark_name: str,
    n_iterations: int = 10,
    batch_size: int = 3,
    verbose: bool = True,
) -> BenchmarkResult:
    """Run optimization on a benchmark function.

    Args:
        benchmark_name: Name of the benchmark
        n_iterations: Number of optimization iterations
        batch_size: Suggestions per iteration
        verbose: Print progress

    Returns:
        BenchmarkResult with optimization outcomes
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {benchmark_name}")
        print("=" * 60)

    start_time = time.time()

    try:
        spec, func = create_spec_from_benchmark(benchmark_name, batch_size)
        spec_info = get_benchmark_spec(benchmark_name)

        if verbose:
            print(f"Parameters: {spec.n_parameters}")
            print(f"Objectives: {spec.n_objectives}")
            print(f"Batch size: {batch_size}")
            print(f"Iterations: {n_iterations}")

        all_observations: list[ObservationData] = []
        hypervolumes: list[float] = []

        for iteration in range(n_iterations):
            if verbose:
                print(f"\n--- Iteration {iteration + 1}/{n_iterations} ---")

            # Generate suggestions
            suggestions = generate_next_batch(
                spec=spec,
                observations=all_observations,
                batch_size=batch_size,
                iteration=iteration,
            )

            if verbose:
                print(f"Generated {len(suggestions)} suggestions")

            # Evaluate suggestions
            new_obs = evaluate_batch(spec, suggestions, func)
            all_observations.extend(new_obs)

            # Compute diagnostics for multi-objective
            if spec.n_objectives > 1:
                # Get all objective values
                y_all = torch.tensor(
                    [
                        [obs.objective_values[f"f{i}"] for i in range(spec.n_objectives)]
                        for obs in all_observations
                    ],
                    dtype=torch.double,
                )

                # Compute Pareto front and hypervolume
                pareto_y, _ = compute_pareto_front(y_all)
                y_max = y_all.max(dim=0).values
                y_range = y_max - y_all.min(dim=0).values
                ref_point = y_max + 0.1 * y_range
                hv = compute_hypervolume(pareto_y, ref_point)
                hypervolumes.append(hv)

                if verbose:
                    print(f"Pareto size: {pareto_y.shape[0]}, Hypervolume: {hv:.4f}")
            else:
                # Single objective: track best value
                best = min(obs.objective_values["f0"] for obs in all_observations)
                if verbose:
                    print(f"Best value: {best:.6f}")

        elapsed = time.time() - start_time

        # Compute final results
        if spec.n_objectives > 1:
            y_all = torch.tensor(
                [
                    [obs.objective_values[f"f{i}"] for i in range(spec.n_objectives)]
                    for obs in all_observations
                ],
                dtype=torch.double,
            )
            pareto_y, _ = compute_pareto_front(y_all)
            final_hv = hypervolumes[-1] if hypervolumes else 0.0
            final_pareto_size = pareto_y.shape[0]
            best_value = None
        else:
            best_value = min(obs.objective_values["f0"] for obs in all_observations)
            final_hv = None
            final_pareto_size = 0

            # Check if close to optimal
            optimal = spec_info.get("optimal_value")
            if optimal is not None:
                gap = abs(best_value - optimal)
                if verbose:
                    print(f"\nOptimality gap: {gap:.6f} (optimal: {optimal})")

        if verbose:
            print(f"\nCompleted in {elapsed:.2f}s")
            print(f"Total evaluations: {len(all_observations)}")

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            n_iterations=n_iterations,
            total_evaluations=len(all_observations),
            best_value=best_value,
            final_hypervolume=final_hv,
            final_pareto_size=final_pareto_size,
            elapsed_time=elapsed,
            success=True,
            message="Completed successfully",
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            n_iterations=n_iterations,
            total_evaluations=0,
            best_value=None,
            final_hypervolume=None,
            final_pareto_size=0,
            elapsed_time=elapsed,
            success=False,
            message=str(e),
        )


def run_all_benchmarks(
    n_iterations: int = 5,
    batch_size: int = 3,
) -> list[BenchmarkResult]:
    """Run optimization on all available benchmarks.

    Args:
        n_iterations: Iterations per benchmark
        batch_size: Suggestions per iteration

    Returns:
        List of BenchmarkResult for each benchmark
    """
    results = []

    # Select benchmarks that are reasonable to run quickly
    quick_benchmarks = [
        "branin",
        "hartmann6",
        "branin_currin",
        "dtlz2",
    ]

    for benchmark in quick_benchmarks:
        result = run_benchmark(
            benchmark_name=benchmark,
            n_iterations=n_iterations,
            batch_size=batch_size,
            verbose=True,
        )
        results.append(result)

    return results


def print_summary(results: list[BenchmarkResult]):
    """Print summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(f"\n{'Benchmark':<20} {'Status':<10} {'Evals':<8} {'Best/HV':<12} {'Time':<8}")
    print("-" * 60)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        if r.best_value is not None:
            metric = f"{r.best_value:.4f}"
        elif r.final_hypervolume is not None:
            metric = f"HV={r.final_hypervolume:.4f}"
        else:
            metric = "N/A"
        row = f"{r.benchmark_name:<20} {status:<10} {r.total_evaluations:<8} "
        print(f"{row}{metric:<12} {r.elapsed_time:.2f}s")

    n_pass = sum(1 for r in results if r.success)
    print(f"\n{n_pass}/{len(results)} benchmarks passed")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run BO benchmarks from BoTorch tutorials")
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Specific benchmark to run (default: branin_currin)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size (default: 3)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available benchmarks",
    )

    args = parser.parse_args()

    if args.list:
        print("Available benchmarks:")
        for name in list_benchmarks():
            spec = get_benchmark_spec(name)
            print(f"  {name}: {spec['description']}")
        return

    torch.manual_seed(42)

    if args.all:
        results = run_all_benchmarks(
            n_iterations=args.iterations,
            batch_size=args.batch_size,
        )
        print_summary(results)
    else:
        benchmark = args.benchmark or "branin_currin"
        result = run_benchmark(
            benchmark_name=benchmark,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            verbose=True,
        )
        print(f"\nResult: {'PASS' if result.success else 'FAIL'}")
        if not result.success:
            print(f"Error: {result.message}")


if __name__ == "__main__":
    main()
