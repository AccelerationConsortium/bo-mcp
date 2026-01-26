#!/usr/bin/env python3
"""Demonstration of single-objective Bayesian Optimization (v1.0.1).

This script demonstrates the new single-objective optimization capabilities:
- qLogNEI acquisition function for noisy observations
- SingleTaskGP model instead of ModelListGP
- Best value tracking and improvement history

Usage:
    uv run python scripts/single_objective_demo.py
"""

import math
import sys
from pathlib import Path

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "bo-engine" / "src"))

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    compute_best_value,
    compute_improvement_history,
    compute_single_objective_improvement_rate,
    determine_single_objective_health_status,
    generate_next_batch,
)


def branin(x1: float, x2: float) -> float:
    """Branin test function (single-objective).

    Global minima: f(x*) ≈ 0.397887
    at x* = (-π, 12.275), (π, 2.275), (9.42478, 2.475)

    Modified domain: [0, 1] × [0, 1] mapped to [-5, 10] × [0, 15]
    """
    # Map [0, 1] to original domain
    x1_orig = x1 * 15 - 5  # [-5, 10]
    x2_orig = x2 * 15  # [0, 15]

    a = 1
    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6
    s = 10
    t = 1 / (8 * math.pi)

    return (
        a * (x2_orig - b * x1_orig**2 + c * x1_orig - r) ** 2 + s * (1 - t) * math.cos(x1_orig) + s
    )


def run_single_objective_optimization() -> None:
    """Run a single-objective Bayesian Optimization demo."""
    print("=" * 60)
    print("Single-Objective Bayesian Optimization Demo (v1.0.1)")
    print("=" * 60)
    print()

    # Define optimization problem
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="branin", minimize=True),
        ],
        batch_size=3,
        acquisition_method=AcquisitionMethod.AUTO,  # Auto-selects qLogNEI for single-objective
    )

    print("Problem: Branin function (minimize)")
    print("Parameters: x1 ∈ [0, 1], x2 ∈ [0, 1]")
    print(f"Batch size: {spec.batch_size}")
    print(f"Acquisition: {spec.acquisition_method.value} (auto-selects qLogNEI)")
    print()

    observations: list[ObservationData] = []
    n_iterations = 5

    for iteration in range(n_iterations):
        print(f"\n{'=' * 40}")
        print(f"Iteration {iteration + 1}/{n_iterations}")
        print(f"{'=' * 40}")

        # Generate suggestions
        suggestions = generate_next_batch(spec, observations, iteration=iteration)

        print(f"\nGenerated {len(suggestions)} suggestions:")
        print(f"  Method: {suggestions[0].generation_method}")
        if suggestions[0].acquisition_function:
            print(f"  Acquisition: {suggestions[0].acquisition_function}")

        # Evaluate suggestions
        for i, sugg in enumerate(suggestions):
            x1 = sugg.parameter_values["x1"]
            x2 = sugg.parameter_values["x2"]
            f_val = branin(x1, x2)

            print(f"\n  Suggestion {i + 1}:")
            print(f"    x1={x1:.4f}, x2={x2:.4f}")
            print(f"    f(x)={f_val:.4f}")
            if sugg.confidence_level:
                print(f"    Confidence: {sugg.confidence_level}")
            if sugg.explanation:
                print(f"    Reason: {sugg.explanation[:80]}...")

            observations.append(
                ObservationData(
                    parameter_values={"x1": x1, "x2": x2},
                    objective_values={"branin": f_val},
                )
            )

        # Compute diagnostics
        objective_values = [obs.objective_values["branin"] for obs in observations]
        best_val, best_idx = compute_best_value(objective_values, minimize=True)
        improvement_history = compute_improvement_history(objective_values, minimize=True)
        improvement_rate = compute_single_objective_improvement_rate(improvement_history)

        print("\n  Diagnostics:")
        print(f"    Total evaluations: {len(observations)}")
        print(f"    Best value: {best_val:.4f}")
        print(f"    Improvement rate: {improvement_rate:.4f}")
        print(
            f"    Best parameters: x1={observations[best_idx].parameter_values['x1']:.4f}, "
            f"x2={observations[best_idx].parameter_values['x2']:.4f}"
        )

    # Final summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)

    objective_values = [obs.objective_values["branin"] for obs in observations]
    best_val, best_idx = compute_best_value(objective_values, minimize=True)
    improvement_history = compute_improvement_history(objective_values, minimize=True)

    print("\nFinal Results:")
    print(f"  Total evaluations: {len(observations)}")
    print(f"  Best value found: {best_val:.4f}")
    print("  Best parameters:")
    print(f"    x1 = {observations[best_idx].parameter_values['x1']:.4f}")
    print(f"    x2 = {observations[best_idx].parameter_values['x2']:.4f}")
    print("\nGlobal optimum: f* ≈ 0.3979")
    print(f"Gap to optimum: {abs(best_val - 0.3979):.4f}")

    # Health status
    status, warnings = determine_single_objective_health_status(
        improvement_history=improvement_history,
        model_correlation=0.8,  # Assume good correlation for demo
    )
    print(f"\nOptimization health: {status}")
    if warnings:
        for w in warnings:
            print(f"  Warning: {w}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_single_objective_optimization()
