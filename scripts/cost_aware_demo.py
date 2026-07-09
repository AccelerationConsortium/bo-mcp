#!/usr/bin/env python3
"""Demonstration of Cost-Aware Bayesian Optimization (EIpu).

Cost-aware BO optimizes Expected Improvement per Unit cost (EIpu),
balancing objective improvement with evaluation cost.

This script demonstrates:
1. EIpu acquisition function
2. How cost information is used
3. When to use cost-aware optimization

Usage:
    uv run python scripts/cost_aware_demo.py
"""

import math

import torch

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)


def expensive_experiment(x1: float, x2: float) -> tuple[float, float]:
    """Simulate an experiment with varying cost.

    Returns (objective_value, cost) where:
    - objective_value: The function value to minimize
    - cost: Time/money/resources needed for this configuration

    Cost increases with x2 (e.g., longer processing time).
    """
    # Objective: Branin-like function
    objective = (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
    ) / 100  # Normalize to ~[0, 1]

    # Cost model: increases with x2 (expensive at high x2)
    base_cost = 10.0
    cost = base_cost * (1 + 5 * x2**2)

    return objective, cost


def demo_eipu_basics() -> None:
    """Demonstrate EIpu acquisition function basics."""
    print("\n" + "=" * 60)
    print("1. EIPU (Expected Improvement per Unit) BASICS")
    print("=" * 60)
    print()
    print("Standard BO maximizes Expected Improvement (EI):")
    print("  EI(x) = E[max(0, f* - f(x))]")
    print()
    print("Cost-aware BO maximizes Expected Improvement per Unit cost:")
    print("  EIpu(x) = EI(x) / E[cost(x)]")
    print()
    print("This balances improvement with evaluation cost, preferring")
    print("cheap experiments that still provide useful information.")
    print()

    # Show acquisition method
    print(f"Acquisition method: {AcquisitionMethod.EIPU.value}")  # ty: ignore[unresolved-attribute]
    print()


def demo_cost_tradeoff() -> None:
    """Demonstrate the cost vs. improvement trade-off."""
    print("\n" + "=" * 60)
    print("2. COST VS IMPROVEMENT TRADE-OFF")
    print("=" * 60)
    print()
    print("Simulating experiments with varying cost...")
    print("Cost model: cost = 10 * (1 + 5*x2^2)")
    print("  - Low x2: cheap experiments")
    print("  - High x2: expensive experiments")
    print()

    # Generate sample experiments
    experiments = [
        (0.2, 0.1),  # Low cost
        (0.5, 0.5),  # Medium cost
        (0.8, 0.9),  # High cost
        (0.3, 0.2),  # Low cost
        (0.7, 0.8),  # High cost
    ]

    print("Sample experiments:")
    for x1, x2 in experiments:
        obj, cost = expensive_experiment(x1, x2)
        print(f"  x1={x1:.1f}, x2={x2:.1f} -> objective={obj:.4f}, cost={cost:.1f}")
    print()


def demo_cost_aware_optimization() -> None:
    """Demonstrate cost-aware optimization loop."""
    print("\n" + "=" * 60)
    print("3. COST-AWARE OPTIMIZATION LOOP")
    print("=" * 60)
    print()

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=True,
        acquisition_method=AcquisitionMethod.EIPU,  # ty: ignore[unresolved-attribute]
    )

    # Initial observations with cost data
    observations: list[ObservationData] = []
    initial_points = [
        (0.1, 0.1),
        (0.5, 0.3),
        (0.9, 0.2),
        (0.3, 0.7),
        (0.7, 0.5),
    ]

    print("Initial observations (with cost):")
    total_cost = 0.0
    best_value = float("inf")

    for x1, x2 in initial_points:
        obj, cost = expensive_experiment(x1, x2)
        observations.append(
            ObservationData(
                parameter_values={"x1": x1, "x2": x2},
                objective_values={"f": obj},
                cost=cost,
            )
        )
        total_cost += cost
        best_value = min(best_value, obj)
        print(f"  x1={x1:.2f}, x2={x2:.2f} -> f={obj:.4f}, cost={cost:.1f}")

    print(f"\nTotal cost so far: {total_cost:.1f}")
    print(f"Best value so far: {best_value:.4f}")
    print()

    # Generate cost-aware suggestions
    print("Generating cost-aware suggestions...")
    suggestions, _ = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=2,
        iteration=1,
    )

    print("\nSuggested experiments (optimizing EIpu):")
    for i, sugg in enumerate(suggestions):
        x1 = sugg.parameter_values["x1"]
        x2 = sugg.parameter_values["x2"]
        obj, cost = expensive_experiment(x1, x2)
        print(f"  {i + 1}. x1={x1:.3f}, x2={x2:.3f}")
        print(f"     Acquisition: {sugg.acquisition_function}")
        print(f"     (would give f={obj:.4f}, cost={cost:.1f})")


def demo_comparison() -> None:
    """Compare cost-aware vs standard BO."""
    print("\n" + "=" * 60)
    print("4. COST-AWARE vs STANDARD BO")
    print("=" * 60)
    print()

    torch.manual_seed(42)

    # Same initial observations
    initial_obs = [
        ObservationData(
            parameter_values={"x1": 0.2, "x2": 0.2},
            objective_values={"f": 0.3},
            cost=14.0,
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"f": 0.2},
            cost=22.5,
        ),
        ObservationData(
            parameter_values={"x1": 0.8, "x2": 0.8},
            objective_values={"f": 0.4},
            cost=42.0,
        ),
    ]

    # Standard BO
    spec_standard = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=False,
    )

    # Cost-aware BO
    spec_cost_aware = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=True,
        acquisition_method=AcquisitionMethod.EIPU,  # ty: ignore[unresolved-attribute]
    )

    print("Standard BO (qLogNEI):")
    sugg_standard, _ = generate_next_batch(
        spec=spec_standard,
        observations=initial_obs,
        batch_size=2,
        iteration=1,
    )
    for sugg in sugg_standard:
        x2 = sugg.parameter_values["x2"]
        _, cost = expensive_experiment(sugg.parameter_values["x1"], x2)
        print(f"  x2={x2:.3f}, expected_cost={cost:.1f}")

    print("\nCost-aware BO (EIpu):")
    sugg_cost_aware, _ = generate_next_batch(
        spec=spec_cost_aware,
        observations=initial_obs,
        batch_size=2,
        iteration=1,
    )
    for sugg in sugg_cost_aware:
        x2 = sugg.parameter_values["x2"]
        _, cost = expensive_experiment(sugg.parameter_values["x1"], x2)
        print(f"  x2={x2:.3f}, expected_cost={cost:.1f}")

    print()
    print("Observation: Cost-aware BO tends to suggest cheaper experiments")
    print("(lower x2) while still seeking improvement.")


def demo_use_cases() -> None:
    """Describe when to use cost-aware BO."""
    print("\n" + "=" * 60)
    print("5. WHEN TO USE COST-AWARE BO")
    print("=" * 60)
    print()
    print("Use EIpu when evaluation cost varies significantly:")
    print()
    print("  GOOD use cases:")
    print("    - Lab experiments with different durations")
    print("    - Simulations with varying computational cost")
    print("    - A/B tests with different sample sizes")
    print("    - Manufacturing with different material costs")
    print()
    print("  NOT recommended when:")
    print("    - All evaluations have similar cost")
    print("    - Finding the optimum is more important than efficiency")
    print("    - Cost is unknown until after evaluation")
    print()
    print("  Note: Cost must be provided with each observation.")
    print("  If cost data is missing, standard BO is used instead.")


def main() -> None:
    """Run all cost-aware BO demonstrations."""
    print("=" * 60)
    print("COST-AWARE BAYESIAN OPTIMIZATION (EIpu) DEMONSTRATION")
    print("=" * 60)

    demo_eipu_basics()
    demo_cost_tradeoff()
    demo_cost_aware_optimization()
    demo_comparison()
    demo_use_cases()

    print("\n" + "=" * 60)
    print("COST-AWARE BO DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
