#!/usr/bin/env python3
"""Demonstration of Outcome Constraint Modeling.

Outcome constraints allow you to define feasibility thresholds on objectives
that are learned from observed data. Unlike hard constraints that must always
be satisfied, outcome constraints use a GP to predict P(feasible | x).

This script demonstrates:
1. Defining outcome constraints (e.g., yield >= 0.8)
2. How the constraint model learns feasibility regions
3. Optimization with feasibility weighting

Usage:
    uv run python scripts/outcome_constraints_demo.py
"""

import math

import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)


def chemical_yield(temp: float, pressure: float) -> tuple[float, bool]:
    """Simulate chemical reaction yield with feasibility constraint.

    Returns (yield, is_feasible) where:
    - yield: The reaction yield (0-1)
    - is_feasible: True if purity >= 0.9 (implicit constraint)

    Higher yield often comes with lower purity, creating a trade-off.
    """
    # Yield model: peaks around (0.6, 0.7) with some noise
    yield_val = (
        0.95 * math.exp(-((temp - 0.6) ** 2 + (pressure - 0.7) ** 2) / 0.1)
        + 0.2 * math.sin(5 * temp) * math.cos(3 * pressure)
        + 0.3
    )
    yield_val = max(0.0, min(1.0, yield_val))

    # Purity degrades at extreme conditions
    purity = 0.95 - 0.5 * (abs(temp - 0.5) + abs(pressure - 0.5))
    purity = max(0.0, min(1.0, purity))

    # Feasible if purity >= 0.9
    is_feasible = purity >= 0.9

    return yield_val, is_feasible


def demo_outcome_constraint_basics() -> None:
    """Demonstrate outcome constraint specification."""
    print("\n" + "=" * 60)
    print("1. OUTCOME CONSTRAINT BASICS")
    print("=" * 60)
    print()
    print("Outcome constraints define feasibility thresholds on objectives.")
    print("Unlike hard constraints, they are LEARNED from observed data.")
    print()

    # Example spec with outcome constraint
    print("Example specification:")
    print("  OptimizationSpec(")
    print("    parameters=[...],")
    print("    objectives=[ObjectiveSpec(name='yield', minimize=False)],")
    print("    outcome_constraints=[")
    print("      OutcomeConstraintSpec(")
    print("        objective_name='yield',")
    print("        threshold=0.7,")
    print("        greater_than=True,  # yield >= 0.7")
    print("      )")
    print("    ],")
    print("  )")
    print()
    print("Specification:")
    print("  Objective: maximize yield")
    print("  Outcome constraint: yield >= 0.7")
    print()
    print("How it works:")
    print("  1. A GP model learns P(yield >= 0.7 | x) from observations")
    print("  2. Acquisition value is weighted by P(feasible)")
    print("  3. Points likely to be feasible AND high-value are preferred")


def demo_feasibility_learning() -> None:
    """Demonstrate how feasibility is learned from data."""
    print("\n" + "=" * 60)
    print("2. FEASIBILITY LEARNING")
    print("=" * 60)
    print()
    print("Simulating chemical process with yield constraint >= 0.7")
    print()

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="temp", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="pressure", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
        batch_size=3,
        outcome_constraints=[
            OutcomeConstraintSpec(
                objective_name="yield",
                threshold=0.7,
                greater_than=True,
            )
        ],
    )

    # Generate initial observations
    torch.manual_seed(42)
    observations: list[ObservationData] = []

    # Sample some initial points
    initial_points = [
        (0.2, 0.3),
        (0.5, 0.5),
        (0.8, 0.8),
        (0.3, 0.7),
        (0.6, 0.4),
        (0.4, 0.6),
    ]

    print("Initial observations:")
    n_feasible = 0
    for temp, pressure in initial_points:
        yield_val, is_feasible = chemical_yield(temp, pressure)
        observations.append(
            ObservationData(
                parameter_values={"temp": temp, "pressure": pressure},
                objective_values={"yield": yield_val},
            )
        )
        feasible_str = "FEASIBLE" if is_feasible else "infeasible"
        constraint_str = ">= 0.7" if yield_val >= 0.7 else "< 0.7"
        print(
            f"  temp={temp:.2f}, pressure={pressure:.2f} -> "
            f"yield={yield_val:.3f} ({constraint_str}) [{feasible_str}]"
        )
        if yield_val >= 0.7:
            n_feasible += 1

    print(f"\nFeasible points: {n_feasible}/{len(initial_points)}")
    print()

    # Generate constrained suggestions
    print("Generating suggestions with outcome constraint...")
    suggestions, _ = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=3,
        iteration=1,
    )

    print("\nSuggested points (balancing yield and feasibility):")
    for i, sugg in enumerate(suggestions):
        temp = sugg.parameter_values["temp"]
        pressure = sugg.parameter_values["pressure"]
        yield_val, is_feasible = chemical_yield(temp, pressure)
        print(f"  {i + 1}. temp={temp:.3f}, pressure={pressure:.3f}")
        print(f"     (would give yield={yield_val:.3f}, feasible={is_feasible})")


def demo_multiple_constraints() -> None:
    """Demonstrate multiple outcome constraints."""
    print("\n" + "=" * 60)
    print("3. MULTIPLE OUTCOME CONSTRAINTS")
    print("=" * 60)
    print()
    print("You can define multiple outcome constraints on different objectives.")
    print()

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="y", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="profit", minimize=False)],
        batch_size=2,
        outcome_constraints=[
            OutcomeConstraintSpec(
                objective_name="profit",
                threshold=50.0,
                greater_than=True,  # profit >= 50
            ),
        ],
    )

    observations = [
        ObservationData(
            parameter_values={"x": 0.2, "y": 0.3},
            objective_values={"profit": 45.0},  # Infeasible
        ),
        ObservationData(
            parameter_values={"x": 0.5, "y": 0.5},
            objective_values={"profit": 65.0},  # Feasible
        ),
        ObservationData(
            parameter_values={"x": 0.8, "y": 0.2},
            objective_values={"profit": 55.0},  # Feasible
        ),
        ObservationData(
            parameter_values={"x": 0.3, "y": 0.7},
            objective_values={"profit": 40.0},  # Infeasible
        ),
    ]

    print("Observations with profit constraint >= 50:")
    for obs in observations:
        profit = obs.objective_values["profit"]
        status = "FEASIBLE" if profit >= 50 else "infeasible"
        x_val = obs.parameter_values["x"]
        y_val = obs.parameter_values["y"]
        print(f"  x={x_val:.1f}, y={y_val:.1f} -> profit={profit:.0f} [{status}]")

    suggestions, _ = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=2,
        iteration=1,
    )

    print("\nSuggestions considering profit constraint:")
    for i, sugg in enumerate(suggestions):
        print(f"  {i + 1}. x={sugg.parameter_values['x']:.3f}, y={sugg.parameter_values['y']:.3f}")


def demo_constraint_directions() -> None:
    """Demonstrate greater-than vs less-than constraints."""
    print("\n" + "=" * 60)
    print("4. CONSTRAINT DIRECTIONS")
    print("=" * 60)
    print()
    print("Outcome constraints support both directions:")
    print("  - greater_than=True: objective >= threshold")
    print("  - greater_than=False: objective <= threshold")
    print()

    # Example: Minimize cost subject to cost <= 100
    print("Example 1: Minimize cost with constraint cost <= 100")
    print("  OutcomeConstraintSpec(objective_name='cost', threshold=100.0, greater_than=False)")
    print()

    # Example: Maximize yield with yield >= 0.8
    print("Example 2: Maximize yield with constraint yield >= 0.8")
    print("  OutcomeConstraintSpec(objective_name='yield', threshold=0.8, greater_than=True)")


def main() -> None:
    """Run all outcome constraint demonstrations."""
    print("=" * 60)
    print("OUTCOME CONSTRAINT MODELING DEMONSTRATION")
    print("=" * 60)

    demo_outcome_constraint_basics()
    demo_feasibility_learning()
    demo_multiple_constraints()
    demo_constraint_directions()

    print("\n" + "=" * 60)
    print("OUTCOME CONSTRAINT DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
