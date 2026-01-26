#!/usr/bin/env python3
"""Comprehensive demonstration of v1.2 and v1.3 features.

This script demonstrates all new features:
- v1.2: TuRBO (Trust Region BO) for high-dimensional optimization
- v1.3: Outcome Constraint Modeling (learned feasibility)
- v1.3: Cost-Aware BO (EIpu - Expected Improvement per Unit cost)

Usage:
    uv run python scripts/v12_v13_features_demo.py
"""

import sys
from pathlib import Path

import torch

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "bo-engine" / "src"))

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    OutcomeConstraintSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
    select_methods,
    should_use_turbo,
)


def print_header(title: str) -> None:
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def demo_version_overview() -> None:
    """Show overview of new features."""
    print_header("BO-ENGINE v1.2 + v1.3 FEATURES OVERVIEW")
    print()
    print("v1.2 Features:")
    print("  - Method Selection API (select_methods())")
    print("  - TuRBO for high-dimensional optimization (20+ parameters)")
    print("  - Trust region dynamics (expand on success, contract on failure)")
    print()
    print("v1.3 Features:")
    print("  - Outcome Constraint Modeling (learned feasibility)")
    print("  - Cost-Aware BO (EIpu acquisition function)")
    print("  - ObservationData.cost field for evaluation costs")
    print()


def demo_turbo_summary() -> None:
    """Quick summary of TuRBO capabilities."""
    print_header("v1.2: TuRBO (Trust Region BO)")

    # Create high-dim spec
    n_params = 30
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(n_params)
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=4,
        use_turbo=True,
    )

    print(f"\nProblem: {n_params}-dimensional single-objective optimization")
    print(f"TuRBO recommended: {should_use_turbo(n_params)}")

    # Method selection shows TuRBO strategy
    methods = select_methods(spec, n_observations=20)
    print("\nMethod Selection:")
    print(f"  Model: {methods.model_type}")
    print(f"  Acquisition: {methods.acquisition_function}")
    print(f"  Strategy: {methods.optimization_strategy}")

    # Generate with TuRBO
    torch.manual_seed(42)
    observations = [
        ObservationData(
            parameter_values={f"x{i}": torch.rand(1).item() for i in range(n_params)},
            objective_values={"f": torch.rand(1).item()},
        )
        for _ in range(10)
    ]

    suggestions, turbo_state = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=4,
        iteration=1,
    )

    print(f"\nGenerated {len(suggestions)} suggestions using TuRBO")
    if turbo_state:
        print(f"  Trust region length: {turbo_state.length:.4f}")
        print(f"  Best value: {turbo_state.best_value:.4f}")


def demo_outcome_constraints_summary() -> None:
    """Quick summary of outcome constraints."""
    print_header("v1.3: Outcome Constraint Modeling")

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="temp", type=ParameterType.CONTINUOUS, bounds=(0.0, 100.0)),
            ParameterSpec(name="pressure", type=ParameterType.CONTINUOUS, bounds=(0.0, 10.0)),
        ],
        objectives=[ObjectiveSpec(name="yield", minimize=False)],
        batch_size=3,
        outcome_constraints=[
            OutcomeConstraintSpec(
                objective_name="yield",
                threshold=0.75,
                greater_than=True,  # yield >= 0.75
            )
        ],
    )

    print("\nProblem: Maximize yield with constraint yield >= 0.75")
    print("The constraint model learns P(yield >= 0.75 | x) from data.")

    # Observations with mixed feasibility
    observations = [
        ObservationData(
            parameter_values={"temp": 50.0, "pressure": 5.0},
            objective_values={"yield": 0.85},  # Feasible
        ),
        ObservationData(
            parameter_values={"temp": 30.0, "pressure": 3.0},
            objective_values={"yield": 0.65},  # Infeasible
        ),
        ObservationData(
            parameter_values={"temp": 70.0, "pressure": 7.0},
            objective_values={"yield": 0.78},  # Feasible
        ),
        ObservationData(
            parameter_values={"temp": 20.0, "pressure": 8.0},
            objective_values={"yield": 0.55},  # Infeasible
        ),
    ]

    n_feasible = sum(1 for o in observations if o.objective_values["yield"] >= 0.75)
    print(f"\nObservations: {len(observations)} total, {n_feasible} feasible")

    suggestions, _ = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=3,
        iteration=1,
    )

    print(f"\nGenerated {len(suggestions)} suggestions considering feasibility")
    print("Suggestions balance yield improvement with P(feasible).")


def demo_cost_aware_summary() -> None:
    """Quick summary of cost-aware BO."""
    print_header("v1.3: Cost-Aware BO (EIpu)")

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=2,
        use_cost_aware=True,
        acquisition_method=AcquisitionMethod.EIPU,
    )

    print("\nProblem: Minimize f(x) while considering evaluation cost")
    print("EIpu = EI(x) / E[cost(x)]")
    print()
    print("Observations with varying cost:")

    observations = [
        ObservationData(
            parameter_values={"x1": 0.1, "x2": 0.1},
            objective_values={"f": 0.5},
            cost=10.0,  # Cheap
        ),
        ObservationData(
            parameter_values={"x1": 0.5, "x2": 0.5},
            objective_values={"f": 0.3},
            cost=50.0,  # Medium
        ),
        ObservationData(
            parameter_values={"x1": 0.9, "x2": 0.9},
            objective_values={"f": 0.2},
            cost=100.0,  # Expensive
        ),
    ]

    for obs in observations:
        print(f"  x={obs.parameter_values}, f={obs.objective_values['f']:.2f}, cost={obs.cost:.0f}")

    suggestions, _ = generate_next_batch(
        spec=spec,
        observations=observations,
        batch_size=2,
        iteration=1,
    )

    print(f"\nGenerated {len(suggestions)} suggestions using EIpu")
    print("EIpu prefers cheaper experiments with good expected improvement.")


def demo_combined_features() -> None:
    """Show how features can be combined."""
    print_header("COMBINING FEATURES")

    print("\nYou can combine v1.3 features:")
    print()

    print("1. Cost-aware + Outcome Constraints:")
    print("   - Optimize EI per unit cost")
    print("   - Subject to P(quality >= 0.8) being high")
    print()

    # Note about TuRBO
    print("2. TuRBO + Outcome Constraints:")
    print("   - TuRBO for high-dim (20+ params)")
    print("   - With learned feasibility constraints")
    print()
    print("Note: TuRBO is single-objective only.")
    print("      Cost-aware is also single-objective only.")


def demo_method_selection() -> None:
    """Show automatic method selection."""
    print_header("METHOD SELECTION API")

    scenarios = [
        ("Low-dim, no data", 5, 1, 0),
        ("Low-dim, some data", 5, 1, 10),
        ("High-dim, some data", 30, 1, 15),
        ("Multi-objective", 5, 2, 10),
    ]

    for name, n_params, n_obj, n_obs in scenarios:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                for i in range(n_params)
            ],
            objectives=[ObjectiveSpec(name=f"f{i}", minimize=True) for i in range(n_obj)],
        )
        methods = select_methods(spec, n_observations=n_obs)

        print(f"\n{name} ({n_params} params, {n_obj} obj, {n_obs} obs):")
        print(f"  Model: {methods.model_type}")
        print(f"  Acquisition: {methods.acquisition_function}")
        print(f"  Strategy: {methods.optimization_strategy}")
        print(f"  Confidence: {methods.confidence}")


def main() -> None:
    """Run all demonstrations."""
    print("=" * 70)
    print("BO-ENGINE v1.2 + v1.3 COMPREHENSIVE FEATURE DEMONSTRATION")
    print("=" * 70)

    torch.manual_seed(42)

    demo_version_overview()
    demo_turbo_summary()
    demo_outcome_constraints_summary()
    demo_cost_aware_summary()
    demo_combined_features()
    demo_method_selection()

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("For detailed demos of each feature, see:")
    print("  - scripts/turbo_demo.py")
    print("  - scripts/outcome_constraints_demo.py")
    print("  - scripts/cost_aware_demo.py")


if __name__ == "__main__":
    main()
