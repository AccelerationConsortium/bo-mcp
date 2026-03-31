#!/usr/bin/env python3
"""Demonstration of TuRBO (Trust Region Bayesian Optimization).

TuRBO is designed for high-dimensional optimization (20+ parameters)
where standard BO struggles due to the curse of dimensionality.

This script demonstrates:
1. TuRBO for high-dimensional single-objective optimization
2. Trust region dynamics (expansion on success, contraction on failure)
3. Comparison with standard BO

Usage:
    uv run python scripts/turbo_demo.py
"""

import sys
from pathlib import Path

import torch

# Add packages to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "bo-engine" / "src"))

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    TurboConfig,
    TurboState,
    create_turbo_state,
    generate_next_batch,
    should_use_turbo,
    update_turbo_after_evaluation,
)


def levy_function(x: dict[str, float]) -> float:
    """Levy function - a challenging high-dimensional test function.

    Has many local minima, making it ideal for testing TuRBO.
    Global minimum at x_i = 1 for all i, f(x*) = 0.
    """
    import math

    values = list(x.values())
    n = len(values)
    w = [(1 + (xi - 1) / 4) for xi in values]

    term1 = math.sin(math.pi * w[0]) ** 2
    term2 = sum(
        (w[i] - 1) ** 2 * (1 + 10 * math.sin(math.pi * w[i] + 1) ** 2) for i in range(n - 1)
    )
    term3 = (w[-1] - 1) ** 2 * (1 + math.sin(2 * math.pi * w[-1]) ** 2)

    return term1 + term2 + term3


def demo_turbo_basics() -> None:
    """Demonstrate TuRBO basic functionality."""
    print("\n" + "=" * 60)
    print("1. TuRBO BASICS")
    print("=" * 60)
    print()
    print("TuRBO (Trust Region Bayesian Optimization) manages a trust region")
    print("that expands on consecutive successes and contracts on failures.")
    print()

    # Check if TuRBO is recommended
    print("TuRBO recommendation check:")
    for n_params in [5, 10, 15, 20, 25, 50]:
        recommended = should_use_turbo(n_params)
        status = "RECOMMENDED" if recommended else "not needed"
        print(f"  {n_params:2d} parameters: {status}")
    print()

    # Create TuRBO state
    state = create_turbo_state(dim=25, batch_size=4, initial_best_value=-10.0)
    print("Initial TuRBO state:")
    print(f"  Dimension: {state.dim}")
    print(f"  Batch size: {state.batch_size}")
    print(f"  Trust region length: {state.length:.4f}")
    print(f"  Success tolerance: {state.success_tolerance}")
    print(f"  Failure tolerance: {state.failure_tolerance}")
    print()


def demo_turbo_dynamics() -> None:
    """Demonstrate trust region dynamics."""
    print("\n" + "=" * 60)
    print("2. TRUST REGION DYNAMICS")
    print("=" * 60)
    print()
    print("Trust region expands after 10 consecutive improvements,")
    print("contracts after failure_tolerance consecutive failures.")
    print()

    state = create_turbo_state(dim=25, batch_size=4, initial_best_value=0.0)
    print(f"Initial length: {state.length:.4f}")
    print()

    # Simulate successes
    print("Simulating 10 consecutive improvements:")
    for i in range(10):
        # Each improvement is better than best_value
        new_y = torch.tensor([state.best_value + 0.1 * (i + 1)])
        state = update_turbo_after_evaluation(
            turbo_state=state,
            new_observations=[
                ObservationData(
                    parameter_values={f"x{j}": 0.5 for j in range(25)},
                    objective_values={"f": -new_y.item()},  # Negated for minimization
                )
            ],
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name=f"x{j}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                    for j in range(25)
                ],
                objectives=[ObjectiveSpec(name="f", minimize=True)],
            ),
        )
        print(
            f"  Iteration {i + 1}: length={state.length:.4f}, "
            f"success_counter={state.success_counter}"
        )

    print(f"\nFinal length after expansion: {state.length:.4f}")
    print()

    # Simulate failures
    print("Simulating failures (no improvement):")
    assert state.failure_tolerance is not None
    for i in range(state.failure_tolerance + 1):
        # No improvement
        new_y = torch.tensor([state.best_value - 1.0])  # Worse than best
        state = update_turbo_after_evaluation(
            turbo_state=state,
            new_observations=[
                ObservationData(
                    parameter_values={f"x{j}": 0.5 for j in range(25)},
                    objective_values={"f": -new_y.item()},
                )
            ],
            spec=OptimizationSpec(
                parameters=[
                    ParameterSpec(name=f"x{j}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                    for j in range(25)
                ],
                objectives=[ObjectiveSpec(name="f", minimize=True)],
            ),
        )
        print(
            f"  Failure {i + 1}: length={state.length:.4f}, failure_counter={state.failure_counter}"
        )
        if state.restart_triggered:
            print("  ** RESTART TRIGGERED **")
            break


def demo_turbo_optimization() -> None:
    """Demonstrate TuRBO for high-dimensional optimization."""
    print("\n" + "=" * 60)
    print("3. TuRBO OPTIMIZATION LOOP")
    print("=" * 60)
    print()
    print("Optimizing 25-dimensional Levy function using TuRBO.")
    print("Global optimum: f(1, 1, ..., 1) = 0")
    print()

    n_params = 25
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(
                name=f"x{i}",
                type=ParameterType.CONTINUOUS,
                bounds=(-5.0, 10.0),  # Levy function domain
            )
            for i in range(n_params)
        ],
        objectives=[ObjectiveSpec(name="f", minimize=True)],
        batch_size=4,
        turbo_config=TurboConfig(),
    )

    observations: list[ObservationData] = []
    turbo_state: TurboState | None = None
    best_value = float("inf")

    # Run optimization
    n_iterations = 5
    print(f"Running {n_iterations} iterations (batch size 4):")
    print()

    for iteration in range(n_iterations):
        # Generate suggestions
        suggestions, turbo_state = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=4,
            iteration=iteration,
            turbo_state=turbo_state,
        )

        # Evaluate suggestions
        new_observations = []
        for sugg in suggestions:
            f_val = levy_function(sugg.parameter_values)
            obs = ObservationData(
                parameter_values=sugg.parameter_values,
                objective_values={"f": f_val},
            )
            new_observations.append(obs)
            observations.append(obs)

            if f_val < best_value:
                best_value = f_val

        # Update TuRBO state
        if turbo_state is not None:
            turbo_state = update_turbo_after_evaluation(
                turbo_state=turbo_state,
                new_observations=new_observations,
                spec=spec,
            )

        # Report progress
        generation_method = suggestions[0].generation_method if suggestions else "unknown"
        tr_length = turbo_state.length if turbo_state else "N/A"
        print(f"  Iteration {iteration + 1}:")
        print(f"    Method: {generation_method}")
        print(f"    Trust region length: {tr_length}")
        print(f"    Best f value: {best_value:.4f}")
        print()

    print(f"Final best value: {best_value:.4f}")
    print("(Optimal value is 0.0)")


def demo_comparison() -> None:
    """Compare TuRBO vs standard BO conceptually."""
    print("\n" + "=" * 60)
    print("4. TuRBO vs STANDARD BO")
    print("=" * 60)
    print()
    print("TuRBO advantages for high-dimensional problems:")
    print()
    print("  1. LOCAL SEARCH: TuRBO focuses on a local trust region,")
    print("     avoiding the curse of dimensionality in high-D spaces.")
    print()
    print("  2. ADAPTIVE: Trust region expands when making progress,")
    print("     contracts when stuck, automatically balancing explore/exploit.")
    print()
    print("  3. SCALABLE: Effective for 20-100+ dimensional problems")
    print("     where standard BO degrades significantly.")
    print()
    print("  4. RESTART: When trust region becomes too small,")
    print("     TuRBO can restart from a new center point.")
    print()
    print("When NOT to use TuRBO:")
    print("  - Low-dimensional problems (<20 parameters)")
    print("  - Multi-objective optimization (not supported)")
    print("  - When global exploration is essential")


def main() -> None:
    """Run all TuRBO demonstrations."""
    print("=" * 60)
    print("TuRBO (TRUST REGION BAYESIAN OPTIMIZATION) DEMONSTRATION")
    print("=" * 60)

    torch.manual_seed(42)

    demo_turbo_basics()
    demo_turbo_dynamics()
    demo_turbo_optimization()
    demo_comparison()

    print("\n" + "=" * 60)
    print("TURBO DEMONSTRATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
