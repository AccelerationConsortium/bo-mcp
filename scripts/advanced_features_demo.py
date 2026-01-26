#!/usr/bin/env python3
"""Demonstration of v1.1 advanced BO features.

This script demonstrates:
1. qLogNParEGO alternative acquisition for multi-objective optimization
2. Input warping with Kumaraswamy CDF for non-stationary objectives
3. LOO Cross-Validation for model quality assessment

Usage:
    uv run python scripts/advanced_features_demo.py
"""

import math
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
    ParameterSpec,
    ParameterType,
    compute_hypervolume,
    compute_loo_cv_metrics,
    compute_pareto_front,
    generate_next_batch,
    get_reference_point,
)


def branin_currin(x1: float, x2: float) -> tuple[float, float]:
    """Branin-Currin bi-objective test function.

    Returns two objective values (both to be minimized).
    """
    # Branin function (modified)
    x1_b = x1 * 15 - 5
    x2_b = x2 * 15
    a, b, c, r, s, t = 1, 5.1 / (4 * math.pi**2), 5 / math.pi, 6, 10, 1 / (8 * math.pi)
    f1 = a * (x2_b - b * x1_b**2 + c * x1_b - r) ** 2 + s * (1 - t) * math.cos(x1_b) + s

    # Currin function
    factor1 = 1 - math.exp(-1 / (2 * x2 + 1e-10))
    numer = 2300 * x1**3 + 1900 * x1**2 + 2092 * x1 + 60
    denom = 100 * x1**3 + 500 * x1**2 + 4 * x1 + 20
    f2 = factor1 * numer / denom

    return f1, f2


def demo_qlogparego() -> None:
    """Demonstrate qLogNParEGO acquisition function."""
    print("\n" + "=" * 60)
    print("1. qLogNParEGO ACQUISITION DEMO")
    print("=" * 60)
    print()
    print("qLogNParEGO uses Chebyshev scalarization with random weights")
    print("to explore different parts of the Pareto front.")
    print()

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f1", minimize=True),
            ObjectiveSpec(name="f2", minimize=True),
        ],
        batch_size=3,
        acquisition_method=AcquisitionMethod.QLOGPAREGO,
    )

    print(f"Acquisition method: {spec.acquisition_method.value}")
    print("Objectives: f1 (minimize), f2 (minimize)")
    print()

    observations: list[ObservationData] = []

    # Initial design
    for iteration in range(3):
        suggestions = generate_next_batch(spec, observations, iteration=iteration)

        for sugg in suggestions:
            x1, x2 = sugg.parameter_values["x1"], sugg.parameter_values["x2"]
            f1, f2 = branin_currin(x1, x2)
            observations.append(
                ObservationData(
                    parameter_values={"x1": x1, "x2": x2},
                    objective_values={"f1": f1, "f2": f2},
                )
            )

        if iteration >= 1:  # After initial design
            print(f"Iteration {iteration + 1}:")
            print(f"  Acquisition: {suggestions[0].acquisition_function}")
            print(f"  Generated {len(suggestions)} suggestions")

    # Compute Pareto front
    y_tensor = torch.tensor(
        [[obs.objective_values["f1"], obs.objective_values["f2"]] for obs in observations],
        dtype=torch.double,
    )
    pareto_y, pareto_mask = compute_pareto_front(y_tensor)

    print("\nResults after 3 iterations:")
    print(f"  Total evaluations: {len(observations)}")
    print(f"  Pareto-optimal points: {pareto_y.shape[0]}")


def demo_input_warping() -> None:
    """Demonstrate input warping with Kumaraswamy CDF."""
    print("\n" + "=" * 60)
    print("2. INPUT WARPING DEMO (Kumaraswamy CDF)")
    print("=" * 60)
    print()
    print("Input warping helps model non-stationary objective functions")
    print("where the response varies more in some regions than others.")
    print()

    # Non-stationary function: rapid change near x=0
    def non_stationary_objective(x: float) -> float:
        """Objective with rapid change near x=0."""
        return math.exp(-10 * x) + 0.1 * math.sin(10 * x)

    spec_with_warping = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f", minimize=True),
        ],
        batch_size=2,
        use_input_warping=True,
    )

    spec_without_warping = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f", minimize=True),
        ],
        batch_size=2,
        use_input_warping=False,
    )

    print("Test function: f(x) = exp(-10x) + 0.1*sin(10x)")
    print("This function changes rapidly near x=0")
    print()

    # Run both configurations
    for name, spec in [
        ("With Warping", spec_with_warping),
        ("Without Warping", spec_without_warping),
    ]:
        observations: list[ObservationData] = []

        for iteration in range(3):
            suggestions = generate_next_batch(spec, observations, iteration=iteration)

            for sugg in suggestions:
                x = sugg.parameter_values["x"]
                f = non_stationary_objective(x)
                observations.append(
                    ObservationData(
                        parameter_values={"x": x},
                        objective_values={"f": f},
                    )
                )

        best_f = min(obs.objective_values["f"] for obs in observations)
        print(f"{name}:")
        print(f"  Best f value: {best_f:.4f}")
        print(f"  Evaluations: {len(observations)}")
        print()


def demo_loo_cv() -> None:
    """Demonstrate LOO Cross-Validation for model quality."""
    print("\n" + "=" * 60)
    print("3. LOO CROSS-VALIDATION DEMO")
    print("=" * 60)
    print()
    print("Leave-One-Out CV provides unbiased estimates of model quality")
    print("by predicting each point using a model trained on all others.")
    print()

    torch.manual_seed(42)

    # Good fit scenario: simple linear function
    print("Scenario A: Simple linear function (should fit well)")
    n_samples = 12
    train_x = torch.rand(n_samples, 2, dtype=torch.double)
    train_y = train_x[:, 0:1] + train_x[:, 1:2]  # f(x) = x1 + x2
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    metrics_linear = compute_loo_cv_metrics(train_x, train_y, bounds)
    print(f"  LOO-CV RMSE: {metrics_linear.rmse:.4f}")
    print(f"  LOO-CV MAE: {metrics_linear.mae:.4f}")
    print(f"  LOO-CV R²: {metrics_linear.r_squared:.4f}")
    print()

    # Hard fit scenario: complex function with noise
    print("Scenario B: Complex noisy function (harder to fit)")
    train_y_noisy = (
        torch.sin(5 * train_x[:, 0:1]) * torch.cos(5 * train_x[:, 1:1])
        + torch.randn(n_samples, 1, dtype=torch.double) * 0.2
    )

    metrics_noisy = compute_loo_cv_metrics(train_x, train_y_noisy, bounds)
    print(f"  LOO-CV RMSE: {metrics_noisy.rmse:.4f}")
    print(f"  LOO-CV MAE: {metrics_noisy.mae:.4f}")
    print(f"  LOO-CV R²: {metrics_noisy.r_squared:.4f}")
    print()

    # Interpretation
    print("Interpretation:")
    print("  R² > 0.9: Excellent model fit")
    print("  R² 0.7-0.9: Good fit")
    print("  R² 0.5-0.7: Moderate fit, consider more data")
    print("  R² < 0.5: Poor fit, model may not be appropriate")


def demo_combined_features() -> None:
    """Demonstrate combining all new features."""
    print("\n" + "=" * 60)
    print("4. COMBINED FEATURES DEMO")
    print("=" * 60)
    print()
    print("Combining: qLogNParEGO + Input Warping + LOO-CV")
    print()

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f1", minimize=True),
            ObjectiveSpec(name="f2", minimize=True),
        ],
        batch_size=3,
        acquisition_method=AcquisitionMethod.QLOGPAREGO,
        use_input_warping=True,
    )

    print("Configuration:")
    print(f"  Acquisition: {spec.acquisition_method.value}")
    print(f"  Input Warping: {spec.use_input_warping}")
    print()

    observations: list[ObservationData] = []

    for iteration in range(4):
        suggestions = generate_next_batch(spec, observations, iteration=iteration)

        for sugg in suggestions:
            x1, x2 = sugg.parameter_values["x1"], sugg.parameter_values["x2"]
            f1, f2 = branin_currin(x1, x2)
            observations.append(
                ObservationData(
                    parameter_values={"x1": x1, "x2": x2},
                    objective_values={"f1": f1, "f2": f2},
                )
            )

    # Compute Pareto front and hypervolume
    y_tensor = torch.tensor(
        [[obs.objective_values["f1"], obs.objective_values["f2"]] for obs in observations],
        dtype=torch.double,
    )
    pareto_y, pareto_mask = compute_pareto_front(y_tensor)

    # Compute reference point and hypervolume
    minimize_mask = torch.tensor([True, True], dtype=torch.bool)
    ref_point = get_reference_point(y_tensor, minimize_mask)
    hv = compute_hypervolume(pareto_y, ref_point)

    print("Results:")
    print(f"  Total evaluations: {len(observations)}")
    print(f"  Pareto-optimal points: {pareto_y.shape[0]}")
    print(f"  Hypervolume: {hv:.4f}")

    # LOO-CV for model quality
    x_tensor = torch.tensor(
        [[obs.parameter_values["x1"], obs.parameter_values["x2"]] for obs in observations],
        dtype=torch.double,
    )
    y_f1 = y_tensor[:, 0:1]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    cv_metrics = compute_loo_cv_metrics(x_tensor, y_f1, bounds)
    print(f"  LOO-CV R² (f1): {cv_metrics.r_squared:.4f}")


def main() -> None:
    """Run all demonstrations."""
    print("=" * 60)
    print("ADVANCED BAYESIAN OPTIMIZATION FEATURES (v1.1)")
    print("=" * 60)

    demo_qlogparego()
    demo_input_warping()
    demo_loo_cv()
    demo_combined_features()

    print("\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
