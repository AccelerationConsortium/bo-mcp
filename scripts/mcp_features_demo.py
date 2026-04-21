#!/usr/bin/env python3
"""Demo script illustrating the MCP server features for BO (v1.0.1/v1.1).

This script demonstrates the Bayesian Optimization features exposed through
the MCP server, including:
- Single-objective optimization with qLogNEI
- Multi-objective optimization with qLogNEHVI and qLogNParEGO
- Input warping for non-stationary objectives
- LOO Cross-Validation diagnostics

Note: This script demonstrates the bo-engine functionality that is
      exposed through the MCP server tools. The actual MCP server
      provides these features via the following tools:
      - generate_suggestions: Generates next batch of experiments
      - get_diagnostics: Returns optimization diagnostics
      - create_campaign: Creates a new optimization campaign
      - submit_results: Submits experimental results
"""

from bo_engine import (
    AcquisitionMethod,
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    compute_best_value,
    compute_improvement_history,
    generate_next_batch,
)
from bo_engine.diagnostics import (
    compute_loo_cv_for_model,
    compute_single_objective_improvement_rate,
)
from bo_engine.models import create_and_fit_single_task_model


def demo_single_objective_optimization():
    """Demonstrate single-objective optimization (v1.0.1 feature).

    This is the functionality exposed by the MCP server's
    generate_suggestions tool when a campaign has a single objective.
    """
    print("=" * 70)
    print("DEMO 1: Single-Objective Optimization (v1.0.1)")
    print("=" * 70)
    print()

    # Define optimization problem: minimize f(x) = (x - 0.7)^2
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f", minimize=True),
        ],
        batch_size=2,
        acquisition_method=AcquisitionMethod.AUTO,  # Will use qLogNEI for single-objective
    )

    print("Problem: Minimize f(x) = (x - 0.7)²")
    print(f"Parameters: {[p.name for p in spec.parameters]}")
    print(f"Objectives: {[o.name for o in spec.objectives]}")
    print(f"Acquisition Method: {spec.acquisition_method.value} -> qLogNEI (auto-selected)")
    print()

    # Simulate optimization iterations
    def objective(x: float) -> float:
        return (x - 0.7) ** 2

    observations: list[ObservationData] = []

    print("Running optimization iterations...")
    print("-" * 50)

    for iteration in range(3):
        # Generate suggestions (this is what MCP's generate_suggestions does)
        suggestions = generate_next_batch(spec, observations, iteration=iteration)

        print(f"\nIteration {iteration + 1}:")
        acq_func = suggestions[0].acquisition_function  # ty: ignore[unresolved-attribute]
        print(f"  Generated {len(suggestions)} suggestions using {acq_func}")

        for sugg in suggestions:
            x = sugg.parameter_values["x"]  # ty: ignore[unresolved-attribute]
            f = objective(x)
            print(f"    x = {x:.4f}, f(x) = {f:.4f}, acq_value = {sugg.acquisition_value}")  # ty: ignore[unresolved-attribute]
            observations.append(
                ObservationData(
                    parameter_values={"x": x},
                    objective_values={"f": f},
                )
            )

    # Compute diagnostics (this is what MCP's get_diagnostics does)
    values = [obs.objective_values["f"] for obs in observations]
    best_value, best_idx = compute_best_value(values, minimize=True)
    improvement_history = compute_improvement_history(values, minimize=True)
    improvement_rate = compute_single_objective_improvement_rate(improvement_history)

    print()
    print("Diagnostics (from get_diagnostics MCP tool):")
    print("-" * 50)
    print(f"  Best value: {best_value:.4f}")
    print(f"  Best parameters: {observations[best_idx].parameter_values}")
    print(f"  Improvement history: {[f'{v:.4f}' for v in improvement_history]}")
    print(f"  Improvement rate: {improvement_rate:.2%}")
    print()


def demo_multi_objective_with_qlogparego():
    """Demonstrate qLogNParEGO acquisition (v1.1 feature).

    The MCP server allows selecting qLogNParEGO as an alternative
    acquisition function for multi-objective optimization.
    """
    print("=" * 70)
    print("DEMO 2: Multi-Objective with qLogNParEGO (v1.1)")
    print("=" * 70)
    print()

    # Define multi-objective problem
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f1", minimize=True),
            ObjectiveSpec(name="f2", minimize=True),
        ],
        batch_size=2,
        acquisition_method=AcquisitionMethod.SCALARIZED_MULTI_OBJ,
    )

    print("Problem: Minimize f1 and f2 (conflicting objectives)")
    print(f"Acquisition Method: {spec.acquisition_method.value}")
    print()

    # Conflicting objectives
    def f1(x1: float, x2: float) -> float:
        return x1**2 + x2**2

    def f2(x1: float, x2: float) -> float:
        return (x1 - 1) ** 2 + (x2 - 1) ** 2

    observations: list[ObservationData] = []

    # Initial observations
    initial_points = [(0.2, 0.3), (0.8, 0.7), (0.5, 0.5)]
    for x1, x2 in initial_points:
        observations.append(
            ObservationData(
                parameter_values={"x1": x1, "x2": x2},
                objective_values={"f1": f1(x1, x2), "f2": f2(x1, x2)},
            )
        )

    print("Initial observations:")
    for obs in observations:
        print(
            f"  x1={obs.parameter_values['x1']:.2f}, x2={obs.parameter_values['x2']:.2f} "
            f"-> f1={obs.objective_values['f1']:.3f}, f2={obs.objective_values['f2']:.3f}"
        )
    print()

    # Generate qLogNParEGO suggestions
    suggestions = generate_next_batch(spec, observations, iteration=1)

    print("qLogNParEGO suggestions (iteration 1):")
    for i, sugg in enumerate(suggestions):
        print(f"  Suggestion {i + 1}:")
        print(
            f"    Parameters: x1={sugg.parameter_values['x1']:.4f}, "  # ty: ignore[unresolved-attribute]
            f"x2={sugg.parameter_values['x2']:.4f}"  # ty: ignore[unresolved-attribute]
        )
        print(f"    Acquisition: {sugg.acquisition_function}")  # ty: ignore[unresolved-attribute]
        print(f"    Model: {sugg.model_type}")  # ty: ignore[unresolved-attribute]
    print()


def demo_input_warping():
    """Demonstrate input warping (v1.1 feature).

    Input warping with Kumaraswamy CDF helps model non-stationary
    objectives where the response varies across the parameter space.
    """
    print("=" * 70)
    print("DEMO 3: Input Warping with Kumaraswamy CDF (v1.1)")
    print("=" * 70)
    print()

    # Define problem with input warping enabled
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f", minimize=True),
        ],
        batch_size=2,
        use_input_warping=True,  # Enable Kumaraswamy input warping
    )

    print(f"Input warping enabled: {spec.use_input_warping}")
    print("This helps with non-stationary objectives where response varies across regions.")
    print()

    # Non-uniform data (clustered near 0, sparse near 1)
    observations = [
        ObservationData(parameter_values={"x1": 0.1, "x2": 0.1}, objective_values={"f": 1.0}),
        ObservationData(parameter_values={"x1": 0.2, "x2": 0.15}, objective_values={"f": 0.8}),
        ObservationData(parameter_values={"x1": 0.15, "x2": 0.2}, objective_values={"f": 0.9}),
        ObservationData(parameter_values={"x1": 0.9, "x2": 0.9}, objective_values={"f": 2.0}),
    ]

    print("Training data (non-uniform distribution):")
    for obs in observations:
        print(
            f"  x1={obs.parameter_values['x1']:.2f}, x2={obs.parameter_values['x2']:.2f} "
            f"-> f={obs.objective_values['f']:.2f}"
        )
    print()

    # Generate suggestions with warping
    suggestions = generate_next_batch(spec, observations, iteration=1)

    print("Suggestions (with input warping):")
    for i, sugg in enumerate(suggestions):
        print(
            f"  {i + 1}. x1={sugg.parameter_values['x1']:.4f}, x2={sugg.parameter_values['x2']:.4f}"  # ty: ignore[unresolved-attribute]
        )
    print()


def _loo_quality(r_squared: float) -> str:
    if r_squared > 0.9:
        return "Excellent"
    if r_squared > 0.7:
        return "Good"
    if r_squared > 0.5:
        return "Moderate"
    return "Poor"


def demo_loo_cv_diagnostics():
    """Demonstrate LOO Cross-Validation diagnostics (v1.1 feature).

    LOO-CV provides model quality metrics (RMSE, MAE, R²) that the
    MCP server's get_diagnostics tool returns.
    """
    print("=" * 70)
    print("DEMO 4: LOO Cross-Validation Diagnostics (v1.1)")
    print("=" * 70)
    print()

    import torch
    from bo_engine.transforms import get_bounds_tensor

    # Create spec with enough data for LOO-CV
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 10.0)),
        ],
        objectives=[
            ObjectiveSpec(name="f", minimize=True),
        ],
    )

    # Generate observations: f(x) = (x - 5)^2 + noise
    import random

    random.seed(42)

    observations = []
    for x in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]:
        f = (x - 5) ** 2 + random.gauss(0, 0.5)
        observations.append(
            ObservationData(
                parameter_values={"x": x},
                objective_values={"f": f},
            )
        )

    print("Observations (f(x) = (x-5)² + noise):")
    for obs in observations:
        print(f"  x={obs.parameter_values['x']:.1f} -> f={obs.objective_values['f']:.2f}")
    print()

    # Prepare tensors for LOO-CV computation
    x_list = [torch.tensor([obs.parameter_values["x"]], dtype=torch.double) for obs in observations]
    train_x = torch.stack(x_list)
    train_y = torch.tensor(
        [[obs.objective_values["f"]] for obs in observations], dtype=torch.double
    )
    bounds = get_bounds_tensor(spec)

    # Fit model and compute LOO-CV metrics
    model = create_and_fit_single_task_model(train_x, train_y, bounds)
    loo_metrics = compute_loo_cv_for_model(model, train_x, train_y)

    print("LOO Cross-Validation Metrics (from get_diagnostics):")
    print("-" * 50)
    if isinstance(loo_metrics, dict):
        for idx, obj in enumerate(spec.objectives):
            if idx in loo_metrics:
                m = loo_metrics[idx]  # ty: ignore[invalid-argument-type]
                print(f"  Objective '{obj.name}':")
                print(f"    RMSE: {m.rmse:.4f}")  # ty: ignore[unresolved-attribute]
                print(f"    MAE: {m.mae:.4f}")  # ty: ignore[unresolved-attribute]
                print(f"    R²: {m.r_squared:.4f}")  # ty: ignore[unresolved-attribute]
                print(f"    Quality: {_loo_quality(m.r_squared)}")  # ty: ignore[unresolved-attribute]
    else:
        obj = spec.objectives[0]
        print(f"  Objective '{obj.name}':")
        print(f"    RMSE: {loo_metrics.rmse:.4f}")
        print(f"    MAE: {loo_metrics.mae:.4f}")
        print(f"    R²: {loo_metrics.r_squared:.4f}")
        print(f"    Quality: {_loo_quality(loo_metrics.r_squared)}")
    print()


def show_mcp_server_tools():
    """Show the MCP server tools that expose these features."""
    print("=" * 70)
    print("MCP SERVER TOOLS")
    print("=" * 70)
    print()

    tools_info = {
        "generate_suggestions": {
            "description": "Generate next batch of experiment suggestions",
            "new_features": [
                "Single-objective support (v1.0.1): Uses qLogNEI acquisition",
                "qLogNParEGO alternative (v1.1): Set acquisition_method='qLogNParEGO'",
                "Input warping (v1.1): Set use_input_warping=True in campaign spec",
            ],
            "file": "packages/bo-mcp-server/src/mcp_server/tools/generate_suggestions.py",
        },
        "get_diagnostics": {
            "description": "Get diagnostic information for a campaign",
            "new_features": [
                "Single-objective diagnostics (v1.0.1): best_value, improvement_history",
                "LOO-CV metrics (v1.1): loo_cv_metrics with RMSE, MAE, R²",
                "Model info now includes input_warping and actual acquisition function",
            ],
            "file": "packages/bo-mcp-server/src/mcp_server/tools/get_diagnostics.py",
        },
    }

    for tool_name, info in tools_info.items():
        print(f"Tool: {tool_name}")
        print(f"  Description: {info['description']}")
        print(f"  File: {info['file']}")
        print("  New Features (v1.0.1/v1.1):")
        for feature in info["new_features"]:
            print(f"    - {feature}")
        print()

    print("Campaign Spec New Fields:")
    print("-" * 50)
    print("  acquisition_method: AcquisitionMethod")
    print("    - 'auto' (default): Auto-selects based on n_objectives")
    print("    - 'qLogNEI': Single-objective (noisy)")
    print("    - 'qLogEI': Single-objective (noiseless)")
    print("    - 'qLogNEHVI': Multi-objective (hypervolume)")
    print("    - 'qLogNParEGO': Multi-objective (Chebyshev scalarization)")
    print()
    print("  use_input_warping: bool")
    print("    - False (default): Standard normalization only")
    print("    - True: Kumaraswamy CDF warping for non-stationary objectives")
    print()


def main():
    """Run all demos."""
    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║            BO-MCP-UI: New Features Demo (v1.0.1 / v1.1)              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("This script demonstrates the Bayesian Optimization features")
    print("exposed through the MCP server tools.")
    print()

    demo_single_objective_optimization()
    demo_multi_objective_with_qlogparego()
    demo_input_warping()
    demo_loo_cv_diagnostics()
    show_mcp_server_tools()

    print("=" * 70)
    print("SUMMARY: What's Implemented vs What's NOT")
    print("=" * 70)
    print()
    print("✓ IMPLEMENTED (v1.0.1):")
    print("  - Single-objective optimization with qLogNEI")
    print("  - SingleTaskGP model for single-objective")
    print("  - Best value tracking and improvement history")
    print()
    print("✓ IMPLEMENTED (v1.1):")
    print("  - qLogNParEGO as alternative acquisition for multi-objective")
    print("  - Input warping with Kumaraswamy CDF")
    print("  - LOO Cross-Validation diagnostics (RMSE, MAE, R²)")
    print()
    print("✗ NOT IMPLEMENTED (planned for v1.2):")
    print("  - SAASBO (Sparse Axis-Aligned Subspace BO)")
    print("  - High-dimensional optimization (>20 dimensions)")
    print()


if __name__ == "__main__":
    main()
