#!/usr/bin/env python3
"""Adaptive Batch Bayesian Optimization Demo.

Demonstrates intelligent BO with:
- Minimal user configuration (just select a benchmark)
- Automatic method selection that adapts over iterations
- SQLite tracking of all inputs, results, and model performance
- Batch evaluations (default batch_size=3)
- Plotly visualizations

Usage:
    uv run python scripts/demos/adaptive_bo_demo.py
    uv run python scripts/demos/adaptive_bo_demo.py --benchmark levy --dims 5
    uv run python scripts/demos/adaptive_bo_demo.py --iterations 20 --batch-size 5
    uv run python scripts/demos/adaptive_bo_demo.py --benchmark hartmann6 --iterations 15
"""

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_bo_benchmarks import BENCHMARKS, BenchmarkFunction, get_benchmark
from adaptive_bo_database import AdaptiveBODatabase
from adaptive_bo_plots import create_dashboard, save_plots
from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results
from bo_mcp_server.tools.validate_intake import validate_intake


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)


def print_iteration_header(iteration: int, total: int) -> None:
    """Print iteration header."""
    print(f"\n{'─' * 60}")
    print(f"ITERATION {iteration}/{total}")
    print("─" * 60)


def print_method_selection(ms: dict[str, Any]) -> None:
    """Display automatic method selection choices with explanations."""
    print("\n  Method Selection:")
    print(f"    Model:       {ms.get('model_type', 'N/A')}")
    print(f"    Acquisition: {ms.get('acquisition_function', 'N/A')}")
    print(f"    Strategy:    {ms.get('optimization_strategy', 'N/A')}")

    transforms = ms.get("input_transforms", [])
    if transforms:
        print(f"    Transforms:  {', '.join(transforms)}")

    confidence = ms.get("confidence", "N/A")
    print(f"    Confidence:  {confidence}")

    explanation = ms.get("explanation", "")
    if explanation:
        print(f"\n    Explanation: {explanation}")

    warnings = ms.get("warnings", [])
    if warnings:
        print("\n    Warnings:")
        for w in warnings:
            print(f"      ! {w}")


def print_config_summary(config: dict[str, Any], benchmark: BenchmarkFunction) -> None:
    """Print a summary of the campaign configuration."""
    print("\n  Configuration:")
    print(f"    Benchmark:   {benchmark.name}")
    print(f"    Parameters:  {len(config['parameters'])} ({benchmark.n_dims}D)")
    print(f"    Objectives:  {len(config['objectives'])}")
    print(f"    Batch size:  {config.get('batch_size', 1)}")
    print(f"    Optimum:     {benchmark.optimum:.6f}")

    print("\n    Parameter bounds:")
    for p in config["parameters"]:
        bounds = p.get("bounds", [])
        print(f"      {p['name']}: [{bounds[0]:.2f}, {bounds[1]:.2f}]")


def build_config_from_benchmark(
    benchmark: BenchmarkFunction,
    batch_size: int,
) -> dict[str, Any]:
    """Build minimal MCP campaign config from benchmark.

    Args:
        benchmark: Benchmark function instance.
        batch_size: Number of parallel evaluations per iteration.

    Returns:
        Campaign configuration dictionary.
    """
    parameters = [
        {
            "name": f"x{i + 1}",
            "type": "continuous",
            "bounds": list(bounds),
        }
        for i, bounds in enumerate(benchmark.bounds)
    ]

    # Determine objectives based on benchmark type
    if benchmark.is_multi_objective:
        # Multi-objective benchmarks return multiple values
        objectives = [
            {"name": "y1", "direction": "minimize"},
            {"name": "y2", "direction": "minimize"},
        ]
    else:
        objectives = [{"name": "y", "direction": "minimize"}]

    return {
        "name": f"Adaptive BO - {benchmark.name}",
        "description": f"Batch BO optimization of {benchmark.name} function",
        "parameters": parameters,
        "objectives": objectives,
        "batch_size": batch_size,
    }


def extract_model_metrics(
    diagnostics: dict[str, Any],
    method_selection: dict[str, Any],
) -> dict[str, float | dict[str, Any]]:
    """Extract model performance metrics from MCP diagnostics.

    Args:
        diagnostics: Diagnostics from get_diagnostics tool.
        method_selection: Method selection info from generate_suggestions.

    Returns:
        Dictionary of metrics to log.
    """
    metrics: dict[str, float | dict[str, Any]] = {}

    # From diagnostics
    if "n_results" in diagnostics:
        metrics["n_observations"] = float(diagnostics["n_results"])

    # Confidence as a proxy for model quality
    confidence_map = {"high": 1.0, "medium": 0.5, "low": 0.2}
    if "confidence" in method_selection:
        confidence = method_selection["confidence"]
        metrics["confidence_score"] = confidence_map.get(confidence, 0.5)

    return metrics


async def setup_user() -> str:
    """Create or retrieve demo user.

    Returns:
        User ID string.
    """
    api_key = "adaptive-bo-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("adaptive-bo@example.com")
        if not user:
            user = User(
                name="Adaptive BO Demo User",
                email="adaptive-bo@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    return str(user.id)


async def run_optimization(
    benchmark: BenchmarkFunction,
    n_iterations: int,
    batch_size: int,
    db: AdaptiveBODatabase,
    owner_id: str,
) -> tuple[str, float, dict[str, float]]:
    """Run the optimization loop.

    Args:
        benchmark: Benchmark function to optimize.
        n_iterations: Number of optimization iterations.
        batch_size: Batch size per iteration.
        db: Database for tracking.
        owner_id: Owner user ID.

    Returns:
        Tuple of (run_id, best_value, best_params).
    """
    # Build configuration
    config = build_config_from_benchmark(benchmark, batch_size)
    print_config_summary(config, benchmark)

    # Validate configuration
    print("\n[3] Validating configuration...")
    validation = await validate_intake(config)

    if not validation["valid"]:
        print(f"    ERROR: {validation['errors']}")
        raise ValueError(f"Invalid configuration: {validation['errors']}")

    print("    Configuration is valid!")

    # Create campaign
    print("\n[4] Creating campaign...")
    result = await create_campaign(config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        raise ValueError(f"Failed to create campaign: {result['errors']}")

    campaign_id = result["campaign_id"]
    print(f"    Campaign ID: {campaign_id}")

    # Create tracking run in our database
    run_id = await db.create_run(
        name=f"Adaptive BO - {benchmark.name}",
        benchmark=benchmark.name,
        n_params=benchmark.n_dims,
        n_objectives=2 if benchmark.is_multi_objective else 1,
        batch_size=batch_size,
        mcp_campaign_id=campaign_id,
    )
    print(f"    Run ID: {run_id}")

    # Optimization loop
    print_header("OPTIMIZATION LOOP")

    best_value = float("inf")
    best_params: dict[str, float] = {}

    for iteration in range(1, n_iterations + 1):
        print_iteration_header(iteration, n_iterations)

        # Log iteration in our database
        iteration_id = await db.log_iteration(run_id, iteration)

        # Generate suggestions
        print("\n  [A] Generating suggestions...")
        sugg_result = await generate_suggestions(campaign_id)

        if not sugg_result["success"]:
            print(f"    ERROR: {sugg_result['errors']}")
            break

        suggestions = sugg_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Extract and log method selection
        method_selection = sugg_result.get("method_selection", {})
        print_method_selection(method_selection)
        await db.log_method_selection(iteration_id, method_selection)

        # Evaluate suggestions (batch)
        print("\n  [B] Evaluating on benchmark...")
        results_to_submit = []
        iteration_best = float("inf")
        is_improvement = False

        for batch_idx, suggestion in enumerate(suggestions):
            params = suggestion["parameter_values"]

            # Evaluate benchmark function
            obj_values = benchmark(**params)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": obj_values,
                    "suggestion_id": suggestion["id"],
                }
            )

            # Track best (for single-objective)
            if not benchmark.is_multi_objective:
                obj_value = list(obj_values.values())[0]
                iteration_best = min(iteration_best, obj_value)

                if obj_value < best_value:
                    best_value = obj_value
                    best_params = params.copy()
                    is_improvement = True

                print(f"    Batch {batch_idx + 1}: {obj_value:.6f}")
            else:
                obj_str = ", ".join(f"{k}={v:.4f}" for k, v in obj_values.items())
                print(f"    Batch {batch_idx + 1}: {obj_str}")

        # Log evaluations
        await db.log_evaluations(iteration_id, results_to_submit)

        # Submit results to MCP server
        print("\n  [C] Submitting results...")
        submit_result = await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

        if not submit_result["success"]:
            print(f"    ERROR: {submit_result['errors']}")
            break

        print(f"    Submitted {len(submit_result['result_ids'])} results")

        # Get diagnostics
        print("\n  [D] Getting diagnostics...")
        diag = await get_diagnostics(campaign_id)

        if diag["success"]:
            print(f"    Total evaluations: {diag['n_results']}")
            if "best_value" in diag and diag["best_value"] is not None:
                print(f"    Best value (server): {diag['best_value']:.6f}")

        # Log model performance metrics
        model_metrics = extract_model_metrics(diag, method_selection)
        await db.log_model_performance(iteration_id, model_metrics)

        # Log best value (single-objective only)
        if not benchmark.is_multi_objective:
            await db.log_best_value(
                iteration_id=iteration_id,
                objective_name="y",
                best_value=best_value,
                best_params=best_params,
                is_improvement=is_improvement,
            )

            print("\n  Summary:")
            print(f"    Iteration best: {iteration_best:.6f}")
            print(f"    Overall best:   {best_value:.6f}")
            print(f"    Gap to optimum: {abs(best_value - benchmark.optimum):.6f}")

    # Mark run as complete
    await db.complete_run(run_id)

    return run_id, best_value, best_params


async def main() -> None:
    """Run the adaptive BO demo."""
    args = parse_args()

    print_header("ADAPTIVE BATCH BO DEMO")
    print()
    print("This demo showcases intelligent Bayesian Optimization with:")
    print("  - Automatic model and acquisition function selection")
    print("  - Batch evaluations for efficiency")
    print("  - SQLite tracking of all optimization data")
    print("  - Interactive Plotly visualizations")
    print()

    # Initialize MCP database
    print("[1] Initializing MCP database...")
    await init_database()

    # Setup user
    print("\n[2] Setting up user...")
    owner_id = await setup_user()

    # Initialize our tracking database
    db_path = Path("data/adaptive_bo_demo.db")
    db = AdaptiveBODatabase(db_path)
    await db.init_schema()
    print(f"\n    Tracking database: {db_path}")

    # Get benchmark function
    benchmark = get_benchmark(args.benchmark, args.dims)
    print(f"\n    Benchmark: {benchmark.name}")
    print(f"    Dimensions: {benchmark.n_dims}")
    print(f"    Known optimum: {benchmark.optimum:.6f}")

    try:
        # Run optimization
        run_id, best_value, best_params = await run_optimization(
            benchmark=benchmark,
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            db=db,
            owner_id=owner_id,
        )

        # Generate visualizations
        print_header("GENERATING VISUALIZATIONS")

        output_dir = Path("data/plots") / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Output directory: {output_dir}")
        print("  Generating plots...")

        plot_files = await save_plots(
            db=db,
            run_id=run_id,
            output_dir=output_dir,
            optimum=benchmark.optimum if not benchmark.is_multi_objective else None,
        )

        print(f"\n  Saved {len(plot_files)} visualizations:")
        for pf in plot_files:
            print(f"    - {pf.name}")

        # Create and save dashboard
        dashboard = await create_dashboard(
            db=db,
            run_id=run_id,
            optimum=benchmark.optimum if not benchmark.is_multi_objective else None,
        )
        dashboard_path = output_dir / "dashboard.html"
        dashboard.write_html(str(dashboard_path))
        print(f"\n  Dashboard: {dashboard_path}")

        # Final summary
        print_header("OPTIMIZATION COMPLETE")
        print()
        print(f"  Benchmark:         {benchmark.name}")
        print(f"  Dimensions:        {benchmark.n_dims}")
        print(f"  Iterations:        {args.iterations}")
        print(f"  Batch size:        {args.batch_size}")
        print(f"  Total evaluations: {args.iterations * args.batch_size}")
        print()

        if not benchmark.is_multi_objective:
            print(f"  Best value found:  {best_value:.6f}")
            print(f"  Known optimum:     {benchmark.optimum:.6f}")
            print(f"  Gap:               {abs(best_value - benchmark.optimum):.6f}")
            print()
            print("  Best parameters:")
            for k, v in sorted(best_params.items()):
                print(f"    {k}: {v:.4f}")

        print()
        print(f"  Tracking database: {db_path}")
        print(f"  Visualizations:    {output_dir}")
        print()
        print("  To view dashboard, open in browser:")
        print(f"    open {dashboard_path}")

    finally:
        await db.close()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adaptive Batch Bayesian Optimization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple 2D Branin optimization
  uv run python scripts/demos/adaptive_bo_demo.py

  # 6D Hartmann function
  uv run python scripts/demos/adaptive_bo_demo.py --benchmark hartmann6

  # Scalable Levy function in 10D
  uv run python scripts/demos/adaptive_bo_demo.py --benchmark levy --dims 10

  # More iterations with larger batches
  uv run python scripts/demos/adaptive_bo_demo.py --iterations 20 --batch-size 5
        """,
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARKS.keys()),
        default="branin",
        help="Benchmark function to optimize (default: branin)",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=None,
        help="Number of dimensions (for scalable benchmarks like levy, ackley)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of optimization iterations (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=3,
        help="Batch size for parallel evaluations (default: 3)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main())
