#!/usr/bin/env python3
"""Demonstration of Input Warping via MCP Server.

This script demonstrates how to use the standalone MCP server with
input warping (Kumaraswamy CDF transform) for non-stationary objectives.

Features demonstrated:
- Input warping for non-stationary functions
- Kumaraswamy CDF transforms
- When input warping helps model fit
- Comparison with/without warping

Usage:
    uv run python scripts/demos/mcp_input_warping_demo.py
"""

import asyncio
import math
import random

from mcp_client_utils import get_or_create_demo_user

from bo_mcp_server.storage import init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results


def non_stationary_objective(x: float, y: float) -> float:
    """Non-stationary objective with rapid changes in some regions.

    The function changes rapidly near x=0 and y=0 but is smooth elsewhere.
    This is challenging for standard GPs with stationary kernels.
    """
    # Rapid change near origin
    rapid_part = 5.0 * math.exp(-10 * (x**2 + y**2))

    # Smooth part elsewhere
    smooth_part = math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)

    # Interaction term
    interaction = 2.0 * x * y

    return rapid_part + smooth_part + interaction + random.gauss(0, 0.1)


def heteroscedastic_objective(x: float, y: float) -> float:
    """Objective with varying sensitivity across the domain.

    Sensitivity is high near corners, low in the center.
    Input warping can help capture this heterogeneity.
    """
    # Base function
    base = (x - 0.5) ** 2 + (y - 0.5) ** 2

    # Higher sensitivity near corners
    corner_sensitivity = 0.0
    for cx, cy in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        dist_to_corner = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        corner_sensitivity += 3.0 * math.exp(-5 * dist_to_corner)

    return base + corner_sensitivity + random.gauss(0, 0.05)


async def run_warping_comparison(owner_id: str) -> None:
    """Run comparison between warped and non-warped optimization."""
    print("\n" + "=" * 60)
    print("INPUT WARPING COMPARISON")
    print("=" * 60)
    print()
    print("Comparing optimization with and without input warping")
    print("on a non-stationary test function.")
    print()

    results_summary = {}

    for use_warping in [False, True]:
        warping_str = "WITH" if use_warping else "WITHOUT"
        print(f"\n{'─' * 40}")
        print(f"Running {warping_str} input warping")
        print("─" * 40)

        campaign_config = {
            "name": f"Input Warping Demo ({warping_str})",
            "description": f"Non-stationary optimization {warping_str.lower()} input warping",
            "parameters": [
                {
                    "name": "x",
                    "type": "continuous",
                    "bounds": [0.0, 1.0],
                },
                {
                    "name": "y",
                    "type": "continuous",
                    "bounds": [0.0, 1.0],
                },
            ],
            "objectives": [
                {"name": "f", "direction": "minimize", "unit": "value"},
            ],
            "batch_size": 3,
            "use_input_warping": use_warping,
        }

        result = await create_campaign(campaign_config, owner_id)
        if not result["success"]:
            print(f"    ERROR: {result['errors']}")
            continue

        campaign_id = result["campaign_id"]
        print(f"    Campaign: {campaign_id}")

        best_value = float("inf")
        n_iterations = 4

        for iteration in range(1, n_iterations + 1):
            suggestions_result = await generate_suggestions(campaign_id)
            if not suggestions_result["success"]:
                break

            suggestions = suggestions_result["suggestions"]
            results_to_submit = []

            for s in suggestions:
                params = s["parameter_values"]
                x = params.get("x", 0.5)
                y = params.get("y", 0.5)

                obj_value = non_stationary_objective(x, y)

                results_to_submit.append(
                    {
                        "parameter_values": params,
                        "objective_values": {"f": obj_value},
                        "suggestion_id": s["suggestion_id"],
                    }
                )

                if obj_value < best_value:
                    best_value = obj_value

            await submit_results(
                campaign_id=campaign_id,
                results=results_to_submit,
                submitted_by=owner_id,
                source="api",
            )

            print(f"    Iteration {iteration}: best = {best_value:.4f}")

        results_summary[warping_str] = best_value
        print(f"    Final best: {best_value:.4f}")

    # Compare results
    print("\n" + "=" * 40)
    print("COMPARISON RESULTS")
    print("=" * 40)
    for config, best in results_summary.items():
        print(f"  {config:12s}: {best:.4f}")


async def main() -> None:
    """Run input warping demonstration via MCP."""
    print("=" * 60)
    print("INPUT WARPING DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Input warping uses Kumaraswamy CDF transforms to handle")
    print("non-stationary objective functions where the response")
    print("varies more rapidly in some regions than others.")
    print()
    print("How it works:")
    print("  1. Inputs are transformed: x' = Kumaraswamy_CDF(x)")
    print("  2. Transform parameters are learned from data")
    print("  3. Warped space is more suitable for stationary GP")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    owner_id = await get_or_create_demo_user(
        email="warping@example.com",
        name="Input Warping Demo User",
        api_key="input-warping-demo-key",
    )
    print(f"    Using demo user: {owner_id}")

    # Run main demonstration
    print("\n[3] Defining non-stationary optimization problem...")

    campaign_config = {
        "name": "Input Warping Main Demo",
        "description": "Optimizing non-stationary function with input warping",
        "parameters": [
            {
                "name": "x",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "First parameter",
            },
            {
                "name": "y",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Second parameter",
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
        "use_input_warping": True,
    }

    print("    Parameters: x, y in [0, 1]")
    print("    Objective: minimize f (non-stationary function)")
    print("    Input warping: ENABLED")
    print()

    # Create campaign
    print("[4] Creating campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization
    n_iterations = 5
    best_value = float("inf")
    best_params: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("OPTIMIZATION WITH INPUT WARPING")
    print("=" * 60)
    print()
    print("The model learns optimal input transforms from data.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions (with input warping)...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x = params.get("x", 0.5)
            y = params.get("y", 0.5)

            obj_value = non_stationary_objective(x, y)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["suggestion_id"],
                }
            )

            print(f"    x={x:.3f}, y={y:.3f} -> f={obj_value:.4f}")

            if obj_value < best_value:
                best_value = obj_value
                best_params = params.copy()

        # Submit results
        print("\n[C] Submitting results...")
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
        print("\n[D] Checking progress...")
        diagnostics = await get_diagnostics(campaign_id)

        if diagnostics["success"]:
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best f value: {best_value:.4f}")

    # Run comparison
    await run_warping_comparison(owner_id)

    # Final summary
    print("\n" + "=" * 60)
    print("INPUT WARPING DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nMain Demo Results:")
    print(f"  Best f value: {best_value:.4f}")
    print("  Best parameters:")
    print(f"    x = {best_params.get('x', 0):.4f}")
    print(f"    y = {best_params.get('y', 0):.4f}")
    print()
    print("Key Insight: Input warping transforms the input space to")
    print("make non-stationary functions more suitable for GP modeling.")
    print("The Kumaraswamy CDF transform learns concentration parameters")
    print("that effectively 'stretch' the input space where needed.")
    print()
    print("When to use Input Warping:")
    print("  - Non-stationary objectives (rapid change in some regions)")
    print("  - Varying sensitivity across the domain")
    print("  - GP model has poor fit without warping")
    print()
    print("When NOT to use Input Warping:")
    print("  - Stationary objectives (constant smoothness)")
    print("  - Very few observations (can't learn transform)")
    print("  - Transform adds unnecessary complexity")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
