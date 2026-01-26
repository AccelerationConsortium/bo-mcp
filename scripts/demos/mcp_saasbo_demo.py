#!/usr/bin/env python3
"""Demonstration of SAASBO (Sparse Axis-Aligned Subspace BO) via MCP Server.

This script demonstrates how to use the standalone MCP server with SAASBO
for very high-dimensional optimization (50+ parameters) where only a
subset of parameters significantly affects the objective.

Features demonstrated:
- SAASBO configuration for high-dimensional problems
- Sparse priors for automatic feature selection
- Parameter importance identification
- When to use SAASBO vs standard BO

Usage:
    uv run python scripts/demos/mcp_saasbo_demo.py
"""

import asyncio
import hashlib
import math
import random

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.get_diagnostics import get_diagnostics
from bo_mcp_server.tools.submit_results import submit_results
from bo_mcp_server.tools.validate_intake import validate_intake


def sparse_high_dim_objective(params: dict[str, float]) -> float:
    """High-dimensional objective where only a few parameters matter.

    True function depends heavily on:
    - x0, x1, x2, x3, x4 (important parameters, ~90% of effect)

    Other parameters are essentially noise (decoys).
    This is common in real-world problems where only a few factors matter.

    Args:
        params: Dictionary of parameter values

    Returns:
        Objective value to minimize
    """
    # Important parameters (high effect)
    x0 = params.get("x0", 0.5)
    x1 = params.get("x1", 0.5)
    x2 = params.get("x2", 0.5)
    x3 = params.get("x3", 0.5)
    x4 = params.get("x4", 0.5)

    # Main effect from important parameters
    # Optimal at: x0=0.3, x1=0.7, x2=0.5, x3=0.4, x4=0.6
    main_effect = (
        2.0 * (x0 - 0.3) ** 2
        + 3.0 * (x1 - 0.7) ** 2
        + 1.0 * (x2 - 0.5) ** 2
        + 1.5 * (x3 - 0.4) ** 2
        + 2.0 * (x4 - 0.6) ** 2
    )

    # Add some interactions
    interaction = 0.5 * math.sin(math.pi * x0 * x1) + 0.3 * math.cos(math.pi * x2 * x3)

    # Small noise from other parameters (minimal effect)
    noise = 0.0
    for name, value in params.items():
        if name not in ["x0", "x1", "x2", "x3", "x4"]:
            noise += 0.001 * (value - 0.5) ** 2

    return main_effect + interaction + noise + random.gauss(0, 0.01)


async def main() -> None:
    """Run SAASBO demonstration via MCP."""
    print("=" * 60)
    print("SAASBO DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("SAASBO (Sparse Axis-Aligned Subspace Bayesian Optimization)")
    print("is designed for high-dimensional problems where only a")
    print("subset of parameters significantly affect the objective.")
    print()
    print("Key features:")
    print("  - Sparse prior automatically identifies important parameters")
    print("  - Uses NUTS sampling for posterior inference")
    print("  - Effective in 50-200 dimensions")
    print("  - Best when <10% of parameters are truly important")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "saasbo-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("saasbo@example.com")
        if not user:
            user = User(
                name="SAASBO Demo User",
                email="saasbo@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Define high-dimensional optimization problem
    # Using 30 parameters for demo (SAASBO typically used for 50+)
    n_params = 30
    n_important = 5  # Only 5 parameters actually matter

    print(f"\n[3] Defining {n_params}-dimensional sparse optimization problem...")
    print(f"    (Only {n_important} parameters are truly important)")

    parameters = []
    for i in range(n_params):
        importance = "IMPORTANT" if i < n_important else "decoy"
        parameters.append(
            {
                "name": f"x{i}",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": f"Parameter {i} ({importance})",
            }
        )

    campaign_config = {
        "name": "SAASBO High-Dimensional Optimization",
        "description": f"Optimizing {n_params}D problem with {n_important} important parameters",
        "parameters": parameters,
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 4,
        # SAASBO configuration
        "use_saasbo": True,
        "acquisition_method": "SAASBO",
    }

    print(f"    Parameters: {n_params} continuous in [0, 1]")
    print(f"    Important: x0, x1, x2, x3, x4 (only {n_important} out of {n_params})")
    print("    Objective: minimize f")
    print("    Method: SAASBO with NUTS sampling")
    print()

    # Validate configuration
    print("[4] Validating configuration...")
    validation = await validate_intake(campaign_config)

    if not validation["valid"]:
        print(f"    ERROR: {validation['errors']}")
        return

    print("    Configuration is valid!")

    # Create campaign
    print("\n[5] Creating campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization iterations
    n_iterations = 4
    best_value = float("inf")
    best_params: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("SAASBO OPTIMIZATION LOOP")
    print("=" * 60)
    print()
    print("Note: SAASBO uses NUTS sampling which is computationally intensive.")
    print("For production 50+ dimensional problems, expect minutes per iteration.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions (SAASBO with NUTS sampling)...")
        print("    (This may take a moment...)")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Show method selection info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Model: {ms.get('model_type', 'N/A')}")
            print(f"    Strategy: {ms.get('optimization_strategy', 'N/A')}")

        # Run experiments
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            obj_value = sparse_high_dim_objective(params)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                }
            )

            # Show important parameters only
            important_vals = [params.get(f"x{i}", 0.5) for i in range(n_important)]
            vals_str = ", ".join(f"x{i}={v:.3f}" for i, v in enumerate(important_vals))
            print(f"    {vals_str} -> f={obj_value:.4f}")

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

    # Final summary
    print("\n" + "=" * 60)
    print("SAASBO DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total parameters: {n_params}")
    print(f"  Truly important: {n_important}")
    print(f"  Best f value: {best_value:.4f}")
    print()
    print("  Best parameters found (important ones):")
    print(f"    x0 = {best_params.get('x0', 0):.4f} (optimal: ~0.30)")
    print(f"    x1 = {best_params.get('x1', 0):.4f} (optimal: ~0.70)")
    print(f"    x2 = {best_params.get('x2', 0):.4f} (optimal: ~0.50)")
    print(f"    x3 = {best_params.get('x3', 0):.4f} (optimal: ~0.40)")
    print(f"    x4 = {best_params.get('x4', 0):.4f} (optimal: ~0.60)")
    print()
    print("Key Insight: SAASBO's sparse prior automatically focused on")
    print(f"the {n_important} important parameters while essentially ignoring")
    print(f"the {n_params - n_important} irrelevant parameters.")
    print()
    print("When to use SAASBO:")
    print("  - Very high-dimensional problems (50-200+ parameters)")
    print("  - Suspected sparsity (only ~10% of parameters matter)")
    print("  - Limited evaluation budget")
    print("  - Single-objective optimization")
    print()
    print("When NOT to use SAASBO:")
    print("  - Low-dimensional problems (<30 parameters)")
    print("  - Most parameters are important (not sparse)")
    print("  - Multi-objective optimization")
    print("  - Need fast suggestions (NUTS is slow)")
    print()
    print("Trade-offs:")
    print("  - NUTS sampling is computationally expensive")
    print("  - Best for <200 observations (O(n^3) scaling)")
    print("  - May struggle if problem is not truly sparse")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
