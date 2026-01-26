#!/usr/bin/env python3
"""Demonstration of Single-Objective Bayesian Optimization via MCP Server.

This script demonstrates how to use the standalone MCP server for single-objective
optimization without REST API or frontend.

Features demonstrated:
- Campaign creation with single objective (minimize/maximize)
- Suggestion generation using qLogNEI acquisition
- Result submission and diagnostics
- Best value tracking

Usage:
    uv run python scripts/demos/mcp_single_objective_demo.py
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


def branin_function(x1: float, x2: float) -> float:
    """Branin test function - a standard single-objective benchmark.

    Domain: x1 in [-5, 10], x2 in [0, 15]
    Global minima: f(x*) = 0.397887 at three locations.
    """
    a = 1.0
    b = 5.1 / (4 * math.pi**2)
    c = 5 / math.pi
    r = 6.0
    s = 10.0
    t = 1 / (8 * math.pi)

    result = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
    return result + random.gauss(0, 0.1)


async def main() -> None:
    """Run single-objective BO demonstration via MCP."""
    print("=" * 60)
    print("SINGLE-OBJECTIVE BO DEMONSTRATION (MCP Standalone)")
    print("=" * 60)
    print()
    print("Optimizing the Branin function using qLogNEI acquisition.")
    print("Global minimum: f* = 0.397887")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "single-objective-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("single-obj@example.com")
        if not user:
            user = User(
                name="Single Objective Demo User",
                email="single-obj@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Define single-objective optimization problem
    print("\n[3] Defining single-objective optimization problem...")

    campaign_config = {
        "name": "Branin Function Optimization",
        "description": "Single-objective optimization of Branin test function",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
                "description": "First design variable",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
                "description": "Second design variable",
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
    }

    print("    Parameters: x1 in [-5, 10], x2 in [0, 15]")
    print("    Objective: minimize f (Branin function)")
    print("    Batch size: 3")
    print()

    # Validate configuration
    print("[4] Validating configuration...")
    validation = await validate_intake(campaign_config)

    if not validation["valid"]:
        print(f"    ERROR: {validation['errors']}")
        return

    if validation["warnings"]:
        print(f"    Warnings: {validation['warnings']}")

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
    n_iterations = 5
    best_value = float("inf")
    best_params: dict[str, float] = {}

    print("\n" + "=" * 60)
    print("OPTIMIZATION LOOP")
    print("=" * 60)

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions
        print("\n[A] Generating suggestions...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Show method selection info
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Model: {ms['model_type']}")
            print(f"    Acquisition: {ms.get('acquisition_function', 'N/A')}")

        # Run experiments
        print("\n[B] Running experiments (evaluating Branin function)...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)

            obj_value = branin_function(x1, x2)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                }
            )

            print(f"    x1={x1:7.3f}, x2={x2:7.3f} -> f={obj_value:.4f}")

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
            print(f"    Health status: {diagnostics.get('health_status', 'N/A')}")
            print(f"    Progress: {diagnostics.get('progress_indicator', 'N/A')}")

    # Final summary
    print("\n" + "=" * 60)
    print("SINGLE-OBJECTIVE OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total iterations: {n_iterations}")
    print(f"  Best f value found: {best_value:.4f}")
    print("  Best parameters:")
    print(f"    x1 = {best_params.get('x1', 0):.4f}")
    print(f"    x2 = {best_params.get('x2', 0):.4f}")
    print()
    print("  Global optimum: f* = 0.397887")
    print(f"  Gap to optimum: {abs(best_value - 0.397887):.4f}")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
