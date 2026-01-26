#!/usr/bin/env python3
"""Demonstration of Transfer Learning (RGPE) via MCP Server.

Transfer learning allows leveraging knowledge from prior optimization
campaigns to accelerate optimization on a new but related task.

This script demonstrates:
1. Creating a "prior" campaign with completed results
2. Creating a new campaign that transfers from the prior
3. How transfer learning improves early-stage optimization

Usage:
    uv run python scripts/transfer_learning_mcp_demo.py
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


def prior_task_objective(x1: float, x2: float) -> float:
    """Objective function for the prior task.

    A shifted Branin function representing a previously optimized system.
    """
    # Shifted Branin
    return (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
        + random.gauss(0, 0.1)
    )


def new_task_objective(x1: float, x2: float) -> float:
    """Objective function for the new task.

    Similar to prior but with slight modifications - represents a related
    but not identical optimization problem (e.g., new product variant).
    """
    # Similar to prior but with small offset and scale change
    prior_val = prior_task_objective(x1, x2)
    # Add a small perturbation to simulate a related but different task
    offset = 0.5 * math.sin(x1 * 0.5) + 0.3 * math.cos(x2 * 0.3)
    return prior_val * 0.95 + offset + random.gauss(0, 0.1)


async def create_prior_campaign(owner_id: str) -> str | None:
    """Create and populate a prior campaign with historical data."""
    print("\n[A] Creating prior campaign (historical optimization)...")

    prior_config = {
        "name": "Prior Optimization Task",
        "description": "Historical optimization to transfer from",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
                "description": "First parameter",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
                "description": "Second parameter",
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 3,
    }

    result = await create_campaign(prior_config, owner_id)
    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return None

    campaign_id = result["campaign_id"]
    print(f"    Prior campaign created: {campaign_id}")

    # Populate with historical data (simulating past optimization)
    print("    Populating with historical data...")
    historical_points = [
        (-3.0, 2.0),
        (0.0, 7.5),
        (3.0, 3.0),
        (6.0, 10.0),
        (9.0, 5.0),
        (3.14, 2.27),  # Near optimum
        (9.42, 2.47),  # Near optimum
    ]

    for _i, (x1, x2) in enumerate(historical_points):
        # Generate suggestions first
        suggestions_result = await generate_suggestions(campaign_id, batch_size=1)
        if not suggestions_result["success"]:
            continue

        s = suggestions_result["suggestions"][0]
        obj_value = prior_task_objective(x1, x2)

        await submit_results(
            campaign_id=campaign_id,
            results=[
                {
                    "parameter_values": {"x1": x1, "x2": x2},
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                }
            ],
            submitted_by=owner_id,
            source="api",
        )

    print(f"    Added {len(historical_points)} historical observations")
    return campaign_id


async def main():
    """Run transfer learning demonstration via MCP."""
    print("=" * 60)
    print("TRANSFER LEARNING (RGPE) DEMONSTRATION (MCP Standalone)")
    print("=" * 60)
    print()
    print("RGPE (Rank-weighted GP Ensemble) transfers knowledge from")
    print("prior optimization campaigns to accelerate new optimization.")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "transfer-learning-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("transfer@example.com")
        if not user:
            user = User(
                name="Transfer Learning Demo User",
                email="transfer@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Create prior campaign with historical data
    print("\n" + "=" * 60)
    print("PHASE 1: CREATE PRIOR CAMPAIGN")
    print("=" * 60)

    prior_campaign_id = await create_prior_campaign(owner_id)
    if prior_campaign_id is None:
        print("Failed to create prior campaign")
        return

    # Get diagnostics from prior campaign
    prior_diagnostics = await get_diagnostics(prior_campaign_id)
    if prior_diagnostics["success"]:
        print("\nPrior campaign summary:")
        print(f"  Total observations: {prior_diagnostics['n_results']}")
        print(f"  Health status: {prior_diagnostics.get('health_status', 'N/A')}")

    # Create new campaign with transfer learning
    print("\n" + "=" * 60)
    print("PHASE 2: CREATE NEW CAMPAIGN WITH TRANSFER LEARNING")
    print("=" * 60)

    new_config = {
        "name": "New Optimization Task (with Transfer)",
        "description": "New optimization leveraging prior knowledge",
        "parameters": [
            {
                "name": "x1",
                "type": "continuous",
                "bounds": [-5.0, 10.0],
                "description": "First parameter",
            },
            {
                "name": "x2",
                "type": "continuous",
                "bounds": [0.0, 15.0],
                "description": "Second parameter",
            },
        ],
        "objectives": [
            {"name": "f", "direction": "minimize", "unit": "value"},
        ],
        "batch_size": 2,
        # v2.0: Transfer learning configuration
        "transfer_learning": {
            "prior_campaign_ids": [prior_campaign_id],
            "num_ranking_samples": 256,
        },
    }

    validation = await validate_intake(new_config)
    if not validation["valid"]:
        print(f"    ERROR: {validation['errors']}")
        return

    result = await create_campaign(new_config, owner_id)
    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    new_campaign_id = result["campaign_id"]
    print(f"    New campaign created: {new_campaign_id}")
    print(f"    Transferring from: {prior_campaign_id}")

    # Run optimization on new task
    print("\n" + "=" * 60)
    print("PHASE 3: OPTIMIZE NEW TASK WITH TRANSFER")
    print("=" * 60)

    n_iterations = 3
    best_value = float("inf")

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Generate suggestions (using RGPE with prior knowledge)
        print("\n[A] Generating suggestions (with transfer learning)...")
        suggestions_result = await generate_suggestions(new_campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Check if method selection mentions transfer
        if "method_selection" in suggestions_result:
            ms = suggestions_result["method_selection"]
            print(f"    Model: {ms['model_type']}")
            if "transfer" in ms.get("explanation", "").lower():
                print("    ✓ Transfer learning active!")

        # Run experiments on new task
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)

            obj_value = new_task_objective(x1, x2)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                }
            )

            print(f"    x1={x1:.2f}, x2={x2:.2f} -> f={obj_value:.4f}")

            if obj_value < best_value:
                best_value = obj_value

        # Submit results
        print("\n[C] Submitting results...")
        submit_result = await submit_results(
            campaign_id=new_campaign_id,
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
        diagnostics = await get_diagnostics(new_campaign_id)

        if diagnostics["success"]:
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best f value: {best_value:.4f}")
            print(f"    Health status: {diagnostics.get('health_status', 'N/A')}")

    # Final summary
    print("\n" + "=" * 60)
    print("TRANSFER LEARNING DEMONSTRATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Prior campaign ID: {prior_campaign_id}")
    print(f"  New campaign ID: {new_campaign_id}")
    print(f"  Best value found: {best_value:.4f}")
    print()
    print("Key Insight: Transfer learning used prior optimization knowledge")
    print("to make better suggestions in early iterations, reducing the")
    print("number of evaluations needed to find good solutions.")
    print()
    print("In practice, transfer learning is most beneficial when:")
    print("  1. Prior tasks are similar (same parameters, related objectives)")
    print("  2. New task has limited evaluation budget")
    print("  3. Prior tasks have sufficient data")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
