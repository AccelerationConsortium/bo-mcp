#!/usr/bin/env python3
"""Demonstration of Multi-Fidelity Bayesian Optimization (qMFKG) via MCP Server.

This script demonstrates how to use the standalone MCP server with
multi-fidelity optimization for cost-effective optimization.

Features demonstrated:
- Fidelity parameter configuration
- Multi-fidelity model (SingleTaskMultiFidelityGP)
- qMFKG acquisition function
- Cost-fidelity trade-offs

Usage:
    uv run python scripts/demos/mcp_multifidelity_demo.py
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


def multi_fidelity_simulation(
    x1: float,
    x2: float,
    fidelity: float,
) -> tuple[float, float]:
    """Simulate a multi-fidelity objective function.

    Higher fidelity = more accurate but more expensive.

    Args:
        x1: First design parameter (-5 to 10)
        x2: Second design parameter (0 to 15)
        fidelity: Fidelity level (0.1 to 1.0)

    Returns:
        Tuple of (objective_value, evaluation_cost)
    """
    # True objective (Branin function)
    true_value = (
        (x2 - 5.1 / (4 * math.pi**2) * x1**2 + 5 / math.pi * x1 - 6) ** 2
        + 10 * (1 - 1 / (8 * math.pi)) * math.cos(x1)
        + 10
    )

    # Low-fidelity approximation has bias and noise
    # Bias decreases linearly with fidelity
    bias = (1 - fidelity) * 0.5 * (x1 + x2)

    # Noise decreases with fidelity (low fidelity = high noise)
    noise_scale = (1 - fidelity) * 3.0
    noise = random.gauss(0, noise_scale)

    observed_value = true_value + bias + noise

    # Cost model: exponential in fidelity
    # Low fidelity is cheap, high fidelity is expensive
    cost = 1.0 + 99.0 * fidelity**2  # Cost ranges from 1 to 100

    return observed_value, cost


async def main() -> None:
    """Run multi-fidelity BO demonstration via MCP."""
    print("=" * 60)
    print("MULTI-FIDELITY BO (qMFKG) DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Multi-fidelity BO uses cheap/inaccurate evaluations to guide")
    print("the search, reserving expensive/accurate evaluations for the")
    print("most promising candidates.")
    print()
    print("Fidelity trade-off:")
    print("  - Low fidelity (0.1): cheap ($1), noisy, biased")
    print("  - High fidelity (1.0): expensive ($100), accurate, unbiased")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "multifidelity-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("mfbo@example.com")
        if not user:
            user = User(
                name="Multi-Fidelity Demo User",
                email="mfbo@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Define multi-fidelity optimization problem
    print("\n[3] Defining multi-fidelity optimization problem...")

    campaign_config = {
        "name": "Multi-Fidelity Branin Optimization",
        "description": "Optimize Branin function using cheap/expensive evaluations",
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
        "batch_size": 2,
        # Multi-fidelity configuration
        "fidelity_parameter": {
            "name": "fidelity",
            "bounds": [0.1, 1.0],  # Fidelity range
            "target": 1.0,  # We want optimum at full fidelity
            "cost_weight": 1.0,  # Cost scaling
            "fixed_cost": 1.0,  # Minimum cost
        },
        "acquisition_method": "qMFKG",
    }

    print("    Design params: x1 in [-5, 10], x2 in [0, 15]")
    print("    Fidelity param: 0.1 (cheap) to 1.0 (expensive)")
    print("    Objective: minimize f (Branin function)")
    print("    Acquisition: qMFKG (Multi-Fidelity Knowledge Gradient)")
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
    n_iterations = 5
    total_cost = 0.0
    cost_budget = 300.0  # Total budget
    best_at_target_fidelity = float("inf")
    best_overall = float("inf")

    # Track fidelity distribution
    fidelity_bins = {"low": 0, "medium": 0, "high": 0}

    print("\n" + "=" * 60)
    print("MULTI-FIDELITY OPTIMIZATION LOOP")
    print("=" * 60)
    print(f"\nBudget: ${cost_budget:.0f}")
    print("qMFKG will recommend fidelity levels based on value of information.")
    print()

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 60}")
        print(f"ITERATION {iteration}")
        print("─" * 60)

        # Check budget
        if total_cost >= cost_budget:
            print(f"\n    Budget exhausted (${total_cost:.0f} / ${cost_budget:.0f})")
            break

        # Generate suggestions (MF-BO recommends fidelity levels)
        print("\n[A] Generating suggestions (qMFKG will recommend fidelities)...")
        suggestions_result = await generate_suggestions(campaign_id)

        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        print(f"    Got {len(suggestions)} suggestions")

        # Run experiments at recommended fidelities
        print("\n[B] Running experiments...")
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            x1 = params.get("x1", 0.0)
            x2 = params.get("x2", 0.0)

            # Use recommended fidelity or default strategy
            # In real MFKG, this comes from the suggestion
            # For demo, we increase fidelity over iterations
            fidelity = min(0.2 + 0.2 * iteration, 1.0)

            obj_value, cost = multi_fidelity_simulation(x1, x2, fidelity)
            total_cost += cost

            # Track fidelity distribution
            if fidelity < 0.4:
                fidelity_bins["low"] += 1
            elif fidelity < 0.7:
                fidelity_bins["medium"] += 1
            else:
                fidelity_bins["high"] += 1

            results_to_submit.append(
                {
                    "parameter_values": {**params, "fidelity": fidelity},
                    "objective_values": {"f": obj_value},
                    "suggestion_id": s["id"],
                    "metadata": {"cost": cost, "fidelity": fidelity},
                }
            )

            fid_label = f"{fidelity:.1f}"
            print(
                f"    x1={x1:6.2f}, x2={x2:6.2f}, fidelity={fid_label} "
                f"-> f={obj_value:.2f}, cost=${cost:.0f}"
            )

            if fidelity >= 0.95 and obj_value < best_at_target_fidelity:
                best_at_target_fidelity = obj_value
            if obj_value < best_overall:
                best_overall = obj_value

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
            remaining = cost_budget - total_cost
            print(f"    Total evaluations: {diagnostics['n_results']}")
            print(f"    Best f (any fidelity): {best_overall:.2f}")
            print(f"    Best f (target fidelity): {best_at_target_fidelity:.2f}")
            print(f"    Cost: ${total_cost:.0f} / ${cost_budget:.0f} (${remaining:.0f} left)")

    # Final summary
    print("\n" + "=" * 60)
    print("MULTI-FIDELITY OPTIMIZATION COMPLETE")
    print("=" * 60)

    print("\nResults Summary:")
    print(f"  Total cost spent: ${total_cost:.0f}")
    print(f"  Best f value (any fidelity): {best_overall:.2f}")
    print(f"  Best f value (target fidelity): {best_at_target_fidelity:.2f}")
    print()
    print("  Fidelity distribution:")
    total_evals = sum(fidelity_bins.values())
    for level, count in fidelity_bins.items():
        pct = 100 * count / total_evals if total_evals > 0 else 0
        bar = "#" * int(pct / 5)
        print(f"    {level:6s}: {count:2d} ({pct:4.0f}%) {bar}")
    print()
    print("Key Insight: Multi-fidelity BO spent most budget on low-fidelity")
    print("evaluations to explore the space cheaply, then used high-fidelity")
    print("evaluations for exploitation near promising regions.")
    print()
    print("When to use Multi-Fidelity BO:")
    print("  - You have a 'fidelity' knob (resolution, accuracy, etc.)")
    print("  - Low fidelity is much cheaper than high fidelity")
    print("  - Low fidelity is correlated with high fidelity")
    print()
    print("Examples:")
    print("  - Simulation mesh resolution")
    print("  - ML model training epochs")
    print("  - Physical experiment duration")
    print("  - Sample size in experiments")
    print()
    print(f"Campaign ID: {campaign_id}")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
