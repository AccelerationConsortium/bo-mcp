#!/usr/bin/env python3
"""Demonstration of Mixture/Sum Constraints via MCP Server.

This script demonstrates how to use the standalone MCP server with
sum constraints (mixture formulations where components sum to 1).

Features demonstrated:
- Sum constraints (parameters must sum to a fixed value)
- Mixture formulation optimization
- Reparameterization for constraint satisfaction

Usage:
    uv run python scripts/demos/mcp_mixture_constraints_demo.py
"""

import asyncio
import hashlib
import math
import random

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.submit_results import submit_results


def alloy_strength(fe: float, ni: float, cr: float) -> float:
    """Simulate alloy strength based on composition.

    Args:
        fe: Iron fraction (0 to 1)
        ni: Nickel fraction (0 to 1)
        cr: Chromium fraction (0 to 1)

    Constraint: fe + ni + cr = 1 (must sum to 1)

    Returns:
        Tensile strength in MPa (to be maximized)
    """
    # Validate constraint
    total = fe + ni + cr
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Composition must sum to 1, got {total}")

    # Base strength model (simplified metallurgy)
    # Iron provides base strength
    base_strength = 200 + 300 * fe

    # Nickel increases toughness (optimal around 20%)
    ni_effect = 100 * math.exp(-((ni - 0.2) ** 2) / 0.02)

    # Chromium improves corrosion resistance and hardness
    cr_effect = 150 * math.sqrt(cr) if cr > 0 else 0

    # Interaction effects (real metallurgy is more complex)
    interaction = 50 * fe * ni * cr * 10  # Three-way interaction

    strength = base_strength + ni_effect + cr_effect + interaction
    return strength + random.gauss(0, 10)


def paint_coverage(pigment: float, binder: float, solvent: float, additive: float) -> float:
    """Simulate paint coverage based on formulation.

    Args:
        pigment: Pigment fraction
        binder: Binder/resin fraction
        solvent: Solvent fraction
        additive: Additive fraction

    Constraint: pigment + binder + solvent + additive = 1

    Returns:
        Coverage in m2/L (to be maximized)
    """
    total = pigment + binder + solvent + additive
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"Formulation must sum to 1, got {total}")

    # Pigment provides opacity
    pigment_effect = 8 * pigment

    # Binder affects adhesion and film formation
    binder_effect = 3 * binder

    # Too much solvent reduces coverage
    solvent_penalty = -2 * max(0, solvent - 0.4)

    # Additives improve flow and leveling
    additive_effect = 5 * additive if additive < 0.1 else 0.5  # Diminishing returns

    coverage = pigment_effect + binder_effect + solvent_penalty + additive_effect + 5
    return max(0, coverage + random.gauss(0, 0.3))


async def run_alloy_demo(owner_id: str) -> None:
    """Run alloy composition optimization with 3-component mixture."""
    print("\n" + "=" * 60)
    print("PART 1: ALLOY COMPOSITION (3-Component Mixture)")
    print("=" * 60)
    print()
    print("Optimizing Fe-Ni-Cr alloy composition for tensile strength.")
    print("Constraint: Fe + Ni + Cr = 1 (fractions must sum to 1)")
    print()

    campaign_config = {
        "name": "Alloy Composition Optimization",
        "description": "Optimize Fe-Ni-Cr alloy for maximum strength",
        "parameters": [
            {
                "name": "Fe",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Iron fraction",
            },
            {
                "name": "Ni",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Nickel fraction",
            },
            {
                "name": "Cr",
                "type": "continuous",
                "bounds": [0.0, 1.0],
                "description": "Chromium fraction",
            },
        ],
        "objectives": [
            {"name": "strength", "direction": "maximize", "unit": "MPa"},
        ],
        "constraints": [
            {
                "type": "sum_equals",
                "parameters": ["Fe", "Ni", "Cr"],
                "value": 1.0,
            }
        ],
        "batch_size": 3,
    }

    print("[A] Creating alloy optimization campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization
    n_iterations = 4
    best_strength = 0.0
    best_comp: dict[str, float] = {}

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 40}")
        print(f"Iteration {iteration}")
        print("─" * 40)

        suggestions_result = await generate_suggestions(campaign_id)
        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            fe = params.get("Fe", 0.33)
            ni = params.get("Ni", 0.33)
            cr = params.get("Cr", 0.34)

            # Verify constraint
            total = fe + ni + cr
            constraint_satisfied = abs(total - 1.0) < 0.01

            strength = alloy_strength(fe, ni, cr)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"strength": strength},
                    "suggestion_id": s["id"],
                }
            )

            status = "OK" if constraint_satisfied else f"SUM={total:.3f}"
            print(f"    Fe={fe:.3f}, Ni={ni:.3f}, Cr={cr:.3f} [{status}] -> {strength:.0f} MPa")

            if strength > best_strength:
                best_strength = strength
                best_comp = params.copy()

        await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

    print("\nBest Alloy Composition:")
    print(f"  Fe: {best_comp.get('Fe', 0) * 100:.1f}%")
    print(f"  Ni: {best_comp.get('Ni', 0) * 100:.1f}%")
    print(f"  Cr: {best_comp.get('Cr', 0) * 100:.1f}%")
    print(f"  Strength: {best_strength:.0f} MPa")


async def run_paint_demo(owner_id: str) -> None:
    """Run paint formulation optimization with 4-component mixture."""
    print("\n" + "=" * 60)
    print("PART 2: PAINT FORMULATION (4-Component Mixture)")
    print("=" * 60)
    print()
    print("Optimizing paint formulation for coverage.")
    print("Constraint: pigment + binder + solvent + additive = 1")
    print()

    campaign_config = {
        "name": "Paint Formulation Optimization",
        "description": "Optimize paint composition for coverage",
        "parameters": [
            {
                "name": "pigment",
                "type": "continuous",
                "bounds": [0.1, 0.6],
                "description": "Pigment fraction",
            },
            {
                "name": "binder",
                "type": "continuous",
                "bounds": [0.1, 0.5],
                "description": "Binder/resin fraction",
            },
            {
                "name": "solvent",
                "type": "continuous",
                "bounds": [0.1, 0.5],
                "description": "Solvent fraction",
            },
            {
                "name": "additive",
                "type": "continuous",
                "bounds": [0.01, 0.15],
                "description": "Additive fraction",
            },
        ],
        "objectives": [
            {"name": "coverage", "direction": "maximize", "unit": "m2/L"},
        ],
        "constraints": [
            {
                "type": "sum_equals",
                "parameters": ["pigment", "binder", "solvent", "additive"],
                "value": 1.0,
            }
        ],
        "batch_size": 3,
    }

    print("[A] Creating paint formulation campaign...")
    result = await create_campaign(campaign_config, owner_id)

    if not result["success"]:
        print(f"    ERROR: {result['errors']}")
        return

    campaign_id = result["campaign_id"]
    print(f"    Campaign created: {campaign_id}")

    # Run optimization
    n_iterations = 4
    best_coverage = 0.0
    best_form: dict[str, float] = {}

    for iteration in range(1, n_iterations + 1):
        print(f"\n{'─' * 40}")
        print(f"Iteration {iteration}")
        print("─" * 40)

        suggestions_result = await generate_suggestions(campaign_id)
        if not suggestions_result["success"]:
            print(f"    ERROR: {suggestions_result['errors']}")
            break

        suggestions = suggestions_result["suggestions"]
        results_to_submit = []

        for s in suggestions:
            params = s["parameter_values"]
            pig = params.get("pigment", 0.3)
            bind = params.get("binder", 0.3)
            solv = params.get("solvent", 0.3)
            add = params.get("additive", 0.1)

            total = pig + bind + solv + add
            coverage = paint_coverage(pig, bind, solv, add)

            results_to_submit.append(
                {
                    "parameter_values": params,
                    "objective_values": {"coverage": coverage},
                    "suggestion_id": s["id"],
                }
            )

            print(
                f"    P={pig:.2f}, B={bind:.2f}, S={solv:.2f}, A={add:.2f} "
                f"[sum={total:.3f}] -> {coverage:.2f} m2/L"
            )

            if coverage > best_coverage:
                best_coverage = coverage
                best_form = params.copy()

        await submit_results(
            campaign_id=campaign_id,
            results=results_to_submit,
            submitted_by=owner_id,
            source="api",
        )

    print("\nBest Paint Formulation:")
    print(f"  Pigment: {best_form.get('pigment', 0) * 100:.1f}%")
    print(f"  Binder: {best_form.get('binder', 0) * 100:.1f}%")
    print(f"  Solvent: {best_form.get('solvent', 0) * 100:.1f}%")
    print(f"  Additive: {best_form.get('additive', 0) * 100:.1f}%")
    print(f"  Coverage: {best_coverage:.2f} m2/L")


async def main() -> None:
    """Run mixture constraints demonstration via MCP."""
    print("=" * 60)
    print("MIXTURE/SUM CONSTRAINTS DEMO (MCP Standalone)")
    print("=" * 60)
    print()
    print("Sum constraints ensure parameters sum to a fixed value.")
    print("Common in formulation optimization (alloys, paints, foods, etc.)")
    print()

    # Initialize database
    print("[1] Initializing database...")
    await init_database()

    # Setup user
    print("[2] Setting up user...")
    api_key = "mixture-demo-key"
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email("mixture@example.com")
        if not user:
            user = User(
                name="Mixture Demo User",
                email="mixture@example.com",
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
            print(f"    Created user: {user.id}")
        else:
            print(f"    Using existing user: {user.id}")

    owner_id = str(user.id)

    # Run demonstrations
    await run_alloy_demo(owner_id)
    await run_paint_demo(owner_id)

    # Summary
    print("\n" + "=" * 60)
    print("MIXTURE CONSTRAINTS DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("Key Insights:")
    print("  1. Sum constraints enforce composition = 100%")
    print("  2. BO uses reparameterization to always satisfy constraints")
    print("  3. All suggestions automatically satisfy sum = 1")
    print("  4. Works with 3+ components in the mixture")


if __name__ == "__main__":
    random.seed(42)
    asyncio.run(main())
