#!/usr/bin/env python
"""Backtest the BayBE backend with BayBE's native simulation utilities.

Complements ``scripts/benchmark_backends.py`` (the neutral cross-backend
harness): this script drives ``baybe.simulation.simulate_experiment``
through :func:`bo_engine_baybe.simulation.simulate_campaign_on_lookup`,
so the BayBE backend can be compared apples-to-apples against BayBE's
published Monte-Carlo backtests. Clearly BayBE-only — it is not the
canonical cross-backend comparison tool.

Usage:
    uv run python scripts/benchmark_baybe_simulation.py
"""

from __future__ import annotations

import itertools

import pandas as pd

from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.simulation import simulate_campaign_on_lookup

# Small discrete demo problem: minimize y = (a - 1)^2 + (b - 2)^2 over a
# 5x5 grid — enumerable, so the lookup table is exact.
GRID_VALUES = [0.0, 1.0, 2.0, 3.0, 4.0]
N_DOE_ITERATIONS = 5
N_MC_ITERATIONS = 3
RANDOM_SEED = 1337


def build_demo_spec() -> OptimizationSpec:
    """The neutral spec whose BayBE campaign gets backtested."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=GRID_VALUES),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=GRID_VALUES),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def build_demo_lookup() -> pd.DataFrame:
    """Exact lookup table for the demo objective."""
    rows = [
        {"a": a, "b": b, "y": (a - 1.0) ** 2 + (b - 2.0) ** 2}
        for a, b in itertools.product(GRID_VALUES, GRID_VALUES)
    ]
    return pd.DataFrame(rows)


def main() -> None:
    """Run the demo backtest and print the per-iteration running best."""
    results = simulate_campaign_on_lookup(
        build_demo_spec(),
        build_demo_lookup(),
        n_doe_iterations=N_DOE_ITERATIONS,
        n_mc_iterations=N_MC_ITERATIONS,
        random_seed=RANDOM_SEED,
    )
    summary = results.groupby("Iteration")["y_CumBest"].mean()
    print("Mean running best per iteration (lower is better):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
