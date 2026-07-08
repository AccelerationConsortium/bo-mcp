"""Smoke test for the BayBE-native simulation backtesting helper.

Pins that :func:`bo_engine_baybe.simulation.simulate_campaign_on_lookup`
drives BayBE's ``simulate_experiment`` per seeded Monte-Carlo repetition
on a tiny exact lookup problem and returns the native result shape with a
monotone running best. Mirrors the BayBE simulation userguide's
lookup-table backtest
(https://emdgroup.github.io/baybe/stable/userguide/simulation.html).
The backtest completes in a few seconds, so it runs in the fast PR tier
(TESTING.md reserves the ``slow`` marker for >30 s work).
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
from bo_engine_baybe.simulation import MONTE_CARLO_RUN_COLUMN, simulate_campaign_on_lookup

_GRID = [0.0, 1.0, 2.0]


def test_simulation_backtest_runs_on_tiny_lookup() -> None:
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="a", type=ParameterType.DISCRETE, values=_GRID),
            ParameterSpec(name="b", type=ParameterType.DISCRETE, values=_GRID),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )
    lookup = pd.DataFrame(
        [
            {"a": a, "b": b, "y": (a - 1.0) ** 2 + (b - 1.0) ** 2}
            for a, b in itertools.product(_GRID, _GRID)
        ]
    )
    results = simulate_campaign_on_lookup(
        spec,
        lookup,
        n_doe_iterations=3,
        n_mc_iterations=2,
        random_seed=7,
    )
    assert not results.empty
    assert {"Iteration", "y_CumBest"} <= set(results.columns)
    # Running best of a minimization backtest must be non-increasing
    # within each Monte-Carlo run (invariant assertion, not exact values).
    for _, run in results.groupby(MONTE_CARLO_RUN_COLUMN):
        best = run.sort_values("Iteration")["y_CumBest"].to_numpy()
        assert (best[1:] <= best[:-1] + 1e-12).all()
