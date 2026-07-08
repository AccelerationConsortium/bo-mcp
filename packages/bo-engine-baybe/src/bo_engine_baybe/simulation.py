"""BayBE-native Monte-Carlo backtesting helpers (internal benchmarking).

Thin wrapper around :func:`baybe.simulation.simulate_experiment` — driven
once per seeded Monte-Carlo repetition, reproducing the repetition scheme
of ``simulate_scenarios`` without requiring its optional ``xyzpy`` extra —
so the benchmarking scripts can backtest BayBE campaigns with BayBE's own
simulation machinery (lookup-table evaluation, imputation modes,
Monte-Carlo repetitions) instead of a hand-rolled loop — apples-to-apples
against BayBE's published benchmarks.

**Not a REST/MCP surface.** This module is consumed by ``scripts/``
benchmarking only; exposing simulation through the API is explicitly out
of scope (the neutral cross-backend benchmark harness in
``bo_engine.benchmarks`` stays the canonical comparison tool).

Reference: BayBE simulation userguide
(https://emdgroup.github.io/baybe/stable/userguide/simulation.html).
"""

from __future__ import annotations

import pandas as pd
from baybe.simulation import simulate_experiment

from bo_engine.types import OptimizationSpec
from bo_engine_baybe.state import _build_campaign

# Column added to the per-run result frames identifying the Monte-Carlo
# repetition (mirrors the column simulate_scenarios would emit; kept as a
# named constant so consumers do not hardcode the label).
MONTE_CARLO_RUN_COLUMN = "Monte_Carlo_Run"


def simulate_campaign_on_lookup(
    spec: OptimizationSpec,
    lookup: pd.DataFrame,
    *,
    batch_size: int = 1,
    n_doe_iterations: int = 5,
    n_mc_iterations: int = 3,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Backtest the campaign implied by ``spec`` against a lookup table.

    Builds the same BayBE ``Campaign`` the backend would run (including
    any ``backend_options['baybe']`` recommender/surrogate configuration)
    and drives BayBE's core ``simulate_experiment`` once per Monte-Carlo
    repetition (a fresh campaign and derived seed each run — the same
    scheme ``simulate_scenarios`` applies, without requiring the optional
    ``baybe[simulation]`` xyzpy extra). The returned frame is BayBE's
    native result shape (one row per iteration with the running-best
    columns) plus the :data:`MONTE_CARLO_RUN_COLUMN` repetition index.

    Args:
        spec: Neutral optimization spec; converted via the production
            ``_build_campaign`` path so benchmarks measure exactly what
            the backend runs.
        lookup: Dataframe with one column per parameter and per target,
            enumerating the ground-truth measurements.
        batch_size: Recommendations per DOE iteration.
        n_doe_iterations: DOE iterations per Monte-Carlo run.
        n_mc_iterations: Independent Monte-Carlo repetitions.
        random_seed: Base seed; repetition ``k`` runs with
            ``random_seed + k`` for reproducible yet independent runs.

    Returns:
        Concatenated ``simulate_experiment`` result dataframe.
    """
    runs: list[pd.DataFrame] = []
    for mc_run in range(n_mc_iterations):
        seed = random_seed + mc_run if random_seed is not None else None
        result = simulate_experiment(
            _build_campaign(spec),
            lookup,
            batch_size=batch_size,
            n_doe_iterations=n_doe_iterations,
            random_seed=seed,
        )
        result[MONTE_CARLO_RUN_COLUMN] = mc_run
        runs.append(result)
    return pd.concat(runs, ignore_index=True)
