"""Cross-version reproducibility drift checks.

These tests document the boundaries of the ``random_seed`` reproducibility
contract — a campaign that supplies a master seed gets byte-identical
suggestions only within a fixed (torch version, device,
``torch.use_deterministic_algorithms`` setting) triple. The tests pin two
invariants:

1. **Same-process replay.** Two independent calls to
   :func:`bo_engine.suggestions.generate_next_batch` with the same spec,
   observations, and seed produce identical parameter values. This is the
   contract callers actually rely on inside a single run.
2. **Cross-environment drift bound.** Suggestions hashed against a golden
   file are checked under ``@pytest.mark.nightly`` so a torch minor-version
   bump produces a failing build with a clear diff instead of silently
   shifting the suggestion stream. The golden file is regenerated when the
   torch pin moves.

The drift bound exists because BoTorch's MLL fit, gpytorch kernel reductions
and CUDA atomic-add ordering can each shift between minor releases. Pinning
this in CI makes the bound observable rather than implicit.

References:
    - PyTorch reproducibility docs:
      https://pytorch.org/docs/stable/notes/randomness.html
    - BoTorch determinism notes:
      https://botorch.org/docs/getting_started/
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from bo_engine.suggestions import generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# Pin the reference campaign here so the golden file regeneration is a
# one-line diff if the seed or layout ever changes.
REFERENCE_SEED = 12345
REFERENCE_DIMS = 3
REFERENCE_BATCH = 4
GOLDEN_PATH = Path(__file__).parent / "data" / "reproducibility_golden_suggestions.json"


def _reference_spec_and_observations() -> tuple[OptimizationSpec, list[ObservationData]]:
    """Build the reference campaign whose suggestion stream is pinned."""
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(REFERENCE_DIMS)
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=REFERENCE_BATCH,
        random_seed=REFERENCE_SEED,
    )
    # Deterministic seed-derived observations so the campaign is fully
    # determined by ``REFERENCE_SEED`` (the suggestion stream is otherwise
    # also a function of the observation stream).
    g = torch.Generator().manual_seed(REFERENCE_SEED)
    x = torch.rand(2 * REFERENCE_DIMS + 1, REFERENCE_DIMS, generator=g, dtype=torch.float64)
    y = torch.sin(3.0 * x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.3 * x[:, 2]
    observations = [
        ObservationData(
            parameter_values={f"x{j}": float(x[i, j].item()) for j in range(REFERENCE_DIMS)},
            objective_values={"y": float(y[i].item())},
        )
        for i in range(x.shape[0])
    ]
    return spec, observations


def _materialize_suggestions(spec: OptimizationSpec, obs: list[ObservationData]) -> list[dict]:
    """Run ``generate_next_batch`` and reduce to a JSON-stable shape."""
    suggestions, _ = generate_next_batch(spec, obs, batch_size=REFERENCE_BATCH, iteration=1)
    return [
        {
            "parameter_values": sr.parameter_values,
            "batch_index": sr.batch_index,
        }
        for sr in suggestions
    ]


class TestSameProcessReplay:
    """Same seed, same observations, same process → identical suggestions.

    The contract callers actually rely on for retry behavior and cache
    replay. Independent of torch version.
    """

    def test_two_calls_produce_identical_parameter_values(self) -> None:
        spec, observations = _reference_spec_and_observations()

        first = _materialize_suggestions(spec, observations)
        second = _materialize_suggestions(spec, observations)

        assert first == second, (
            "Same-process replay diverged. ``random_seed`` is supposed to "
            "guarantee identical suggestions within a fixed environment; "
            "investigate any new global RNG mutation or non-deterministic "
            "code path."
        )


@pytest.mark.nightly
class TestCrossVersionDrift:
    """Suggestions vs. a pinned golden file. Fires on torch/gpytorch bumps.

    Marked ``nightly`` because the comparison is environment-dependent and
    fails as soon as the production torch pin changes. The check is
    intentionally strict (round-tripped float equality after JSON encode)
    so any drift is visible in the failing diff.
    """

    def test_reference_campaign_matches_golden(self) -> None:
        if not GOLDEN_PATH.exists():
            pytest.skip(
                "Golden file not generated yet. Run "
                "``scripts/regenerate_reproducibility_golden.py`` after "
                "pinning a new torch version."
            )

        spec, observations = _reference_spec_and_observations()
        actual = _materialize_suggestions(spec, observations)
        expected = json.loads(GOLDEN_PATH.read_text())

        assert actual == expected, (
            "Reproducibility drift detected against the golden file. This "
            "typically means a torch / gpytorch / botorch minor version "
            "moved. Update the pin, regenerate "
            "``reproducibility_golden_suggestions.json`` deliberately, and "
            "document the bump in the PR description."
        )
