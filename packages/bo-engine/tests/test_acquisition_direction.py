"""Acquisition-direction regression tests.

BoTorch's qLog* acquisition family (``qLogEI`` / ``qLogNEI``) carries no
direction flag and always treats **larger** sampled values as improvements
— see ``botorch.acquisition.logei`` and the closed-loop tutorial
(https://botorch.org/docs/tutorials/closed_loop_botorch_only/), which
negates minimization problems before optimization. The engine therefore
adopts maximization form internally and negates minimize objectives once
at the data boundary (see :mod:`bo_engine.types`).

These tests pin that direction end to end on a deterministic 1-D
quadratic: the grid argmax of the acquisition surface must lie in the
optimum's basin for *both* user-facing directions. An inverted pipeline
fails spectacularly here — the acquisition surface peaks at the opposite
end of the domain — so no optimizer randomness or statistical tolerance
is involved.

Reference for the test design (acquisition-surface inspection on a known
1-D function): https://botorch.org/docs/tutorials/fit_model_with_torch_optimizer/
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bo_engine.acquisition import create_single_objective_acquisition
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.suggestions import generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

OPTIMUM_X = 0.25
# The failure mode under a sign inversion is an argmax at the opposite end
# of the domain (x = 1.0), so a generous basin still separates the two
# outcomes unambiguously.
BASIN_RADIUS = 0.15
N_TRAIN = 12
N_GRID = 101


def _acquisition_argmax(train_y_bo: torch.Tensor) -> float:
    """Fit a GP on maximization-form targets and return the acq-surface argmax."""
    train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    model = create_and_fit_single_task_model(train_x, train_y_bo, bounds)
    acqf = create_single_objective_acquisition(
        model=model,
        train_x=train_x,
        train_y=train_y_bo,
        maximize=True,
        use_noisy=True,
    )
    grid = torch.linspace(0, 1, N_GRID, dtype=torch.double).reshape(-1, 1, 1)
    with torch.no_grad():
        values = acqf(grid).squeeze()
    return float(grid.squeeze()[values.argmax()].item())


@pytest.mark.smoke
class TestAcquisitionSurfaceDirection:
    """The acquisition surface must peak in the optimum's basin."""

    def test_minimize_objective_peaks_at_minimum(self) -> None:
        """minimize=True on f(x) = (x - 0.25)²: acq argmax near x = 0.25.

        The pipeline negates minimize objectives into maximization form,
        so the factory receives ``-f``.
        """
        torch.manual_seed(0)
        train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
        train_y = (train_x - OPTIMUM_X) ** 2
        argmax_x = _acquisition_argmax(-train_y)
        assert abs(argmax_x - OPTIMUM_X) <= BASIN_RADIUS, (
            f"Acquisition surface peaks at x={argmax_x:.3f}, expected within "
            f"{BASIN_RADIUS} of the minimum x={OPTIMUM_X} — the pipeline is "
            "optimizing in the wrong direction."
        )

    def test_maximize_objective_peaks_at_maximum(self) -> None:
        """minimize=False on f(x) = -(x - 0.25)²: acq argmax near x = 0.25.

        Maximize objectives pass through the boundary un-negated.
        """
        torch.manual_seed(0)
        train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
        train_y = -((train_x - OPTIMUM_X) ** 2)
        argmax_x = _acquisition_argmax(train_y)
        assert abs(argmax_x - OPTIMUM_X) <= BASIN_RADIUS, (
            f"Acquisition surface peaks at x={argmax_x:.3f}, expected within "
            f"{BASIN_RADIUS} of the maximum x={OPTIMUM_X} — the pipeline is "
            "optimizing in the wrong direction."
        )


@pytest.mark.smoke
class TestPipelineDirection:
    """End-to-end direction check through ``generate_next_batch``.

    Guards the full wiring (boundary negation → model fit → acquisition →
    provenance): with the basin already densely observed, the suggested
    point and its predicted objective must be near the optimum, not the
    anti-optimum.
    """

    def _run_one_bo_step(self, *, minimize: bool) -> tuple[float, float]:
        """Observe a dense grid of the quadratic, return (suggested_x, predicted_y)."""
        sign = 1.0 if minimize else -1.0
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[ObjectiveSpec(name="y", minimize=minimize)],
            batch_size=1,
        )
        xs = np.linspace(0.0, 1.0, N_TRAIN)
        observations = [
            ObservationData(
                parameter_values={"x": float(x)},
                objective_values={"y": float(sign * (x - OPTIMUM_X) ** 2)},
            )
            for x in xs
        ]
        rng = np.random.default_rng(0)
        suggestions, _ = generate_next_batch(spec, observations, iteration=1, rng=rng)
        suggestion = suggestions[0]
        assert suggestion.predicted_objectives is not None
        return suggestion.parameter_values["x"], suggestion.predicted_objectives["y"]

    def test_minimize_suggests_near_minimum(self) -> None:
        suggested_x, predicted_y = self._run_one_bo_step(minimize=True)
        # Anti-optimization drives suggestions toward x = 1.0 (f ≈ 0.56);
        # working BO exploits the basin around x = 0.25.
        assert abs(suggested_x - OPTIMUM_X) <= 2 * BASIN_RADIUS
        # Provenance must report the prediction on the raw user scale: near
        # the basin the objective is ≈ 0, far away it approaches 0.56.
        assert predicted_y == pytest.approx((suggested_x - OPTIMUM_X) ** 2, abs=0.05)

    def test_maximize_suggests_near_maximum(self) -> None:
        suggested_x, predicted_y = self._run_one_bo_step(minimize=False)
        assert abs(suggested_x - OPTIMUM_X) <= 2 * BASIN_RADIUS
        assert predicted_y == pytest.approx(-((suggested_x - OPTIMUM_X) ** 2), abs=0.05)
