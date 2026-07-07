"""Direction regression tests for the SAASBO suggestion entry point.

``generate_saasbo_suggestions`` builds ``qLogExpectedImprovement``, which —
like the whole qLog* family — always maximizes (see
``botorch.acquisition.logei`` and
https://botorch.org/docs/tutorials/closed_loop_botorch_only/). The entry
point therefore negates minimize objectives into the engine's
maximization form at the data boundary (see :mod:`bo_engine.types`).

The fully Bayesian SAAS fit (NUTS) is far too slow for the PR gate and is
irrelevant to the sign logic, so these tests swap the model factory for a
plain fitted ``SingleTaskGP`` and stub the lengthscale-importance report;
what remains under test is exactly the direction handling: boundary
negation, ``best_f`` selection, and the acquisition optimization.
"""

from __future__ import annotations

import pytest
import torch
from botorch.models import SingleTaskGP

import bo_engine.saasbo as saasbo_module
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.saasbo import generate_saasbo_suggestions

OPTIMUM_X = 0.25
# A sign inversion drives the suggestion to the opposite end of the
# domain (x = 1.0), so a generous basin separates the outcomes cleanly.
BASIN_RADIUS = 0.3
N_TRAIN = 12


@pytest.fixture
def plain_gp_saasbo(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the NUTS-fitted SAAS model with a plain fitted GP."""
    default_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

    def fake_create_and_fit(
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        config: object = None,  # noqa: ARG001
        bounds: torch.Tensor | None = None,
    ) -> SingleTaskGP:
        return create_and_fit_single_task_model(
            train_x, train_y, default_bounds if bounds is None else bounds
        )

    monkeypatch.setattr(saasbo_module, "create_and_fit_saasbo_model", fake_create_and_fit)
    monkeypatch.setattr(
        saasbo_module, "compute_saasbo_importance_report", lambda _model, _names=None: []
    )


@pytest.mark.usefixtures("plain_gp_saasbo")
class TestSaasboSuggestionDirection:
    """Suggestions must land in the optimum's basin for both directions."""

    def _suggest(self, train_y: torch.Tensor, *, minimize: bool) -> float:
        torch.manual_seed(0)
        train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        candidates, _acq_values, metadata = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=1, minimize=minimize
        )
        assert metadata["minimize"] is minimize
        return float(candidates[0, 0].item())

    def test_minimize_suggests_near_minimum(self) -> None:
        """minimize=True on f(x) = (x - 0.25)²: suggestion near x = 0.25."""
        train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
        train_y = (train_x - OPTIMUM_X) ** 2
        suggested_x = self._suggest(train_y, minimize=True)
        assert abs(suggested_x - OPTIMUM_X) <= BASIN_RADIUS, (
            f"SAASBO suggested x={suggested_x:.3f}, expected within "
            f"{BASIN_RADIUS} of the minimum x={OPTIMUM_X} — the entry point "
            "is optimizing in the wrong direction."
        )

    def test_maximize_suggests_near_maximum(self) -> None:
        """minimize=False on f(x) = -(x - 0.25)²: suggestion near x = 0.25."""
        train_x = torch.linspace(0, 1, N_TRAIN, dtype=torch.double).unsqueeze(-1)
        train_y = -((train_x - OPTIMUM_X) ** 2)
        suggested_x = self._suggest(train_y, minimize=False)
        assert abs(suggested_x - OPTIMUM_X) <= BASIN_RADIUS, (
            f"SAASBO suggested x={suggested_x:.3f}, expected within "
            f"{BASIN_RADIUS} of the maximum x={OPTIMUM_X} — the entry point "
            "is optimizing in the wrong direction."
        )
