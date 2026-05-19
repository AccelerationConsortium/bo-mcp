"""Tests for the configurable Kumaraswamy warp prior on input warping.

The ``ChainedInputTransform`` order is ``normalize → warp``: ``Normalize``
brings raw inputs into ``[0, 1]`` and ``Warp`` reshapes that bounded space
to absorb non-stationarity. The LogNormal prior on each Kumaraswamy
concentration therefore assumes a unit-cube input domain. We expose the
``loc`` / ``scale`` of that prior so non-stationary objectives with
empirically calibrated warps (e.g. Branin-on-log, Gramacy/Lee periodic)
can pass their own values instead of the BoTorch stock defaults.

References:
    - Snoek et al., "Input Warping for Bayesian Optimization of
      Non-Stationary Functions", ICML 2014 — original Kumaraswamy input
      warping formulation.
    - BoTorch ``Warp`` source (LogNormalPrior(loc=0, scale=0.75) default).
"""

from __future__ import annotations

import torch
from botorch.models.transforms.input import ChainedInputTransform, Warp
from gpytorch.priors.torch_priors import LogNormalPrior

from bo_engine.constants import WARP_PRIOR_LOC, WARP_PRIOR_SCALE
from bo_engine.models import create_input_transform

N_DIMS = 3


def _make_bounds() -> torch.Tensor:
    return torch.stack(
        [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
    )


class TestDefaultPrior:
    """Stock prior must match the documented BoTorch tutorial values."""

    def test_default_prior_matches_constants(self) -> None:
        transform = create_input_transform(N_DIMS, _make_bounds(), use_input_warping=True)
        assert isinstance(transform, ChainedInputTransform)
        warp = next(t for t in transform.values() if isinstance(t, Warp))

        prior = warp.concentration1_prior
        assert isinstance(prior, LogNormalPrior)
        assert float(prior.loc.item()) == WARP_PRIOR_LOC
        assert float(prior.scale.item()) == WARP_PRIOR_SCALE


class TestCustomPrior:
    """Custom loc/scale must flow through to both concentration priors."""

    def test_custom_loc_and_scale_applied(self) -> None:
        transform = create_input_transform(
            N_DIMS,
            _make_bounds(),
            use_input_warping=True,
            warp_prior_loc=0.5,
            warp_prior_scale=0.25,
        )
        assert isinstance(transform, ChainedInputTransform)
        warp = next(t for t in transform.values() if isinstance(t, Warp))
        for attr in ("concentration1_prior", "concentration0_prior"):
            prior = getattr(warp, attr)
            assert isinstance(prior, LogNormalPrior)
            assert float(prior.loc.item()) == 0.5
            assert float(prior.scale.item()) == 0.25


class TestNoWarpReturnsNormalizeOnly:
    """``use_input_warping=False`` must short-circuit before constructing Warp."""

    def test_no_warp_short_circuit(self) -> None:
        transform = create_input_transform(N_DIMS, _make_bounds(), use_input_warping=False)
        # Without warping the returned transform is the bare Normalize.
        assert not isinstance(transform, ChainedInputTransform)
