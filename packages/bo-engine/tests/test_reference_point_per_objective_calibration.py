"""Per-objective reference-point calibration — the existing relative floor.

The audit's concern (Phase H 8.30) was that multi-objective hypervolume
collapses when objectives have heterogeneous magnitudes. The existing
formula
``window_j = max(range_j, |worst_j| * RELATIVE_TOLERANCE, MIN_OBJECTIVE_RANGE)``
already provides per-objective scaling via the relative floor: a small-
spread objective with a non-trivial absolute scale (e.g. purity around
``worst = 1.0`` with range = 0.002) gets a margin proportional to its own
scale instead of collapsing to ``MIN_OBJECTIVE_RANGE``.

These tests pin the load-bearing behaviour: the relative floor must be
strictly larger than ``MIN_OBJECTIVE_RANGE`` whenever the objective has a
non-trivial absolute scale, and rescaling one objective rescales only
its own margin.

References:
    - Ishibuchi et al., "Reference Point Specification in Inverted
      Generational Distance for EMO" (2015).
    - Knowles, "ParEGO" IEEE TEC 2006 — per-objective normalization keeps
      scalarization symmetric across heterogeneous magnitudes.
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.constants import MIN_OBJECTIVE_RANGE, REFERENCE_POINT_RELATIVE_TOLERANCE
from bo_engine.reference_point import (
    ReferencePointConfig,
    ReferencePointStrategy,
    get_reference_point_dynamic,
)


class TestRelativeFloorRescuesTinySpread:
    """A small-spread non-zero-scale objective should get a proportional margin.

    Without the relative floor, an objective whose range collapses (e.g. a
    near-constant phase of the campaign) would degrade to
    ``MIN_OBJECTIVE_RANGE``; with the relative floor, the margin tracks
    the objective's own absolute scale.
    """

    def test_tiny_spread_uses_relative_floor_not_absolute_floor(self) -> None:
        # Range of 1e-9 (well below MIN_OBJECTIVE_RANGE=1e-6) but
        # |worst|=1.0 → relative floor = 0.01 dominates the window.
        purity = torch.tensor([1.0, 1.0 + 1e-9, 1.0 - 1e-9], dtype=torch.float64)
        train_y = purity.unsqueeze(-1)

        config = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)
        ref_point, _ = get_reference_point_dynamic(train_y, config=config)

        worst = train_y.max(dim=0).values
        margin = (ref_point - worst).item()
        # The relative floor is |worst| * REFERENCE_POINT_RELATIVE_TOLERANCE = 0.01.
        # The margin is config.margin * window = 0.1 * 0.01 = 0.001, NOT
        # 0.1 * 1e-6 = 1e-7 (the absolute-floor branch).
        expected = config.margin * worst.abs().item() * REFERENCE_POINT_RELATIVE_TOLERANCE
        assert margin == pytest.approx(expected, rel=1e-9)
        # Sanity: this is strictly larger than what the absolute-floor
        # branch would have produced.
        assert margin > config.margin * MIN_OBJECTIVE_RANGE * 100


class TestRescalingInvariance:
    """Rescaling a single objective by a constant must rescale only its margin."""

    def test_rescaling_one_objective_rescales_only_its_margin(self) -> None:
        y0 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
        y1 = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0], dtype=torch.float64)
        baseline = torch.stack([y0, y1], dim=-1)
        rescaled = torch.stack([y0, y1 * 1000.0], dim=-1)

        config = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)
        baseline_ref, _ = get_reference_point_dynamic(baseline, config=config)
        rescaled_ref, _ = get_reference_point_dynamic(rescaled, config=config)

        baseline_margin = baseline_ref - baseline.max(dim=0).values
        rescaled_margin = rescaled_ref - rescaled.max(dim=0).values

        # Objective 0 was not touched — its margin must be unchanged.
        assert torch.isclose(baseline_margin[0], rescaled_margin[0], atol=1e-12)
        # Objective 1 was rescaled by 1000 — its margin should be roughly 1000x.
        ratio = float(rescaled_margin[1] / baseline_margin[1])
        assert 100 <= ratio <= 10000


class TestHeterogeneousMagnitudesBothVisible:
    """Cost ~[0,1000] and purity ~[0.99, 1.00]: both margins are strictly positive."""

    def test_both_objectives_keep_a_meaningful_margin(self) -> None:
        cost = torch.tensor([100.0, 500.0, 1000.0, 750.0, 200.0], dtype=torch.float64)
        purity = torch.tensor([0.992, 0.995, 0.998, 0.997, 0.993], dtype=torch.float64)
        train_y = torch.stack([cost, purity], dim=-1)

        config = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)
        ref_point, _ = get_reference_point_dynamic(train_y, config=config)

        worst = train_y.max(dim=0).values
        margin = (ref_point - worst).cpu().numpy()
        # Cost lives in the range-dominated regime.
        assert margin[0] >= config.margin * (cost.max() - cost.min()).item() / 2
        # Purity's range (~0.006) is larger than the relative floor (0.01 *
        # 0.998 ≈ 0.00998), so the margin is range-driven for purity too in
        # this scenario. Both must be a meaningful fraction of the
        # objective's own scale.
        purity_scale = purity.max().item() - purity.min().item()
        assert margin[1] >= config.margin * purity_scale / 2
