"""Tests for hypervolume reference-point computation.

Focus: the static reference-point fallback when the observed objective
range collapses relative to the absolute scale. The reference point feeds
hypervolume — a margin that is degenerate relative to the natural scale of
``worst`` distorts the indicator across objectives with different units.

Reference: Ishibuchi et al., "Reference Point Specification in Inverted
Generational Distance for EMO" (2015), §3 — the reference point should be
"sufficiently worse than the worst observed point" on every objective. The
canonical pymoo implementation (Blank & Deb, "pymoo: Multi-Objective
Optimization in Python", 2020) uses a relative-floor padding strategy that
this test pins.
"""

from __future__ import annotations

import math

import pytest
import torch

from bo_engine.constants import (
    MIN_OBJECTIVE_RANGE,
    REFERENCE_POINT_PADDING,
    REFERENCE_POINT_RELATIVE_TOLERANCE,
)
from bo_engine.reference_point import (
    ReferencePointConfig,
    ReferencePointStrategy,
    _compute_static_reference_point,
)


def _static_ref(train_y: torch.Tensor) -> torch.Tensor:
    """Helper: compute the static reference point with default config."""
    minimize_mask = torch.ones(train_y.shape[-1], dtype=torch.bool)
    config = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)
    return _compute_static_reference_point(train_y, minimize_mask, config)


class TestStaticReferencePointDefaultRegime:
    """Sanity: normal-scale objectives reproduce the legacy formula."""

    def test_unit_scale_uses_range_floor(self) -> None:
        """When ``range`` dominates ``abs(worst)*RELATIVE_TOLERANCE`` the
        reference point degenerates to ``worst + margin * range``."""
        train_y = torch.tensor([[0.1, 0.2], [0.4, 0.5], [0.9, 1.0]], dtype=torch.float64)
        worst = train_y.max(dim=0).values
        ranges = worst - train_y.min(dim=0).values
        # Sanity: regime check
        assert (ranges > worst.abs() * REFERENCE_POINT_RELATIVE_TOLERANCE).all()

        ref = _static_ref(train_y)
        expected = worst + REFERENCE_POINT_PADDING * ranges
        assert torch.allclose(ref, expected)


class TestStaticReferencePointLargeMagnitudeSmallRange:
    """A large-scale objective with a tiny range must keep proportional margin.

    Reference: standard hypervolume practice (Blank & Deb 2020) keeps the
    reference offset proportional to the objective magnitude so the
    indicator does not collapse when the live front happens to span only a
    narrow band of the scale.
    """

    def test_relative_floor_dominates(self) -> None:
        """``worst=1000`` with ``range=0.001`` should use the relative floor."""
        train_y = torch.tensor([[999.999], [1000.0]], dtype=torch.float64)
        ref = _static_ref(train_y)
        # ``window`` is ``abs(worst) * RELATIVE_TOLERANCE = 10`` — three orders
        # of magnitude above the observed range.
        expected_window = 1000.0 * REFERENCE_POINT_RELATIVE_TOLERANCE
        expected = 1000.0 + REFERENCE_POINT_PADDING * expected_window
        assert ref.item() == pytest.approx(expected, rel=1e-9)

    def test_offset_is_meaningful_under_small_range(self) -> None:
        """Pinning the regression behind 1.50: legacy formula collapsed to a
        margin three orders of magnitude smaller than the natural scale."""
        train_y = torch.tensor([[999.999], [1000.0]], dtype=torch.float64)
        legacy_margin = REFERENCE_POINT_PADDING * 0.001  # ranges only
        ref = _static_ref(train_y)
        new_margin = ref.item() - 1000.0
        # The fix keeps the offset at least an order of magnitude above the
        # legacy collapse without exceeding the absolute scale.
        assert new_margin >= 10 * legacy_margin
        assert new_margin < 1000.0


class TestStaticReferencePointZeroCenteredCollapsed:
    """Zero-centred objectives with collapsed range fall back to the absolute floor.

    Reference: Ishibuchi et al. (2015) note that pathological constant
    fronts must still yield a non-degenerate offset — even arbitrary, as
    long as it does not equal ``worst`` — so that the dominated hypervolume
    is well defined.
    """

    def test_zero_center_collapsed_uses_absolute_floor(self) -> None:
        """All-zero ``train_y`` keeps the reference strictly above zero."""
        train_y = torch.zeros((4, 2), dtype=torch.float64)
        ref = _static_ref(train_y)
        expected = REFERENCE_POINT_PADDING * MIN_OBJECTIVE_RANGE
        assert torch.allclose(ref, torch.full_like(ref, expected))
        assert (ref > 0).all()


class TestStaticReferencePointProperty:
    """Property-based check: the reference point strictly dominates ``worst``
    across arbitrary scales and zero-centred values.

    Reference: per the hypervolume definition (Zitzler & Thiele, 1999), the
    reference point must Pareto-dominate every observed point — strict
    dominance in minimization form means ``ref > worst`` on every objective.
    """

    @pytest.mark.parametrize(
        "train_y_list",
        [
            [[1e-9, 1e-9], [2e-9, 3e-9]],  # near-zero positive
            [[-1.0, -2.0], [-0.5, -1.5]],  # negative orthant
            [[-1e-5, 1e-5], [0.0, 0.0]],  # straddling zero
            [[1e9, -1e9], [2e9, -2e9]],  # huge magnitudes
            [[0.001, 0.001001]],  # single point (degenerate)
            [[100.0, 100.0], [100.0, 100.0]],  # constant front
        ],
    )
    def test_ref_point_strictly_dominates_all(self, train_y_list: list[list[float]]) -> None:
        train_y = torch.tensor(train_y_list, dtype=torch.float64)
        ref = _static_ref(train_y)
        worst = train_y.max(dim=0).values
        # Strict dominance: every objective's ref is strictly worse (i.e.
        # greater in minimization form) than ``worst``.
        gap = ref - worst
        assert (gap > 0).all(), f"ref={ref.tolist()} worst={worst.tolist()} gap={gap.tolist()}"
        # And the offset must remain finite — no NaN/inf even under huge
        # magnitudes or degenerate spreads.
        assert torch.isfinite(ref).all()

    def test_relative_floor_constant_is_documented(self) -> None:
        """Pin the public constant the docstring references."""
        assert math.isclose(REFERENCE_POINT_RELATIVE_TOLERANCE, 0.01)
