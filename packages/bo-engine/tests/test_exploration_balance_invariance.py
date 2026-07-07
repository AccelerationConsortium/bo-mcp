"""Unit-invariance of the exploration/exploitation and diversity diagnostics.

The ``balance_assessment`` compares dimensionless thresholds against the
exploration ratio and diversity score, so both inputs must be normalized:
suggestions into the unit hypercube (diversity is measured against the
unit-cube expected pairwise distance) and the model uncertainty by the
observed objective scale. Without normalization the verdict is an artifact
of the measurement units — parameter ranges >> 1 saturate diversity at 1.0
and an objective std > 0.5 saturates exploration.

Test pattern follows ``test_convergence_scale_invariance.py``: the same
campaign expressed in different units must produce identical verdicts.

References:
    - Snoek et al., "Practical Bayesian Optimization of Machine Learning
      Algorithms" (NeurIPS 2012) — GP-based BO operates on normalized
      inputs/standardized outputs precisely so its behaviour is invariant
      to the units of the raw problem.
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.diagnostics import compute_suggestion_diversity
from bo_engine.diagnostics_usability import compute_exploration_exploitation_metrics

# A batch clustered in one corner of the unit cube (diversity well below
# saturation) so a scale-induced shift in the score is visible.
_UNIT_SUGGESTIONS = [[0.10, 0.12], [0.15, 0.18], [0.12, 0.22]]
_UNIT_BOUNDS = [[0.0, 0.0], [1.0, 1.0]]
_UNIT_UNCERTAINTIES = [0.05, 0.08, 0.06]
_UNIT_OBJECTIVE_SCALE = 1.0


def _metrics(x_scale: float, y_scale: float):
    suggestions = torch.tensor(_UNIT_SUGGESTIONS, dtype=torch.double) * x_scale
    bounds = torch.tensor(_UNIT_BOUNDS, dtype=torch.double) * x_scale
    uncertainties = [u * y_scale for u in _UNIT_UNCERTAINTIES]
    return compute_exploration_exploitation_metrics(
        suggestions=suggestions,
        best_point=suggestions[0],
        uncertainties=uncertainties,
        bounds=bounds,
        objective_scale=_UNIT_OBJECTIVE_SCALE * y_scale,
    )


class TestBalanceScaleInvariance:
    """Identical campaign in different units → identical verdicts."""

    def test_parameter_rescaling_leaves_metrics_unchanged(self) -> None:
        baseline = _metrics(x_scale=1.0, y_scale=1.0)
        rescaled = _metrics(x_scale=1000.0, y_scale=1.0)

        assert rescaled.balance_assessment == baseline.balance_assessment
        assert rescaled.diversity_score == pytest.approx(baseline.diversity_score)
        assert rescaled.average_distance_to_best == pytest.approx(baseline.average_distance_to_best)

    def test_objective_rescaling_leaves_metrics_unchanged(self) -> None:
        baseline = _metrics(x_scale=1.0, y_scale=1.0)
        rescaled = _metrics(x_scale=1.0, y_scale=1000.0)

        assert rescaled.balance_assessment == baseline.balance_assessment
        assert rescaled.exploration_ratio == pytest.approx(baseline.exploration_ratio)

    def test_diversity_score_not_saturated_by_wide_ranges(self) -> None:
        """A clustered batch must stay low-diversity in any units."""
        clustered = _metrics(x_scale=1000.0, y_scale=1.0)
        assert clustered.diversity_score < 0.5


class TestUnknownObjectiveScale:
    """Raw-unit uncertainty must not be compared against dimensionless thresholds."""

    def test_missing_scale_falls_back_to_neutral_ratio(self) -> None:
        suggestions = torch.tensor(_UNIT_SUGGESTIONS, dtype=torch.double)
        bounds = torch.tensor(_UNIT_BOUNDS, dtype=torch.double)
        metrics = compute_exploration_exploitation_metrics(
            suggestions=suggestions,
            best_point=None,
            uncertainties=[123.0, 456.0],
            bounds=bounds,
            objective_scale=None,
        )
        assert metrics.exploration_ratio == pytest.approx(0.5)
        assert metrics.average_distance_to_best is None


class TestSuggestionDiversityBounds:
    """``compute_suggestion_diversity`` normalizes raw values when given bounds."""

    def test_bounds_normalization_matches_unit_cube_score(self) -> None:
        unit = torch.tensor(_UNIT_SUGGESTIONS, dtype=torch.double)
        raw = unit * torch.tensor([100.0, 1000.0], dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0], [100.0, 1000.0]], dtype=torch.double)

        assert compute_suggestion_diversity(raw, bounds=bounds) == pytest.approx(
            compute_suggestion_diversity(unit)
        )
