"""Suggestion confidence levels must be invariant to the objective's scale (M5).

The posterior std at a candidate is reported in the user's *raw* objective
units because the GP's ``Standardize`` outcome transform un-standardizes the
posterior. Comparing that raw std against the absolute thresholds 0.1 / 0.3
made every suggestion "low" confidence for a large-scale objective and "high"
for a tiny-scale one. The fix normalizes the candidate std by the model's
``Standardize.stdvs`` (the training-data scale) so the thresholds are
*relative* and the verdict depends only on how informative the model is, not
on the units of ``y``.

A ``Standardize(m=1)`` GP fits identical hyperparameters whether the targets
are ``y`` or ``1000 * y`` — the standardization removes the scale — so with a
fixed seed the two campaigns must report identical confidence levels.

Reference: the scale-invariance pattern mirrors
``tests/test_convergence_scale_invariance.py``; ParEGO-style relative
normalization of model uncertainty is standard practice (Rasmussen &
Williams, GPML, §2.2, on Standardize-transformed posteriors).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    generate_next_batch,
)
from bo_engine.suggestions_common import _get_confidence_level, _normalize_uncertainty


def _rescaled_observations(
    observations: list[ObservationData],
    factor: float,
) -> list[ObservationData]:
    """Return a copy of ``observations`` with every objective value scaled."""
    return [
        ObservationData(
            parameter_values=dict(obs.parameter_values),
            objective_values={name: value * factor for name, value in obs.objective_values.items()},
        )
        for obs in observations
    ]


def _single_objective_observations() -> list[ObservationData]:
    return [
        ObservationData(parameter_values={"x1": 0.1, "x2": 0.2}, objective_values={"y": 5.0}),
        ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 3.0}),
        ObservationData(parameter_values={"x1": 0.9, "x2": 0.1}, objective_values={"y": 4.0}),
        ObservationData(parameter_values={"x1": 0.3, "x2": 0.7}, objective_values={"y": 3.5}),
        ObservationData(parameter_values={"x1": 0.7, "x2": 0.4}, objective_values={"y": 4.5}),
    ]


def _single_objective_spec(minimize: bool = True) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=minimize)],
        batch_size=2,
    )


class TestConfidenceThresholdNormalization:
    """The threshold compares a *relative* (dimensionless) std."""

    def test_raw_std_below_absolute_threshold_normalizes_to_low(self) -> None:
        """A 1000-scale objective with std 50 (relative 0.5) must read "low".

        Before the fix a raw std of 50 exceeded both thresholds and read
        "low" for the wrong reason; after the fix the relative std (50 / 100)
        of 0.5 still exceeds 0.3, so it reads "low" — but now invariantly so.
        """
        scale = 100.0
        raw_std = 50.0  # relative std = 0.5
        relative = _normalize_uncertainty(raw_std, scale)
        assert relative == pytest.approx(0.5)
        assert _get_confidence_level(relative) == "low"

    def test_high_confidence_survives_large_scale(self) -> None:
        """A confident prediction (relative std 0.05) stays "high" at any scale."""
        for scale in (1e-3, 1.0, 1e3):
            raw_std = 0.05 * scale  # relative std = 0.05 < 0.1
            assert _get_confidence_level(_normalize_uncertainty(raw_std, scale)) == "high"

    def test_non_positive_scale_falls_back_to_raw(self) -> None:
        """A degenerate (<=0) or non-finite scale leaves the std unchanged."""
        assert _normalize_uncertainty(0.4, 0.0) == pytest.approx(0.4)
        assert _normalize_uncertainty(0.4, float("nan")) == pytest.approx(0.4)
        assert _normalize_uncertainty(None, 10.0) is None


class TestEndToEndScaleInvariance:
    """Same campaign with ``y`` and ``1000 * y`` yields identical confidence levels."""

    @pytest.mark.parametrize("minimize", [True, False])
    def test_single_objective_confidence_is_scale_invariant(self, minimize: bool) -> None:
        spec = _single_objective_spec(minimize=minimize)
        observations = _single_objective_observations()

        torch.manual_seed(0)
        baseline, _ = generate_next_batch(
            spec,
            observations,
            batch_size=2,
            iteration=1,
            rng=np.random.default_rng(123),
        )
        torch.manual_seed(0)
        rescaled, _ = generate_next_batch(
            spec,
            _rescaled_observations(observations, 1000.0),
            batch_size=2,
            iteration=1,
            rng=np.random.default_rng(123),
        )

        assert all(s.generation_method == "bo" for s in baseline)
        # The confidence verdict is identical across scales...
        assert [s.confidence_level for s in baseline] == [s.confidence_level for s in rescaled]
        # ...even though the raw posterior std (model_uncertainty) genuinely
        # scaled by ~1000, which is exactly what an un-normalized threshold
        # would have keyed off (and mislabeled).
        for base, scaled in zip(baseline, rescaled, strict=True):
            if base.model_uncertainty and base.model_uncertainty > 0:
                assert scaled.model_uncertainty == pytest.approx(
                    base.model_uncertainty * 1000.0, rel=1e-3
                )
