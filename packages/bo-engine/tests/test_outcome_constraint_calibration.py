"""Tests for outcome-constraint calibration assessment.

Pins the contract: the constraint-feasibility GP must
expose calibration metrics (Brier score and Expected Calibration Error in
addition to mean absolute deviation) so the diagnostics layer can warn
before agents schedule infeasible experiments.

References:
    - Brier (1950), "Verification of Forecasts Expressed in Terms of
      Probability" — squared-error scoring rule for binary forecasts.
    - Naeini, Cooper & Hauskrecht, "Obtaining Well Calibrated Probabilities
      Using Bayesian Binning", AAAI 2015 — equal-width-bin ECE definition.
    - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017 —
      establishes the 10-bin default and the reliability-diagram framing.
"""

from __future__ import annotations

import math

import pytest
import torch

from bo_engine.constants import (
    CONSTRAINT_CALIBRATION_N_BINS,
    CONSTRAINT_CALIBRATION_WARN_THRESHOLD,
    NOISE_PRIOR_MIN_INFERRED,
)
from bo_engine.outcome_constraints import (
    ConstraintModelConfig,
    ConstraintModelingMethod,
    OutcomeConstraintSpec,
    _expected_calibration_error,
    assess_constraint_model_quality,
    build_constraint_model_binary,
    build_constraint_model_continuous,
    compute_outcome_constraint_calibration,
)


class TestExpectedCalibrationError:
    """Bucket-wise reliability-curve gap (Naeini et al., 2015).

    The ECE definition: bin probabilities into ``n_bins`` equal-width
    bins, compute ``|mean(prob) - frac(label==1)|`` per bin, and weight by
    bin occupancy. Empty bins drop out.
    """

    def test_perfect_calibration_is_zero(self) -> None:
        """When the predicted mass equals the empirical mass per bin ECE = 0."""
        # 5 zeros at prob 0.0, 5 ones at prob 1.0 — perfectly calibrated.
        probs = torch.tensor([0.0] * 5 + [1.0] * 5, dtype=torch.float64)
        labels = torch.tensor([0.0] * 5 + [1.0] * 5, dtype=torch.float64)
        assert _expected_calibration_error(probs, labels) == pytest.approx(0.0)

    def test_constant_overconfidence_recovers_gap(self) -> None:
        """Predicting 0.9 when the empirical rate is 0.5 → ECE = 0.4.

        Mirrors the canonical Guo et al. (2017) miscalibration example —
        modern classifiers that confidently predict P=0.9 on data where
        the empirical accuracy is only 50%.
        """
        probs = torch.full((100,), 0.9, dtype=torch.float64)
        labels = torch.cat([torch.zeros(50), torch.ones(50)]).to(torch.float64)
        ece = _expected_calibration_error(probs, labels)
        assert ece == pytest.approx(0.4, abs=1e-9)

    def test_uses_documented_bin_count(self) -> None:
        assert CONSTRAINT_CALIBRATION_N_BINS == 10

    def test_empty_input_returns_zero(self) -> None:
        """No data → no calibration error to report."""
        probs = torch.empty(0, dtype=torch.float64)
        labels = torch.empty(0, dtype=torch.float64)
        assert _expected_calibration_error(probs, labels) == 0.0


def _make_balanced_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    """Build a balanced (8 feasible / 8 infeasible) 1-D dataset.

    Returns a training tensor of inputs in [0, 1] and the corresponding
    objective values straddling 0.5 so the constraint GP can fit a
    non-degenerate boundary.
    """
    torch.manual_seed(0)
    feasible_x = torch.linspace(0.05, 0.45, steps=8, dtype=torch.float64).unsqueeze(-1)
    infeasible_x = torch.linspace(0.55, 0.95, steps=8, dtype=torch.float64).unsqueeze(-1)
    train_x = torch.cat([feasible_x, infeasible_x], dim=0)
    # feasible objectives strictly below threshold, infeasible strictly above
    obj = torch.cat(
        [
            torch.full((8,), 0.2, dtype=torch.float64),
            torch.full((8,), 0.8, dtype=torch.float64),
        ]
    ).unsqueeze(-1)
    return train_x, obj


class TestAssessConstraintModelQuality:
    """``assess_constraint_model_quality`` returns Brier + ECE alongside legacy MAD."""

    def test_returns_new_calibration_metrics(self) -> None:
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_binary(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.BINARY),
        )
        metrics = assess_constraint_model_quality(result, train_x, obj)

        # Legacy keys still present so callers do not break.
        assert "calibration_error" in metrics
        assert "auc" in metrics
        # New keys introduced by 1.52.
        assert "brier_score" in metrics
        assert "expected_calibration_error" in metrics
        assert "is_calibrated" in metrics

        # All probabilities/labels are in [0, 1] so each scalar metric is too.
        for key in ("calibration_error", "brier_score", "expected_calibration_error"):
            value = metrics[key]
            assert isinstance(value, float)
            assert 0.0 <= value <= 1.0

        # ``is_calibrated`` is tied to the documented threshold.
        assert metrics["is_calibrated"] == (
            metrics["calibration_error"] <= CONSTRAINT_CALIBRATION_WARN_THRESHOLD
        )

    def test_well_separated_data_yields_low_brier(self) -> None:
        """Perfectly separable feasibility → small Brier on training data."""
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_binary(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.BINARY),
        )
        metrics = assess_constraint_model_quality(result, train_x, obj)
        # On a well-separated training set the model should at minimum be
        # better than a coin-flip Brier (0.25).
        assert metrics["brier_score"] < 0.25

    def test_single_class_data_yields_uninformative_auc_without_raising(self) -> None:
        """All-feasible (single-class) data falls back to AUC 0.5 via the guard.

        AUC is undefined with only one class present, so the both-classes
        guard (``0 < sum < len``) is false, ``roc_auc_score`` is never called,
        and ``auc`` keeps its uninformative 0.5 default. (The ``ValueError``
        fallback branch is exercised separately in
        :meth:`test_auc_value_error_falls_back_to_uninformative`.)
        """
        torch.manual_seed(0)
        train_x = torch.linspace(0.05, 0.95, steps=10, dtype=torch.float64).unsqueeze(-1)
        # Every observation is below the threshold ⇒ all feasible ⇒ one class.
        obj = torch.full((10,), 0.2, dtype=torch.float64).unsqueeze(-1)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_binary(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.BINARY),
        )

        metrics = assess_constraint_model_quality(result, train_x, obj)
        assert metrics["auc"] == 0.5

    def test_auc_value_error_falls_back_to_uninformative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``ValueError`` from ``roc_auc_score`` is caught and yields AUC 0.5.

        This pins the actual L2 fix (the replaced ``except`` branch): with
        *both* classes present the guard passes and ``roc_auc_score`` is
        called, so monkeypatching it to raise ``ValueError`` proves the
        handler swallows it and falls back to 0.5. Reverting the production
        code to ``except ImportError`` would let the ``ValueError`` escape and
        fail this test.
        """
        import bo_engine.outcome_constraints as oc_module

        called = {"hit": False}

        def raising_auc(*_args, **_kwargs):
            called["hit"] = True
            msg = "Only one class present in y_true"
            raise ValueError(msg)

        monkeypatch.setattr(oc_module, "roc_auc_score", raising_auc)

        # Balanced data so the both-classes guard passes and the patched
        # ``roc_auc_score`` is actually invoked.
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_binary(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.BINARY),
        )

        metrics = assess_constraint_model_quality(result, train_x, obj)
        assert called["hit"] is True
        assert metrics["auc"] == 0.5


class TestComputeOutcomeConstraintCalibration:
    """End-to-end wrapper used by the server diagnostics layer."""

    def test_returns_one_report_per_constraint(self) -> None:
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        objective_values = {"yield": obj.squeeze(-1)}
        constraint_specs = [
            OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False),
            OutcomeConstraintSpec(objective_name="yield", threshold=0.3, greater_than=True),
        ]
        reports = compute_outcome_constraint_calibration(
            constraint_specs=constraint_specs,
            train_x=train_x,
            objective_values=objective_values,
            bounds=bounds,
        )
        assert len(reports) == 2
        for report in reports:
            assert report["constraint_name"] == "yield"
            assert "brier_score" in report
            assert "expected_calibration_error" in report
            assert math.isfinite(report["brier_score"])

    def test_missing_objective_returns_error_marker(self) -> None:
        """A constraint that points at an unobserved objective surfaces a marker."""
        train_x, _ = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        specs = [OutcomeConstraintSpec(objective_name="missing", threshold=0.5, greater_than=False)]
        reports = compute_outcome_constraint_calibration(
            constraint_specs=specs,
            train_x=train_x,
            objective_values={},
            bounds=bounds,
        )
        assert reports == [
            {
                "constraint_name": "missing",
                "threshold": 0.5,
                "greater_than": False,
                "error": "missing_objective",
            }
        ]


def _constraint_noise_lower_bound(model):
    """Return the active inferred-noise lower bound of a constraint GP.

    Mirrors ``tests/test_noise_prior.py``: the hardened model factory floors
    the trainable noise at ``NOISE_PRIOR_MIN_INFERRED`` via a
    ``GreaterThan`` constraint on ``raw_noise``. A constraint GP built inline
    (raw ``SingleTaskGP``) would default to ``1e-4``-free GPyTorch behaviour
    and fail this bound, so it pins routing through the factory.

    The attribute walk goes through ``getattr`` to side-step the
    ``Tensor | Module`` union ty cannot narrow for the GPyTorch public stubs
    (same approach as ``test_noise_prior._likelihood_attr``).
    """
    obj = getattr(model, "likelihood")  # noqa: B009
    for name in ("noise_covar", "raw_noise_constraint", "lower_bound"):
        obj = getattr(obj, name)
    return obj


class TestConstraintModelsUseHardenedFactory:
    """Both constraint builders must inherit the factory's noise floor.

    ``build_constraint_model_continuous`` / ``_binary`` previously fit a raw
    ``SingleTaskGP`` inline, bypassing the calibrated noise prior and the
    ``GreaterThan(NOISE_PRIOR_MIN_INFERRED)`` floor that the rest of the engine
    relies on against singular-Cholesky failures. Routing them through
    ``create_and_fit_single_task_model`` restores that floor.
    """

    def test_continuous_builder_floors_noise(self) -> None:
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_continuous(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.CONTINUOUS),
        )
        lower_bound = _constraint_noise_lower_bound(result.model)
        assert torch.isclose(
            lower_bound,
            torch.tensor(NOISE_PRIOR_MIN_INFERRED, dtype=lower_bound.dtype),
        )

    def test_binary_builder_floors_noise(self) -> None:
        train_x, obj = _make_balanced_dataset()
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        spec = OutcomeConstraintSpec(objective_name="yield", threshold=0.5, greater_than=False)
        result = build_constraint_model_binary(
            train_x,
            obj,
            bounds,
            spec,
            ConstraintModelConfig(method=ConstraintModelingMethod.BINARY),
        )
        lower_bound = _constraint_noise_lower_bound(result.model)
        assert torch.isclose(
            lower_bound,
            torch.tensor(NOISE_PRIOR_MIN_INFERRED, dtype=lower_bound.dtype),
        )
