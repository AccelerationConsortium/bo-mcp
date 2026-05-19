"""Tests for the ``auto_shift_for_log`` flag on the log-transform path (8.37).

The default ``log_transform=True`` factory rejects any non-positive
observation because ``Log`` produces ``-inf`` on zero / negative inputs.
Noisy real-lab data often has occasional non-positive readings; rejecting
the whole campaign forces users to preprocess outside the system.
``auto_shift_for_log=True`` computes ``shift = -min(y) + ε`` once at fit
and tracks it on the returned model so downstream code can back it out.

References:
    - BoTorch ``Log`` outcome-transform reference:
      https://botorch.org/api/models.html#botorch.models.transforms.outcome.Log
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.models import (
    LOG_AUTO_SHIFT_EPSILON,
    compute_log_auto_shift,
    create_and_fit_single_task_model,
)

SEED = 42
N_DIMS = 2


def _make_training_data(targets: list[float]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(SEED)
    n = len(targets)
    train_x = torch.rand(n, N_DIMS, dtype=torch.float64)
    train_y = torch.tensor(targets, dtype=torch.float64).unsqueeze(-1)
    bounds = torch.stack(
        [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
    )
    return train_x, train_y, bounds


class TestShiftComputation:
    """``compute_log_auto_shift`` returns the right additive shift."""

    def test_positive_data_yields_zero_shift(self) -> None:
        train_y = torch.tensor([[0.1], [0.5], [2.0]], dtype=torch.float64)
        assert compute_log_auto_shift(train_y) == 0.0

    def test_negative_min_yields_positive_shift_with_epsilon(self) -> None:
        train_y = torch.tensor([[-1.0], [0.5], [2.0]], dtype=torch.float64)
        shift = compute_log_auto_shift(train_y)
        assert shift == pytest.approx(1.0 + LOG_AUTO_SHIFT_EPSILON)

    def test_zero_observation_still_shifted_off_singularity(self) -> None:
        train_y = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
        shift = compute_log_auto_shift(train_y)
        assert shift == pytest.approx(LOG_AUTO_SHIFT_EPSILON)


class TestAutoShiftFitSucceeds:
    """A single non-positive observation must not block the fit when auto-shift is on."""

    def test_fit_succeeds_with_one_negative_observation(self) -> None:
        train_x, train_y, bounds = _make_training_data([-0.5, 0.2, 0.8, 1.5, 2.0, 0.3, 1.1])

        # Without auto-shift, log_transform must reject the data.
        with pytest.raises(ValueError, match="strictly positive"):
            create_and_fit_single_task_model(train_x, train_y, bounds, log_transform=True)

        # With auto-shift, the fit succeeds and the shift is recorded.
        model = create_and_fit_single_task_model(
            train_x, train_y, bounds, log_transform=True, auto_shift_for_log=True
        )
        assert hasattr(model, "_auto_shift_for_log")
        # Shift must equal ``-min(y) + eps`` and the model is fitted (has
        # likelihood noise hyperparameter populated post-fit).
        assert model._auto_shift_for_log == pytest.approx(0.5 + LOG_AUTO_SHIFT_EPSILON)


class TestNoOpWhenAlreadyPositive:
    """``auto_shift_for_log`` must be a no-op on data that is already positive."""

    def test_no_shift_attribute_when_data_strictly_positive(self) -> None:
        train_x, train_y, bounds = _make_training_data([0.1, 0.2, 0.3, 0.4, 0.5])
        model = create_and_fit_single_task_model(
            train_x, train_y, bounds, log_transform=True, auto_shift_for_log=True
        )
        # Strictly positive → no shift recorded.
        assert not hasattr(model, "_auto_shift_for_log")


class TestSuggestionProvenanceSubtractsShift:
    """Reported ``predicted_objectives`` must be on the user's raw scale.

    The reviewer found that ``auto_shift_for_log`` correctly applied the
    shift at fit time and recorded it on the model, but suggestion
    provenance reported the GP's posterior mean directly — i.e. the
    shifted-scale value. The fix subtracts the recorded shift in
    ``_build_single_objective_provenance`` so users see predictions
    on the same scale as their observed data.
    """

    def test_provenance_helper_subtracts_shift_from_predicted_mean(self) -> None:
        """Direct unit test: shifted-scale posterior mean minus shift = raw."""
        from bo_engine.suggestions import _build_single_objective_provenance

        means = torch.tensor([1.7, 2.4, 3.1, 0.8], dtype=torch.float64)
        stds = torch.tensor([0.05, 0.08, 0.06, 0.07], dtype=torch.float64)
        acq_values = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float64)
        shift = 0.5

        # Without shift subtraction, ``predicted_objectives["y"]`` would
        # equal ``means[i]`` (1.7); with the fix, it must equal
        # ``means[i] - shift`` (1.2).
        for i in range(means.numel()):
            _, _, _, predicted, _ = _build_single_objective_provenance(
                means, stds, acq_values, i, "y", minimize=True, auto_shift=shift
            )
            assert predicted is not None
            expected = float(means[i].item()) - shift
            assert predicted["y"] == pytest.approx(expected, rel=1e-9)

    def test_provenance_helper_unchanged_when_no_shift(self) -> None:
        """Zero shift (the common case) keeps the legacy behavior intact."""
        from bo_engine.suggestions import _build_single_objective_provenance

        means = torch.tensor([1.7, 2.4], dtype=torch.float64)
        stds = torch.tensor([0.1, 0.1], dtype=torch.float64)
        acq_values = torch.tensor([0.5, 0.5], dtype=torch.float64)

        for i in range(means.numel()):
            _, _, _, predicted_no_shift, _ = _build_single_objective_provenance(
                means, stds, acq_values, i, "y", minimize=True, auto_shift=0.0
            )
            _, _, _, predicted_legacy, _ = _build_single_objective_provenance(
                means, stds, acq_values, i, "y", minimize=True
            )
            assert predicted_no_shift == predicted_legacy


class TestMultiObjectiveAutoShiftRejected:
    """``auto_shift_for_log`` is single-objective only.

    The multi-objective factory builds per-objective sub-models but does
    not thread per-objective shifts through suggestion provenance — the
    bookkeeping is documented as single-objective-only. We reject the
    combination at the suggestion boundary so the campaign fails loudly
    rather than half-applying the shift.
    """

    def test_multi_objective_with_auto_shift_raises(self) -> None:
        from bo_engine.suggestions import generate_next_batch
        from bo_engine.types import (
            ObjectiveSpec,
            ObservationData,
            OptimizationSpec,
            ParameterSpec,
            ParameterType,
        )

        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True, log_transform=True),
                ObjectiveSpec(name="y2", minimize=True),
            ],
            auto_shift_for_log=True,
            random_seed=42,
            batch_size=2,
        )
        observations = [
            ObservationData(
                parameter_values={"x": float(x)},
                objective_values={"y1": 0.1 + x, "y2": 1.0 - x},
            )
            for x in (0.1, 0.3, 0.5, 0.7, 0.9)
        ]

        with pytest.raises(ValueError, match="single-objective"):
            generate_next_batch(spec, observations, iteration=1)
