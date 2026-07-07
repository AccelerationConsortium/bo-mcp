"""Diagnostics-GP / production-GP configuration parity.

The transparency surface (LOO-CV, correlation, ``hyperparameters``)
must describe the *same* model configuration the suggestion pipeline
fits: a fixed-noise campaign's diagnostics must come from a fixed-noise
GP, a categorical-kernel campaign's ``kernel_type`` must reflect the
mixed kernel, and a log-transform campaign's fit must go through the
log outcome chain. Both paths resolve their model options through the
shared :func:`bo_engine.suggestions_training.resolve_model_options`.

References:
    - BoTorch ``SingleTaskGP`` with ``train_Yvar`` (fixed-noise GP):
      https://botorch.org/docs/models — supplying known measurement
      variance replaces the trainable noise hyperparameter, so a
      diagnostics fit without it would report an MLL-estimated noise that
      the production model does not have.
"""

from __future__ import annotations

import torch

from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.diagnostics_usability import extract_hyperparameters
from bo_engine.models import ChainedOutcomeTransform, _has_fixed_noise
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _continuous_spec(log_transform: bool = False) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True, log_transform=log_transform)],
    )


def _observations(n: int = 8, uncertainty: float | None = None) -> list[ObservationData]:
    torch.manual_seed(11)
    xs = torch.linspace(0.05, 0.95, n)
    return [
        ObservationData(
            parameter_values={"x": float(x)},
            objective_values={"y": float((x - 0.3) ** 2 + 1.0)},
            measurement_uncertainty=({"y": uncertainty} if uncertainty is not None else None),
        )
        for x in xs
    ]


class TestFixedNoiseParity:
    """Measurement-uncertainty campaigns get a fixed-noise diagnostics GP."""

    def test_diagnostic_fit_uses_fixed_noise_likelihood(self) -> None:
        backend = BoTorchBackend()
        spec = _continuous_spec()
        observations = _observations(uncertainty=0.05)

        fit = backend._fit_diagnostic_model(spec, observations, is_single=True)

        assert fit is not None
        assert _has_fixed_noise(fit.model), (
            "The diagnostics GP must use the campaign's FixedNoiseGaussianLikelihood — "
            "a trainable-noise fit reports an MLL-estimated noise variance the "
            "production surrogate does not have."
        )

    def test_hyperparameters_section_survives_fixed_noise(self) -> None:
        """The noise readout must handle the per-observation noise vector."""
        backend = BoTorchBackend()
        spec = _continuous_spec()
        observations = _observations(uncertainty=0.05)

        fit = backend._fit_diagnostic_model(spec, observations, is_single=True)
        assert fit is not None
        result = backend._compute_hyperparameters(fit)

        hp = result["hyperparameters"]
        assert hp is not None
        assert hp["noise_variance"] > 0.0

    def test_trainable_noise_without_uncertainty(self) -> None:
        backend = BoTorchBackend()
        spec = _continuous_spec()
        observations = _observations(uncertainty=None)

        fit = backend._fit_diagnostic_model(spec, observations, is_single=True)

        assert fit is not None
        assert not _has_fixed_noise(fit.model)


class TestCategoricalKernelParity:
    """``use_categorical_kernel`` campaigns get the mixed kernel in diagnostics."""

    @staticmethod
    def _mixed_spec() -> OptimizationSpec:
        return OptimizationSpec(
            parameters=[
                ParameterSpec(name="t", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(
                    name="solv", type=ParameterType.CATEGORICAL, categories=["etoh", "meoh"]
                ),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            use_categorical_kernel=True,
        )

    @staticmethod
    def _mixed_observations() -> list[ObservationData]:
        torch.manual_seed(13)
        observations = []
        for i in range(8):
            t = 0.1 + 0.1 * i
            solvent = "etoh" if i % 2 == 0 else "meoh"
            observations.append(
                ObservationData(
                    parameter_values={"t": t, "solv": solvent},
                    objective_values={"y": (t - 0.4) ** 2 + (0.2 if solvent == "meoh" else 0.0)},
                )
            )
        return observations

    def test_kernel_type_reflects_mixed_kernel(self) -> None:
        backend = BoTorchBackend()
        fit = backend._fit_diagnostic_model(
            self._mixed_spec(), self._mixed_observations(), is_single=True
        )

        assert fit is not None
        hp = extract_hyperparameters(fit.model, ["t", "solv"])
        assert hp.kernel_type == "AdditiveKernel", (
            "A use_categorical_kernel campaign must report the mixed additive "
            "kernel, not the plain RBF the old diagnostics fit used."
        )


class TestLogTransformParity:
    """Log-transform campaigns fit the diagnostics GP through the log chain."""

    def test_outcome_transform_includes_log_stage(self) -> None:
        backend = BoTorchBackend()
        spec = _continuous_spec(log_transform=True)
        observations = _observations(uncertainty=None)

        fit = backend._fit_diagnostic_model(spec, observations, is_single=True)

        assert fit is not None
        assert isinstance(fit.model.outcome_transform, ChainedOutcomeTransform), (
            "A log_transform objective must fit diagnostics on the log scale — "
            "raw-scale LOO metrics of multi-decade data describe a different model."
        )
