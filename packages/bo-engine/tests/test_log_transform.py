"""Tests for the optional ``log_transform`` outcome stack.

Reference: BoTorch documents the ``Log`` outcome transform precisely for
multi-decade objectives that are modeled as multivariate log-normal
distributions. See
https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log
and the chained-transform pattern in
https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.ChainedOutcomeTransform.
"""

from typing import cast

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
)
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

from bo_engine.models import (
    DeltaMethodLog,
    create_and_fit_single_task_model,
    create_model,
    create_single_task_model,
    inspect_standardize_stdvs,
)


def _fixed_noise_vector(model: SingleTaskGP) -> torch.Tensor:
    """Flat fixed observation-noise vector recorded on the model's likelihood."""
    likelihood = model.likelihood
    assert isinstance(likelihood, FixedNoiseGaussianLikelihood)
    noise = likelihood.noise
    assert isinstance(noise, torch.Tensor)
    return noise.detach().reshape(-1)


def test_default_outcome_transform_is_pure_standardize() -> None:
    """Without ``log_transform`` the GP keeps the historical ``Standardize`` stack."""
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = torch.rand(8, 1, dtype=torch.double) + 0.1
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    model = create_single_task_model(train_x, train_y, bounds)

    assert isinstance(model.outcome_transform, Standardize)


def test_log_transform_wraps_in_chained_log_then_standardize() -> None:
    """``log_transform=True`` chains ``Log`` before ``Standardize``."""
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = torch.exp(torch.linspace(-3.0, 3.0, 8, dtype=torch.double)).unsqueeze(-1)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    model = create_single_task_model(train_x, train_y, bounds, log_transform=True)

    assert isinstance(model.outcome_transform, ChainedOutcomeTransform)
    children = dict(model.outcome_transform.items())
    assert isinstance(children["log"], Log)
    assert isinstance(children["standardize"], Standardize)


def test_log_transform_per_objective_list_for_model_list() -> None:
    """Per-objective list lets users enable log-transform only on multi-decade targets."""
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = torch.stack(
        [
            torch.rand(8, dtype=torch.double) + 0.1,  # bounded objective
            torch.exp(torch.linspace(-2.0, 2.0, 8, dtype=torch.double)),  # multi-decade
        ],
        dim=-1,
    )
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    model_list = create_model(train_x, train_y, bounds, log_transform=[False, True])
    transforms = [m.outcome_transform for m in model_list.models]

    assert isinstance(transforms[0], Standardize)
    assert isinstance(transforms[1], ChainedOutcomeTransform)


def test_log_transform_list_length_must_match_objectives() -> None:
    """Mismatched list length fails fast at model construction."""
    train_x = torch.rand(8, 2, dtype=torch.double)
    train_y = torch.rand(8, 2, dtype=torch.double) + 0.1
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    with pytest.raises(ValueError, match="number of objectives"):
        create_model(train_x, train_y, bounds, log_transform=[True])


def test_objective_spec_log_transform_threads_into_suggestions() -> None:
    """``ObjectiveSpec.log_transform=True`` reaches the GP fit during generation.

    Smoke test: an end-to-end ``generate_next_batch`` call with a
    log-transformed minimize objective should produce valid suggestions
    and the underlying GP should own a ``ChainedOutcomeTransform`` —
    not just the default ``Standardize``. Without the wiring fix this
    test would either crash (flag dropped silently) or fit a plain
    ``Standardize`` model.
    """
    import numpy as np

    from bo_engine import (
        ObjectiveSpec,
        ObservationData,
        OptimizationSpec,
        ParameterSpec,
        ParameterType,
        generate_next_batch,
    )

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="rate", minimize=True, log_transform=True)],
        batch_size=2,
    )
    # Multi-decade positive targets — the case ``log_transform`` is for.
    raw = np.array([0.01, 0.1, 1.0, 10.0])
    observations = [
        ObservationData(parameter_values={"x": float(x)}, objective_values={"rate": float(y)})
        for x, y in zip(np.linspace(0.0, 1.0, len(raw)), raw, strict=True)
    ]
    suggestions, _state = generate_next_batch(spec, observations)
    assert len(suggestions) == 2
    # Predictions must come back on the raw user scale: positive and within
    # the observed multi-decade envelope (a sign or scale slip lands far
    # outside it).
    for suggestion in suggestions:
        assert suggestion.predicted_objectives is not None
        predicted = suggestion.predicted_objectives["rate"]
        assert 0.0 < predicted < 100.0


def test_log_transform_rejects_non_positive_targets_single_objective() -> None:
    """Zero / negative train_Y values surface at construction, not deep in fit.

    Reference: BoTorch's ``Log`` outcome transform requires
    strictly positive inputs — see
    https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log.
    Letting a zero or negative target slip past would produce
    ``-inf`` / ``nan`` and break the subsequent ``Standardize`` mean
    estimate; the explicit guard turns that silent corruption into an
    actionable ``ValueError`` at the boundary.
    """
    train_x = torch.rand(6, 2, dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    train_y_zero = torch.tensor([1.0, 2.0, 0.0, 3.0, 4.0, 5.0], dtype=torch.double).unsqueeze(-1)
    with pytest.raises(ValueError, match="strictly positive"):
        create_single_task_model(train_x, train_y_zero, bounds, log_transform=True)

    train_y_negative = torch.tensor([1.0, 2.0, -1.0, 3.0, 4.0, 5.0], dtype=torch.double).unsqueeze(
        -1
    )
    with pytest.raises(ValueError, match="strictly positive"):
        create_single_task_model(train_x, train_y_negative, bounds, log_transform=True)


def test_log_transform_rejects_non_positive_targets_multi_objective() -> None:
    """Per-objective rejection: only the offending column raises."""
    train_x = torch.rand(6, 2, dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    train_y = torch.stack(
        [
            torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.double),
            torch.tensor([0.1, 0.2, -0.3, 0.4, 0.5, 0.6], dtype=torch.double),
        ],
        dim=-1,
    )
    with pytest.raises(ValueError, match=r"objective\[1\]"):
        create_model(train_x, train_y, bounds, log_transform=[True, True])


def test_objective_spec_log_transform_rejects_maximize() -> None:
    """A maximize + log_transform combination raises at suggestion time.

    ``log_transform`` is part of the public spec contract only for
    minimize objectives (see ``ObjectiveSpec``); the suggestion
    generation path must surface the unsupported maximize combination as
    an actionable ``ValueError`` at the boundary instead of silently
    fitting something else.
    """
    import numpy as np

    from bo_engine import (
        ObjectiveSpec,
        ObservationData,
        OptimizationSpec,
        ParameterSpec,
        ParameterType,
        generate_next_batch,
    )

    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="rate", minimize=False, log_transform=True)],
        batch_size=1,
    )
    observations = [
        ObservationData(parameter_values={"x": float(x)}, objective_values={"rate": float(y)})
        for x, y in zip(np.linspace(0.0, 1.0, 4), [0.01, 0.1, 1.0, 10.0], strict=True)
    ]
    with pytest.raises(ValueError, match="minimize=True"):
        generate_next_batch(spec, observations)


def test_log_transform_negated_targets_chain_negate_before_log() -> None:
    """Maximization-form (negated) minimize targets chain ``Negate → Log → Standardize``.

    Under the engine's maximization-form convention a minimize objective
    reaches the model factory negated, while BoTorch's ``Log`` requires the
    raw positive scale. ``target_negated=True`` therefore inserts a
    ``Negate`` stage in front of ``Log`` and re-negates the untransformed
    posterior, keeping the posterior on the scale of the supplied data.
    """
    from bo_engine.models import Negate, create_and_fit_single_task_model

    torch.manual_seed(7)
    train_x = torch.rand(8, 2, dtype=torch.double)
    raw_y = torch.exp(torch.linspace(-3.0, 3.0, 8, dtype=torch.double)).unsqueeze(-1)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    model = create_and_fit_single_task_model(
        train_x, -raw_y, bounds, log_transform=True, target_negated=True
    )

    assert isinstance(model.outcome_transform, ChainedOutcomeTransform)
    children = dict(model.outcome_transform.items())
    assert isinstance(children["negate"], Negate)
    assert isinstance(children["log"], Log)
    assert isinstance(children["standardize"], Standardize)

    # The posterior must live on the supplied (negated) scale: strictly
    # negative for strictly positive raw targets.
    model.eval()
    with torch.no_grad():
        mean = model.posterior(train_x).mean
    assert (mean < 0).all(), "Posterior mean must stay on the negated input scale"
    # Un-negating must recover the multi-decade ordering of the raw
    # targets — a broken chain (e.g. log applied to negated data, or a
    # missing re-negation) destroys the monotone correspondence.
    recovered = -mean.squeeze(-1)
    assert torch.equal(recovered.argsort(), raw_y.squeeze(-1).argsort()), (
        "Un-negated posterior means must preserve the raw target ordering"
    )
    # And the dominant observation is recovered on the right decade.
    largest = float(recovered.max().item())
    assert raw_y.max().item() / 2 < largest < raw_y.max().item() * 2


class TestLogTransformWithKnownMeasurementUncertainty:
    """``log_transform`` + ``train_yvar`` builds a fixed-noise GP via the delta method.

    BoTorch's stock ``Log`` outcome transform raises ``NotImplementedError``
    when observation noise is supplied, so a campaign that combined
    ``log_transform=True`` with full measurement-uncertainty coverage used
    to crash at GP construction. The engine's ``DeltaMethodLog`` closes the
    gap by propagating the variance into log space with the first-order
    Taylor (delta method) approximation ``Var[log Y] ~= Var[Y] / y**2``.

    References:
        - Casella & Berger, *Statistical Inference*, 2nd ed. (2002),
          §5.5.4 "The Delta Method": ``Var[g(Y)] ~= g'(mu)**2 Var[Y]``;
          with ``g = log`` this is the standard error-propagation rule
          ``Var[log Y] ~= Var[Y] / y**2``.
        - BoTorch ``Log`` outcome transform (raises on ``Yvar``):
          https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log
    """

    @staticmethod
    def _problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Multi-decade positive targets on deliberately non-unit bounds."""
        torch.manual_seed(11)
        n = 10
        bounds = torch.tensor([[0.0, 0.0], [100.0, 5.0]], dtype=torch.double)
        train_x = torch.rand(n, 2, dtype=torch.double) * (bounds[1] - bounds[0]) + bounds[0]
        raw_y = torch.exp(torch.linspace(-2.0, 2.0, n, dtype=torch.double)).unsqueeze(-1)
        yvar = (0.05 * raw_y) ** 2  # 5% relative stddev, in variance units
        return train_x, raw_y, yvar, bounds

    def test_fixed_noise_log_model_builds_and_predicts_finite(self) -> None:
        """The combination that used to raise ``NotImplementedError`` now fits.

        Uses the engine's maximization-form convention for a minimize
        objective (negated targets + ``target_negated=True``) and non-unit
        parameter bounds. The posterior must be finite and stay on the
        negated input scale.
        """
        train_x, raw_y, yvar, bounds = self._problem()

        model = create_and_fit_single_task_model(
            train_x,
            -raw_y,
            bounds,
            train_yvar=yvar,
            log_transform=True,
            target_negated=True,
        )

        assert isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
        model.eval()
        with torch.no_grad():
            posterior = model.posterior(train_x)
        assert torch.isfinite(posterior.mean).all()
        assert torch.isfinite(posterior.variance).all()
        assert (posterior.variance > 0).all()
        # Direction contract: posterior lives on the supplied (negated)
        # scale, and un-negating recovers the raw multi-decade ordering.
        assert (posterior.mean < 0).all()
        recovered = -posterior.mean.squeeze(-1)
        assert torch.equal(recovered.argsort(), raw_y.squeeze(-1).argsort())

    def test_likelihood_noise_is_delta_method_variance(self) -> None:
        """The fixed noise equals ``(yvar / y**2) / stdvs**2`` exactly.

        The outcome chain is ``Negate -> DeltaMethodLog -> Standardize``:
        ``Negate`` passes the (sign-invariant) variance through,
        ``DeltaMethodLog`` divides by the raw positive ``y**2`` (delta
        method, Casella & Berger 2002, §5.5.4), and ``Standardize``
        rescales by its stored ``stdvs**2``. A model that silently dropped
        or mis-scaled the variance would fail this equality.
        """
        train_x, raw_y, yvar, bounds = self._problem()

        model = create_single_task_model(
            train_x,
            -raw_y,
            bounds,
            train_yvar=yvar,
            log_transform=True,
            target_negated=True,
        )

        stddev = inspect_standardize_stdvs(model)[0]
        expected = (yvar / raw_y.pow(2)) / stddev**2
        assert torch.allclose(_fixed_noise_vector(model), expected.reshape(-1), rtol=1e-9)

    def test_multi_objective_mixed_log_flags_route_yvar_per_objective(self) -> None:
        """Only the log-transformed column gets the delta-method mapping."""
        torch.manual_seed(11)
        n = 8
        bounds = torch.tensor([[0.0, 0.0], [100.0, 5.0]], dtype=torch.double)
        train_x = torch.rand(n, 2, dtype=torch.double) * (bounds[1] - bounds[0]) + bounds[0]
        raw_log_obj = torch.exp(torch.linspace(-2.0, 2.0, n, dtype=torch.double))
        plain_obj = torch.linspace(0.0, 1.0, n, dtype=torch.double)
        # Column 0: minimize + log_transform (arrives negated); column 1: maximize.
        train_y = torch.stack([-raw_log_obj, plain_obj], dim=-1)
        yvar = torch.stack(
            [(0.1 * raw_log_obj) ** 2, torch.full((n,), 0.01, dtype=torch.double)],
            dim=-1,
        )

        model_list = create_model(
            train_x,
            train_y,
            bounds,
            train_yvar=yvar,
            log_transform=[True, False],
            target_negated=[True, False],
        )

        log_model, plain_model = cast(list[SingleTaskGP], list(model_list.models))
        assert isinstance(log_model.likelihood, FixedNoiseGaussianLikelihood)
        assert isinstance(plain_model.likelihood, FixedNoiseGaussianLikelihood)

        log_stddev, plain_stddev = inspect_standardize_stdvs(model_list)
        expected_log = (yvar[:, 0:1] / raw_log_obj.unsqueeze(-1).pow(2)) / log_stddev**2
        assert torch.allclose(_fixed_noise_vector(log_model), expected_log.reshape(-1), rtol=1e-9)

        expected_plain = yvar[:, 1:2] / plain_stddev**2
        assert torch.allclose(
            _fixed_noise_vector(plain_model), expected_plain.reshape(-1), rtol=1e-9
        )

    def test_delta_method_transform_round_trips_noise(self) -> None:
        """``forward`` then ``untransform`` recovers targets and noise.

        The inverse map is ``Var[Y] ~= exp(z)**2 * Var[Z]`` for
        ``z = log(y)`` — the delta method applied to ``g = exp``.
        """
        transform = DeltaMethodLog()
        y = torch.tensor([[0.5], [2.0], [40.0]], dtype=torch.double)
        yvar = torch.tensor([[0.01], [0.04], [4.0]], dtype=torch.double)

        y_tf, yvar_tf = transform(y, yvar)
        assert torch.allclose(y_tf, torch.log(y))
        assert yvar_tf is not None
        assert torch.allclose(yvar_tf, yvar / y.pow(2))

        y_rt, yvar_rt = transform.untransform(y_tf, yvar_tf)
        assert torch.allclose(y_rt, y)
        assert yvar_rt is not None
        assert torch.allclose(yvar_rt, yvar)

    def test_delta_method_requires_strictly_positive_targets(self) -> None:
        """Non-positive pre-log targets with noise raise an actionable error.

        The delta method divides by ``y**2`` and the log itself needs
        ``y > 0``; the transform re-validates so standalone use cannot emit
        non-finite noise silently.
        """
        transform = DeltaMethodLog()
        y = torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.double)
        yvar = torch.full_like(y, 0.01)

        with pytest.raises(ValueError, match="strictly positive"):
            transform(y, yvar)


def test_negate_outcome_transform_is_involution() -> None:
    """``Negate`` is its own inverse for targets, noise, and posteriors."""
    from bo_engine.models import Negate

    transform = Negate()
    y = torch.tensor([[1.5], [-2.0], [0.25]], dtype=torch.double)
    yvar = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.double)

    y_tf, yvar_tf = transform(y, yvar)
    assert torch.equal(y_tf, -y)
    assert yvar_tf is not None
    assert torch.equal(yvar_tf, yvar)

    y_round_trip, yvar_round_trip = transform.untransform(y_tf, yvar_tf)
    assert torch.equal(y_round_trip, y)
    assert yvar_round_trip is not None
    assert torch.equal(yvar_round_trip, yvar)


def test_negate_chain_supports_subset_output() -> None:
    """``subset_output`` must work through a ``Negate → Log → Standardize`` chain.

    BoTorch's ``subset_model`` utilities subset a model's outcome transform
    via ``OutcomeTransform.subset_output`` (see
    https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.OutcomeTransform.subset_output);
    a chain member without the method raises ``NotImplementedError`` deep
    inside those utilities.
    """
    from bo_engine.models import Negate, create_and_fit_single_task_model

    torch.manual_seed(7)
    train_x = torch.rand(8, 2, dtype=torch.double)
    raw_y = torch.exp(torch.linspace(-2.0, 2.0, 8, dtype=torch.double)).unsqueeze(-1)
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

    model = create_and_fit_single_task_model(
        train_x, -raw_y, bounds, log_transform=True, target_negated=True
    )

    chain = model.outcome_transform
    assert isinstance(chain, ChainedOutcomeTransform)
    subset = chain.subset_output([0])
    assert isinstance(subset, ChainedOutcomeTransform)
    children = dict(subset.items())
    assert isinstance(children["negate"], Negate)
    assert isinstance(children["log"], Log)
    assert isinstance(children["standardize"], Standardize)
    # The fitted chain is in eval mode; the subset must preserve that so it
    # does not re-estimate transform parameters on the next forward pass.
    assert not children["negate"].training
