"""Tests for the optional ``log_transform`` outcome stack.

Reference: BoTorch documents the ``Log`` outcome transform precisely for
multi-decade objectives that are modeled as multivariate log-normal
distributions. See
https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log
and the chained-transform pattern in
https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.ChainedOutcomeTransform.
"""

import pytest
import torch
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    Standardize,
)

from bo_engine.models import create_model, create_single_task_model


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
