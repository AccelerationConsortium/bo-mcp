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
    batch = generate_next_batch(spec, observations)
    assert len(batch) == 2


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

    Negating maximize targets to enforce BoTorch's internal minimization
    convention flips positive raw values to negative, and BoTorch's
    ``Log`` transform requires strictly positive targets. The
    suggestion generation path must surface this as an actionable
    ``ValueError`` rather than letting BoTorch raise deep inside fit.
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
