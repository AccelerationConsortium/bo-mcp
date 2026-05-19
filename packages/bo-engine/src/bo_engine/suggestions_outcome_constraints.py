"""Outcome-constraint GP construction for the suggestion pipeline.

Split from :mod:`bo_engine.suggestions` so the per-constraint GP fits
and their negative-feasible sign convention live in one place. The
single- and multi-objective batch pipelines invoke
:func:`_build_outcome_constraint_models` and forward the resulting
``(model, signed_threshold)`` pairs into BoTorch's MC feasibility
machinery via :mod:`bo_engine.outcome_constraints`.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from bo_engine.device import get_device, get_dtype
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.types import ObservationData, OptimizationSpec


class OutcomeConstraintConfigurationError(ValueError):
    """Raised when an outcome constraint cannot be honored by the engine.

    Outcome constraints reference an objective by name. The intake-layer
    validator rejects unknown names at create time so this exception only
    fires when a spec that was valid at intake later lost the named
    objective (e.g. an in-place spec edit, a backend conversion that
    dropped a column, or a stale persisted spec). Surfacing it as a typed
    error keeps such regressions loud — the previous silent ``return
    None`` quietly disabled the constraint and produced unconstrained
    suggestions that looked feasible.
    """


def _build_outcome_constraint_models(
    spec: OptimizationSpec,
    observations: list[ObservationData],
    train_x: Tensor,
    bounds: Tensor,
) -> list[tuple[Any, float]] | None:
    """Build constraint models for outcome constraints.

    For each outcome constraint, trains a GP and returns
    ``(model, signed_threshold)`` pairs that the existing
    ``_make_outcome_constraint_callable`` consumes as
    ``threshold - samples`` (BoTorch's MC feasibility convention:
    **negative return = feasible**; see
    :func:`botorch.utils.objective.compute_smoothed_feasibility_indicator`).
    The shape supports both modeling methods:

    * ``"continuous"`` (default; Gardner et al., ICML 2014): GP fits the raw
      constrained-objective values. We encode the constraint direction at
      fit time so a single callable formula ``threshold - samples`` lands
      at the right sign for BoTorch's negative-feasible convention:

      - For ``>=`` (feasible when ``raw_obj >= bound``) the GP is fit on
        raw values with ``threshold = bound``. The callable
        ``threshold - samples`` is negative exactly when
        ``samples > bound`` — i.e. feasible.
      - For ``<=`` (feasible when ``raw_obj <= bound``) we fit on the
        *negated* objective and store ``threshold = -bound``. The same
        callable becomes ``(-bound) - (-raw_obj_pred) = raw_obj_pred -
        bound``, which is negative exactly when
        ``raw_obj_pred < bound`` — i.e. feasible.

      Posterior CDF evaluation in
      :func:`outcome_constraints.compute_constraint_probability`
      therefore reads the *boundary distance*, preserving gradient
      information near the boundary.
    * ``"binary"`` (legacy): GP fits 1/0 feasibility labels — kept for
      genuinely binary outcomes (pass/fail quality gates).

    Args:
        spec: Optimization specification with outcome_constraints
        observations: Historical observations
        train_x: Training inputs (already encoded)
        bounds: Parameter bounds

    Returns:
        List of (constraint_model, threshold) tuples, or None if no constraints
    """
    if not spec.outcome_constraints:
        return None

    objective_names = {obj.name for obj in spec.objectives}
    method = (spec.outcome_constraint_method or "continuous").lower()
    if method not in {"continuous", "binary"}:
        raise OutcomeConstraintConfigurationError(
            f"Unknown outcome_constraint_method={spec.outcome_constraint_method!r}; "
            "expected 'continuous' or 'binary'."
        )

    constraint_models: list[tuple[Any, float]] = []
    for oc in spec.outcome_constraints:
        if oc.objective_name not in objective_names:
            msg = (
                f"Outcome constraint references objective '{oc.objective_name}', "
                f"which is not declared on the spec (declared objectives: "
                f"{sorted(objective_names)})."
            )
            raise OutcomeConstraintConfigurationError(msg)
        obj_values = _collect_constraint_values(oc.objective_name, observations)
        obj_tensor = torch.tensor(obj_values, dtype=get_dtype(), device=get_device()).unsqueeze(-1)

        constraint_models.append(
            _fit_outcome_constraint_model(oc, obj_tensor, train_x, bounds, method)
        )

    return constraint_models if constraint_models else None


def _collect_constraint_values(
    objective_name: str, observations: list[ObservationData]
) -> list[float]:
    """Pull the constrained-objective column from observations.

    Raises ``OutcomeConstraintConfigurationError`` when any observation is
    missing the named objective — outcome constraints require complete
    coverage so the constraint GP is fit on the same support as the
    objective GP.
    """
    values: list[float] = []
    for obs in observations:
        if objective_name not in obs.objective_values:
            msg = (
                f"Outcome constraint on '{objective_name}' cannot be honored: "
                "at least one observation is missing this objective. Every "
                "observation must record the constrained objective."
            )
            raise OutcomeConstraintConfigurationError(msg)
        values.append(obs.objective_values[objective_name])
    return values


def _fit_outcome_constraint_model(
    oc: Any,
    obj_tensor: Tensor,
    train_x: Tensor,
    bounds: Tensor,
    method: str,
) -> tuple[Any, float]:
    """Fit one outcome-constraint GP and return ``(model, signed_threshold)``.

    Encodes the constraint direction via a sign flip on the training
    targets (for the continuous path) so the acquisition callable
    ``threshold - samples`` lands at the right sign — BoTorch treats
    *negative* output as feasible (see
    :func:`botorch.utils.objective.compute_smoothed_feasibility_indicator`).
    """
    if method == "continuous":
        targets = obj_tensor if oc.greater_than else -obj_tensor
        threshold = float(oc.threshold) if oc.greater_than else -float(oc.threshold)
        constraint_model = create_and_fit_single_task_model(
            train_x, targets, bounds, use_input_warping=False
        )
        return constraint_model, threshold

    # Binary path (legacy)
    if oc.greater_than:
        feasible = (obj_tensor >= oc.threshold).double()
    else:
        feasible = (obj_tensor <= oc.threshold).double()
    constraint_model = create_and_fit_single_task_model(
        train_x, feasible, bounds, use_input_warping=False
    )
    return constraint_model, oc.feasibility_threshold
