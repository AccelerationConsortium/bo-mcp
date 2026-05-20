"""Training-data assembly and pending-point conditioning.

Split from :mod:`bo_engine.suggestions` so the helpers that translate a
list of :class:`~bo_engine.types.ObservationData` into tensors (parameter
matrix, objective matrix, optional per-row noise variance, optional cost
vector) and the pending-point encoder live in their own module. The
acquisition pipelines in
:mod:`bo_engine.suggestions_single_objective` /
:mod:`bo_engine.suggestions_multi_objective` consume these helpers
verbatim — no behavior change.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from bo_engine.device import get_device, get_dtype
from bo_engine.transforms import encode_categorical, stack_encoded_values
from bo_engine.types import ObservationData, OptimizationSpec

logger = logging.getLogger(__name__)


def _prepare_training_data(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> tuple[Tensor, Tensor]:
    """Prepare training data from observations.

    Args:
        observations: List of ObservationData entities
        spec: Optimization specification

    Returns:
        Tuple of (train_x, train_y) tensors
    """
    x_list = []
    y_list = []

    for obs in observations:
        # Encode parameters
        x = encode_categorical(obs.parameter_values, spec)
        x_list.append(x)

        # Get objective values in spec order
        y = torch.tensor(
            [obs.objective_values[obj.name] for obj in spec.objectives],
            dtype=get_dtype(),
            device=get_device(),
        )
        y_list.append(y)

    train_x = torch.stack(x_list)
    train_y = torch.stack(y_list)

    return train_x, train_y


def _prepare_train_yvar(
    observations: list[ObservationData],
    spec: OptimizationSpec,
) -> Tensor | None:
    """Build the per-observation noise-variance tensor in objective order.

    Returns ``None`` when any observation is missing measurement uncertainty
    for any objective — partial coverage falls back to the trainable noise
    path (rather than imputing zeros, which would silently claim the
    uncovered points are noise-free). Returned tensor has shape
    ``(n_observations, n_objectives)`` and units of variance (stddev**2).
    """
    if not observations:
        return None
    objective_names = [obj.name for obj in spec.objectives]
    rows: list[list[float]] = []
    for obs in observations:
        unc = obs.measurement_uncertainty
        if unc is None:
            return None
        row: list[float] = []
        for name in objective_names:
            if name not in unc:
                return None
            stddev = float(unc[name])
            row.append(stddev * stddev)
        rows.append(row)
    return torch.tensor(rows, dtype=get_dtype(), device=get_device())


def _prepare_cost_data(
    observations: list[ObservationData], use_cost_aware: bool = False
) -> Tensor | None:
    """Extract cost data from observations.

    Args:
        observations: List of observations with optional cost field
        use_cost_aware: Whether cost-aware mode was requested (for warning)

    Returns:
        Tensor of costs if all observations have costs, else None
    """
    costs = []
    has_some_costs = False
    for obs in observations:
        if obs.cost is None:
            if has_some_costs and use_cost_aware:
                logger.warning(
                    "Cost-aware mode requested but %d/%d observations lack cost data. "
                    "Falling back to non-cost-aware optimization.",
                    sum(1 for o in observations if o.cost is None),
                    len(observations),
                )
            return None
        has_some_costs = True
        costs.append(obs.cost)
    return torch.tensor(costs, dtype=get_dtype(), device=get_device())


def _encode_pending_points(
    pending_points: list[dict[str, Any]] | None,
    spec: OptimizationSpec,
) -> Tensor | None:
    """Encode pending parameter dicts into the tensor form consumed by BoTorch.

    Returns ``None`` when no pending points are supplied or when every
    candidate is missing at least one spec parameter (encoding would raise
    ``KeyError``).  Malformed entries are skipped with a debug log rather
    than crashing the suggestion path — the acquisition still benefits
    from the subset that encodes cleanly.
    """
    if not pending_points:
        return None
    valid: list[dict[str, Any]] = []
    for params in pending_points:
        if all(p.name in params for p in spec.parameters):
            valid.append(params)
        else:
            logger.debug("Skipping pending point missing required parameter(s): %s", params)
    if not valid:
        return None
    return stack_encoded_values(valid, spec)
