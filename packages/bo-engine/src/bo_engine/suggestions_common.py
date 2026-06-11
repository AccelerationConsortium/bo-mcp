"""Common helpers shared by the single- and multi-objective batch builders.

Split from :mod:`bo_engine.suggestions` so the small prediction-extraction
and confidence-level helpers used by both
:mod:`bo_engine.suggestions_single_objective` and
:mod:`bo_engine.suggestions_multi_objective` live in a single module
without forcing one to import from the other.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from bo_engine.constants import (
    CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD,
    CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD,
)

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP

    GPModel = SingleTaskGP | ModelListGP


def _get_model_predictions(model: GPModel, candidates: Tensor) -> tuple[Tensor, Tensor]:
    """Get model predictions (mean and std) at candidate points.

    Args:
        model: Fitted GP model
        candidates: Candidate points to evaluate

    Returns:
        Tuple of (means, stds) where each has shape matching the model output
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(candidates)
        means = posterior.mean.squeeze(-1)
        variances = posterior.variance.squeeze(-1)
        if means.dim() == 0:
            means = means.unsqueeze(0)
        if variances.dim() == 0:
            variances = variances.unsqueeze(0)
        stds = variances.sqrt()
    return means, stds


def _extract_scalar_prediction(
    tensor: Tensor,
    index: int,
) -> float | None:
    """Safely extract a scalar from a 0-d or 1-d tensor at the given index.

    Returns None if the index is out of bounds.

    Args:
        tensor: Tensor of acquisition values, means, or stds
        index: Batch index

    Returns:
        Float value or None
    """
    if tensor.dim() == 0:
        return tensor.item() if index == 0 else None
    return tensor[index].item() if index < tensor.numel() else None


def _normalize_uncertainty(uncertainty: float | None, scale: float) -> float | None:
    """Express a raw-scale posterior std in standardized-target units.

    The candidate posterior std is reported in the user's raw objective
    units because the GP's ``Standardize`` outcome transform un-standardizes
    the posterior. Dividing by ``scale`` — the model's ``Standardize.stdvs``
    (the training-data scale) — recovers a dimensionless std so the
    confidence thresholds in :mod:`bo_engine.constants` are *relative* and
    the verdict is invariant under rescaling the objective. A non-finite or
    non-positive scale carries no usable information, so the value is left
    unchanged.

    Args:
        uncertainty: Raw-scale posterior std at the candidate, or ``None``.
        scale: The model's ``Standardize.stdvs`` for the objective.

    Returns:
        The scale-relative std, or ``None`` if ``uncertainty`` is ``None``.
    """
    if uncertainty is None:
        return None
    if not math.isfinite(scale) or scale <= 0.0:
        return uncertainty
    return uncertainty / scale


def _mean_relative_uncertainty(
    stds: Tensor,
    index: int,
    scales: Sequence[float],
) -> float | None:
    """Average a candidate's per-objective stds after normalizing each by its scale.

    Mirrors the raw averaging used for the user-facing ``model_uncertainty``
    field but divides each objective's std by that objective's own
    ``Standardize.stdvs`` before averaging, so the confidence verdict is
    invariant to rescaling any single objective rather than only a uniform
    rescaling of all of them.

    Args:
        stds: Posterior std predictions (batch x n_objectives), or a 1-D
            tensor for a single objective.
        index: Candidate index in the batch.
        scales: Per-objective ``Standardize.stdvs`` values.

    Returns:
        Mean scale-relative std across objectives, or ``None`` when the
        candidate index is out of range.
    """
    if stds.dim() > 1 and index < stds.shape[0]:
        per_objective = stds[index]
    elif index < stds.numel():
        return _normalize_uncertainty(stds[index].item(), scales[0] if scales else 1.0)
    else:
        return None

    relative: list[float] = []
    for objective_index in range(per_objective.numel()):
        scale = scales[objective_index] if objective_index < len(scales) else 1.0
        normalized = _normalize_uncertainty(per_objective[objective_index].item(), scale)
        if normalized is not None:
            relative.append(normalized)
    if not relative:
        return None
    return sum(relative) / len(relative)


def _get_confidence_level(relative_uncertainty: float | None) -> str:
    """Map a scale-relative posterior std to a confidence label.

    ``relative_uncertainty`` must already be normalized by the model's
    ``Standardize.stdvs`` (see :func:`_normalize_uncertainty`) so the
    thresholds from :mod:`bo_engine.constants` are dimensionless and the
    label is invariant to the objective's numeric scale.
    """
    if relative_uncertainty is None:
        return "medium"
    if relative_uncertainty < CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD:
        return "high"
    if relative_uncertainty < CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD:
        return "medium"
    return "low"
