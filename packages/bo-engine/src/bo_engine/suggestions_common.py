"""Common helpers shared by the single- and multi-objective batch builders.

Split from :mod:`bo_engine.suggestions` so the small prediction-extraction
and confidence-level helpers used by both
:mod:`bo_engine.suggestions_single_objective` and
:mod:`bo_engine.suggestions_multi_objective` live in a single module
without forcing one to import from the other.
"""

from __future__ import annotations

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


def _get_confidence_level(uncertainty: float | None) -> str:
    """Determine confidence level from uncertainty."""
    if uncertainty is None:
        return "medium"
    if uncertainty < CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD:
        return "high"
    if uncertainty < CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD:
        return "medium"
    return "low"
