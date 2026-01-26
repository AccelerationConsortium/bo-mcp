"""Input/output transformations for BO."""

from typing import Any

import torch
from torch import Tensor

from bo_engine.device import get_device, get_dtype
from bo_engine.types import OptimizationSpec, ParameterType


def get_bounds_tensor(spec: OptimizationSpec) -> Tensor:
    """Get bounds tensor for continuous/discrete parameters.

    Returns tensor of shape (2, n_dims) where n_dims includes one-hot encoded
    categorical dimensions.
    """
    lower = []
    upper = []

    for param in spec.parameters:
        if param.type == ParameterType.CONTINUOUS:
            assert param.bounds is not None
            lower.append(param.bounds[0])
            upper.append(param.bounds[1])
        elif param.type == ParameterType.DISCRETE:
            if param.bounds is not None:
                lower.append(param.bounds[0])
                upper.append(param.bounds[1])
            elif param.values is not None:
                lower.append(float(min(param.values)))
                upper.append(float(max(param.values)))
        elif param.type == ParameterType.CATEGORICAL:
            # One-hot encoding: each category is a dimension in [0, 1]
            assert param.categories is not None
            for _ in param.categories:
                lower.append(0.0)
                upper.append(1.0)

    return torch.tensor([lower, upper], dtype=get_dtype(), device=get_device())


def encode_categorical(values: dict[str, Any], spec: OptimizationSpec) -> Tensor:
    """Encode parameter values to tensor, including one-hot for categoricals.

    Returns tensor of shape (n_dims,).
    """
    encoded = []

    for param in spec.parameters:
        value = values[param.name]

        if param.type == ParameterType.CONTINUOUS:
            encoded.append(float(value))
        elif param.type == ParameterType.DISCRETE:
            encoded.append(float(value))
        elif param.type == ParameterType.CATEGORICAL:
            # One-hot encoding
            assert param.categories is not None
            for cat in param.categories:
                encoded.append(1.0 if value == cat else 0.0)

    return torch.tensor(encoded, dtype=get_dtype(), device=get_device())


def decode_categorical(tensor: Tensor, spec: OptimizationSpec) -> dict[str, Any]:
    """Decode tensor back to parameter values.

    Tensor should be of shape (n_dims,).
    """
    values = {}
    idx = 0

    for param in spec.parameters:
        if param.type == ParameterType.CONTINUOUS:
            values[param.name] = float(tensor[idx].item())
            idx += 1
        elif param.type == ParameterType.DISCRETE:
            # Round to nearest integer for discrete
            values[param.name] = round(tensor[idx].item())
            idx += 1
        elif param.type == ParameterType.CATEGORICAL:
            # Decode one-hot: pick highest value
            assert param.categories is not None
            n_cats = len(param.categories)
            cat_values = tensor[idx : idx + n_cats]
            best_cat_idx = int(cat_values.argmax().item())
            values[param.name] = param.categories[best_cat_idx]
            idx += n_cats

    return values


def normalize_inputs(x: Tensor, bounds: Tensor) -> Tensor:
    """Normalize inputs to [0, 1] based on bounds.

    Args:
        x: Input tensor of shape (..., n_dims)
        bounds: Bounds tensor of shape (2, n_dims)

    Returns:
        Normalized tensor of shape (..., n_dims)
    """
    lower = bounds[0]
    upper = bounds[1]
    return (x - lower) / (upper - lower + 1e-10)


def unnormalize_inputs(x_normalized: Tensor, bounds: Tensor) -> Tensor:
    """Unnormalize inputs from [0, 1] back to original scale.

    Args:
        x_normalized: Normalized tensor of shape (..., n_dims)
        bounds: Bounds tensor of shape (2, n_dims)

    Returns:
        Unnormalized tensor of shape (..., n_dims)
    """
    lower = bounds[0]
    upper = bounds[1]
    return x_normalized * (upper - lower) + lower


def standardize_outputs(y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Standardize outputs to zero mean and unit variance.

    Args:
        y: Output tensor of shape (n_samples, n_objectives)

    Returns:
        Tuple of (standardized_y, mean, std)
    """
    mean = y.mean(dim=0)
    std = y.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return (y - mean) / std, mean, std


def unstandardize_outputs(y_standardized: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """Unstandardize outputs back to original scale.

    Args:
        y_standardized: Standardized tensor
        mean: Mean used for standardization
        std: Std used for standardization

    Returns:
        Unstandardized tensor
    """
    return y_standardized * std + mean


def get_n_dims(spec: OptimizationSpec) -> int:
    """Get total number of dimensions including one-hot encoded categoricals."""
    n_dims = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            assert param.categories is not None
            n_dims += len(param.categories)
        else:
            n_dims += 1
    return n_dims
