"""Input/output transformations for BO."""

import itertools
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from bo_engine.constants import DISCRETE_ENUMERATION_MAX_POINTS
from bo_engine.device import get_device, get_dtype
from bo_engine.types import OptimizationSpec, ParameterType


class SearchSpaceType(StrEnum):
    """Classification of the optimization search space."""

    CONTINUOUS = "continuous"
    PURELY_CATEGORICAL = "purely_categorical"
    MIXED = "mixed"


def classify_search_space(spec: OptimizationSpec) -> SearchSpaceType:
    """Classify the search space based on parameter types.

    A spec is PURELY_CATEGORICAL when all parameters are categorical,
    CONTINUOUS when none are categorical, and MIXED otherwise.
    ParameterType.DISCRETE (integer-valued numeric) is treated as continuous
    since it uses the standard L-BFGS optimization path.

    Args:
        spec: Optimization specification

    Returns:
        SearchSpaceType classification
    """
    has_categorical = any(p.type == ParameterType.CATEGORICAL for p in spec.parameters)
    all_categorical = all(p.type == ParameterType.CATEGORICAL for p in spec.parameters)

    if all_categorical:
        return SearchSpaceType.PURELY_CATEGORICAL
    elif has_categorical:
        return SearchSpaceType.MIXED
    else:
        return SearchSpaceType.CONTINUOUS


def count_categorical_combinations(spec: OptimizationSpec) -> int:
    """Count the total number of categorical combinations.

    Returns the product of len(param.categories) across all categorical
    parameters. Returns 0 if there are no categorical parameters.

    Args:
        spec: Optimization specification

    Returns:
        Number of categorical combinations, or 0 if no categoricals
    """
    product = 1
    has_categorical = False

    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
            has_categorical = True
            product *= len(param.categories)

    return product if has_categorical else 0


def enumerate_discrete_choices(spec: OptimizationSpec) -> Tensor:
    """Build a tensor of all one-hot-encoded categorical combinations.

    Creates a (num_combos, n_dims) tensor for use with optimize_acqf_discrete.
    Uses itertools.product over category indices, then one-hot encodes each
    combination.

    Raises ValueError if the number of combinations exceeds
    DISCRETE_ENUMERATION_MAX_POINTS.

    Args:
        spec: Optimization specification (must be purely categorical)

    Returns:
        Tensor of shape (num_combos, n_dims) with all discrete choices

    Raises:
        ValueError: If the number of combinations exceeds the enumeration limit
    """
    n_combos = count_categorical_combinations(spec)
    if n_combos > DISCRETE_ENUMERATION_MAX_POINTS:
        raise ValueError(
            f"Discrete space has {n_combos} combinations, exceeding the "
            f"enumeration limit of {DISCRETE_ENUMERATION_MAX_POINTS}. "
            "Consider reducing the number of categories."
        )

    device = get_device()
    dtype = get_dtype()

    # Collect category index ranges for each categorical parameter
    cat_ranges: list[range] = []
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
            cat_ranges.append(range(len(param.categories)))

    # Build all combinations
    rows = []
    for combo in itertools.product(*cat_ranges):
        row = []
        cat_idx = 0
        for param in spec.parameters:
            if param.type == ParameterType.CATEGORICAL:
                if param.categories is None:
                    raise ValueError(
                        f"Categorical parameter '{param.name}' has no categories defined"
                    )
                n_cats = len(param.categories)
                one_hot = [0.0] * n_cats
                one_hot[combo[cat_idx]] = 1.0
                row.extend(one_hot)
                cat_idx += 1
            else:
                # Non-categorical dims get 0.0 placeholder
                row.append(0.0)

        rows.append(row)

    return torch.tensor(rows, dtype=dtype, device=device)


def build_fixed_features_list(
    spec: OptimizationSpec,
) -> list[dict[int, float]]:
    """Build fixed_features_list for optimize_acqf_mixed.

    Enumerates all combinations of categorical parameters as one-hot encodings
    and returns a list of dicts mapping dimension index to fixed value. Only
    categorical dimension indices appear in the dicts — continuous/discrete
    dimensions are omitted so the mixed optimizer can optimize over them.

    Args:
        spec: Optimization specification with at least one categorical parameter

    Returns:
        List of dicts, each mapping dimension index to fixed one-hot value
    """
    # Collect categorical parameter info: (start_dim, n_cats) pairs
    cat_info: list[tuple[int, int]] = []
    dim_idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
            n_cats = len(param.categories)
            cat_info.append((dim_idx, n_cats))
            dim_idx += n_cats
        else:
            dim_idx += 1

    # Build all combinations of category indices
    cat_ranges = [range(n_cats) for _, n_cats in cat_info]

    fixed_features_list: list[dict[int, float]] = []
    for combo in itertools.product(*cat_ranges):
        features: dict[int, float] = {}
        for (start_dim, n_cats), cat_choice in zip(cat_info, combo, strict=True):
            for j in range(n_cats):
                features[start_dim + j] = 1.0 if j == cat_choice else 0.0
        fixed_features_list.append(features)

    return fixed_features_list


def _get_param_bounds(param: Any) -> tuple[list[float], list[float]]:
    """Get lower and upper bounds for a single parameter.

    For continuous parameters, returns the explicit bounds.
    For discrete parameters, returns bounds or min/max of values.
    For categorical parameters, returns [0, 1] per category (one-hot encoding).

    Args:
        param: Parameter specification

    Returns:
        Tuple of (lower_bounds, upper_bounds) lists for this parameter's dimensions
    """
    if param.type == ParameterType.CONTINUOUS:
        if param.bounds is None:
            raise ValueError(f"Continuous parameter '{param.name}' has no bounds defined")
        return [param.bounds[0]], [param.bounds[1]]

    if param.type == ParameterType.DISCRETE:
        if param.bounds is not None:
            return [param.bounds[0]], [param.bounds[1]]
        if param.values is not None:
            return [float(min(param.values))], [float(max(param.values))]
        return [], []

    # ParameterType.CATEGORICAL — one-hot encoding: each category is a dimension in [0, 1]
    if param.categories is None:
        raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
    n_cats = len(param.categories)
    return [0.0] * n_cats, [1.0] * n_cats


def get_bounds_tensor(spec: OptimizationSpec) -> Tensor:
    """Get bounds tensor for continuous/discrete parameters.

    Returns tensor of shape (2, n_dims) where n_dims includes one-hot encoded
    categorical dimensions.
    """
    lower: list[float] = []
    upper: list[float] = []

    for param in spec.parameters:
        lo, hi = _get_param_bounds(param)
        lower.extend(lo)
        upper.extend(hi)

    return torch.tensor([lower, upper], dtype=get_dtype(), device=get_device())


def _encode_param_value(param: Any, value: Any) -> list[float]:
    """Encode a single parameter value into its tensor representation.

    Continuous and discrete values become a single float.
    Categorical values become a one-hot encoded list.

    Args:
        param: Parameter specification
        value: The raw parameter value

    Returns:
        List of floats representing this parameter's encoded dimensions
    """
    if param.type == ParameterType.CATEGORICAL:
        if param.categories is None:
            raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
        return [1.0 if value == cat else 0.0 for cat in param.categories]
    return [float(value)]


def encode_categorical(values: dict[str, Any], spec: OptimizationSpec) -> Tensor:
    """Encode parameter values to tensor, including one-hot for categoricals.

    Returns tensor of shape (n_dims,).
    """
    encoded: list[float] = []
    for param in spec.parameters:
        encoded.extend(_encode_param_value(param, values[param.name]))
    return torch.tensor(encoded, dtype=get_dtype(), device=get_device())


def _decode_param_value(param: Any, tensor: Tensor, idx: int) -> tuple[Any, int]:
    """Decode a single parameter value from its tensor representation.

    For continuous parameters, extracts a float.
    For discrete parameters, extracts and rounds to the nearest integer.
    For categorical parameters, decodes one-hot via softmax and argmax.

    Args:
        param: Parameter specification
        tensor: Full encoded tensor
        idx: Current index into the tensor

    Returns:
        Tuple of (decoded_value, new_index)
    """
    if param.type == ParameterType.CONTINUOUS:
        return float(tensor[idx].item()), idx + 1

    if param.type == ParameterType.DISCRETE:
        return round(tensor[idx].item()), idx + 1

    # ParameterType.CATEGORICAL — decode one-hot via softmax then argmax.
    # Softmax sharpens the encoding so ties from continuous relaxation
    # are resolved deterministically.
    if param.categories is None:
        raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
    n_cats = len(param.categories)
    cat_values = tensor[idx : idx + n_cats]
    sharpened = torch.softmax(cat_values, dim=0)
    best_cat_idx = int(sharpened.argmax().item())
    return param.categories[best_cat_idx], idx + n_cats


def decode_categorical(tensor: Tensor, spec: OptimizationSpec) -> dict[str, Any]:
    """Decode tensor back to parameter values.

    Tensor should be of shape (n_dims,).
    """
    values = {}
    idx = 0

    for param in spec.parameters:
        values[param.name], idx = _decode_param_value(param, tensor, idx)

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
            if param.categories is None:
                raise ValueError(f"Categorical parameter '{param.name}' has no categories defined")
            n_dims += len(param.categories)
        else:
            n_dims += 1
    return n_dims
