"""Input/output transformations for BO."""

import itertools
import math
from enum import StrEnum
from typing import Any

import torch
from torch import Tensor

from bo_engine.constants import (
    DISCRETE_ENUMERATION_MAX_POINTS,
    MIXED_CATEGORICAL_COMBO_THRESHOLD,
    NUMERICAL_EPSILON,
    SAFE_DIVISION_EPSILON,
)
from bo_engine.device import get_device, get_dtype
from bo_engine.types import OptimizationSpec, ParameterSpec, ParameterType


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
    if has_categorical:
        return SearchSpaceType.MIXED
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
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            has_categorical = True
            product *= len(param.categories)

    return product if has_categorical else 0


def mixed_space_combo_overflow(spec: OptimizationSpec) -> int | None:
    """Categorical-combination count when a mixed space exceeds the BoTorch limit.

    Single source of truth for the two surfaces that enforce
    ``MIXED_CATEGORICAL_COMBO_THRESHOLD`` — intake capability validation
    (``BoTorchBackend.validate_capabilities``) and the acquisition guard in
    ``optimize_acquisition`` — so the limit reported at intake and the limit
    enforced at runtime cannot drift apart.

    Returns ``None`` for any space the mixed optimizer accepts. Specs whose
    categorical parameters lack ``categories`` also return ``None``: this is
    a reporting helper, and the canonical error for that malformed shape is
    raised by the parameter builders, not here.
    """
    if classify_search_space(spec) != SearchSpaceType.MIXED:
        return None
    if any(p.type == ParameterType.CATEGORICAL and p.categories is None for p in spec.parameters):
        return None
    n_combos = count_categorical_combinations(spec)
    return n_combos if n_combos > MIXED_CATEGORICAL_COMBO_THRESHOLD else None


def mixed_space_combo_limit_message(n_combinations: int) -> str:
    """Canonical message for a mixed space above the combination limit.

    Shared verbatim by the intake capability report and the acquisition-time
    ``NotImplementedError`` so both surfaces describe the same limit.
    """
    return (
        f"Mixed spaces with more than {MIXED_CATEGORICAL_COMBO_THRESHOLD} "
        f"categorical combinations are not yet supported by BoTorch acquisition "
        f"(this space has {n_combinations}). Consider reducing the number of "
        "categories or selecting another backend."
    )


def _enumerable_choice_count(param: ParameterSpec) -> int | None:
    """Number of enumerable choices for one parameter, or ``None`` if not enumerable.

    Categorical parameters contribute their category count; discrete
    parameters contribute their explicit ``values`` count or, for
    bounds-only specs, the number of integers in ``[ceil(lo), floor(hi)]``
    (the grid the backends materialize). Continuous parameters — and
    discrete parameters that declare neither values nor bounds, whose
    canonical error is raised by the parameter builders — return ``None``.
    """
    if param.type == ParameterType.CATEGORICAL and param.categories is not None:
        return len(param.categories)
    if param.type == ParameterType.DISCRETE:
        if param.values is not None:
            return len(param.values)
        if param.bounds is not None:
            lo, hi = math.ceil(param.bounds[0]), math.floor(param.bounds[1])
            return max(hi - lo + 1, 0)
    return None


def count_discrete_combinations(spec: OptimizationSpec) -> int:
    """Total Cartesian-product size across discrete and categorical parameters.

    This is the number of rows a backend materializes when it enumerates the
    full discrete portion of the search space (e.g. BayBE's
    ``SearchSpace.from_product``), so the enumeration limit must be enforced
    on this product — a per-parameter check lets two just-under-limit grids
    multiply into an unbuildable frame. Returns 1 when the spec has no
    enumerable parameters.
    """
    product = 1
    for param in spec.parameters:
        n_choices = _enumerable_choice_count(param)
        if n_choices is not None:
            product *= n_choices
    return product


def discrete_enumeration_limit_error(n_combinations: int) -> ValueError:
    """Shared error for a discrete product above ``DISCRETE_ENUMERATION_MAX_POINTS``.

    Both backends raise this same error so a spec rejected for grid size
    reads identically regardless of which backend enumerated it.
    """
    msg = (
        f"Discrete space has {n_combinations} combinations, exceeding the "
        f"enumeration limit of {DISCRETE_ENUMERATION_MAX_POINTS}. "
        "Consider reducing the number of categories or discrete values, "
        "or using continuous parameters."
    )
    return ValueError(msg)


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
        raise discrete_enumeration_limit_error(n_combos)

    device = get_device()
    dtype = get_dtype()

    # Collect category index ranges for each categorical parameter
    cat_ranges: list[range] = []
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            cat_ranges.append(range(len(param.categories)))

    # Pre-compute parameter layout: list of (n_dims, is_categorical) tuples
    param_layout = _get_parameter_layout(spec)

    # Build all combinations
    rows = [_encode_combo(combo, param_layout) for combo in itertools.product(*cat_ranges)]

    return torch.tensor(rows, dtype=dtype, device=device)


def _get_parameter_layout(spec: OptimizationSpec) -> list[tuple[int, bool]]:
    """Return (n_dims, is_categorical) for each parameter."""
    layout: list[tuple[int, bool]] = []
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            layout.append((len(param.categories), True))
        else:
            layout.append((1, False))
    return layout


def _encode_combo(combo: tuple[int, ...], layout: list[tuple[int, bool]]) -> list[float]:
    """Encode a single categorical combination into a flat row."""
    row: list[float] = []
    cat_idx = 0
    for n_dims, is_cat in layout:
        if is_cat:
            one_hot = [0.0] * n_dims
            one_hot[combo[cat_idx]] = 1.0
            row.extend(one_hot)
            cat_idx += 1
        else:
            row.append(0.0)
    return row


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
    cat_info = _collect_categorical_dim_info(spec)
    cat_ranges = [range(n_cats) for _, n_cats in cat_info]

    return [_build_one_hot_features(combo, cat_info) for combo in itertools.product(*cat_ranges)]


def _collect_categorical_dim_info(
    spec: OptimizationSpec,
) -> list[tuple[int, int]]:
    """Return (start_dim, n_cats) for each categorical parameter."""
    cat_info: list[tuple[int, int]] = []
    dim_idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            n_cats = len(param.categories)
            cat_info.append((dim_idx, n_cats))
            dim_idx += n_cats
        else:
            dim_idx += 1
    return cat_info


def _build_one_hot_features(
    combo: tuple[int, ...],
    cat_info: list[tuple[int, int]],
) -> dict[int, float]:
    """Build a single fixed-features dict from a categorical combination."""
    features: dict[int, float] = {}
    for (start_dim, n_cats), cat_choice in zip(cat_info, combo, strict=True):
        for j in range(n_cats):
            features[start_dim + j] = 1.0 if j == cat_choice else 0.0
    return features


def _get_param_bounds(param: ParameterSpec) -> tuple[list[float], list[float]]:
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
            msg = f"Continuous parameter '{param.name}' has no bounds defined"
            raise ValueError(msg)
        return [param.bounds[0]], [param.bounds[1]]

    if param.type == ParameterType.DISCRETE:
        if param.bounds is not None:
            return [param.bounds[0]], [param.bounds[1]]
        if param.values is not None:
            return [float(min(param.values))], [float(max(param.values))]
        # A discrete parameter encodes to exactly one column (see
        # ``_encode_param_value``), so contributing zero bound columns here
        # would silently shift every downstream parameter's bounds. Fail loudly
        # instead, mirroring the continuous/categorical branches.
        msg = f"Discrete parameter '{param.name}' requires bounds or values"
        raise ValueError(msg)

    # ParameterType.CATEGORICAL — one-hot encoding: each category is a dimension in [0, 1]
    if param.categories is None:
        msg = f"Categorical parameter '{param.name}' has no categories defined"
        raise ValueError(msg)
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


def get_categorical_dim_indices(spec: OptimizationSpec) -> list[int]:
    """Return the encoded-space column indices belonging to one-hot blocks.

    The encoded feature space puts each continuous / discrete parameter
    into a single column and each categorical parameter into a block of
    ``len(categories)`` one-hot columns. This helper returns the indices
    of the categorical columns so callers (e.g.
    :func:`bo_engine.models.build_mixed_kernel`) can apply a
    Hamming-style kernel to those dimensions and an RBF kernel to the
    remaining (continuous + discrete) dimensions.

    Empty list when ``spec`` has no categorical parameters.
    """
    indices: list[int] = []
    dim_idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            n_cats = len(param.categories)
            indices.extend(range(dim_idx, dim_idx + n_cats))
            dim_idx += n_cats
        else:
            dim_idx += 1
    return indices


def get_categorical_blocks(spec: OptimizationSpec) -> list[list[int]]:
    """Return the encoded one-hot column indices grouped per categorical parameter.

    Unlike :func:`get_categorical_dim_indices` (which flattens every
    categorical column into one list), this preserves the block boundaries:
    each inner list holds the one-hot columns of a single categorical
    parameter. The boundaries cannot be recovered from the flat list alone
    because two adjacent categorical parameters produce contiguous columns,
    so :func:`bo_engine.models.build_mixed_kernel` consumes the grouped form
    to give each parameter its own ``CategoricalKernel`` (and therefore its
    own shared lengthscale).

    Empty list when ``spec`` has no categorical parameters.
    """
    blocks: list[list[int]] = []
    dim_idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            if param.categories is None:
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            n_cats = len(param.categories)
            blocks.append(list(range(dim_idx, dim_idx + n_cats)))
            dim_idx += n_cats
        else:
            dim_idx += 1
    return blocks


def _encode_param_value(param: ParameterSpec, value: int | float | str) -> list[float]:
    """Encode a single parameter value into its tensor representation.

    Continuous and discrete values become a single float.
    Categorical values become a one-hot encoded list.

    Args:
        param: Parameter specification
        value: The raw parameter value

    Returns:
        List of floats representing this parameter's encoded dimensions

    Raises:
        ValueError: If a categorical value is not one of the declared
            categories. Silently emitting an all-zero one-hot block would
            corrupt the training data (and argmax-decode back to the *first*
            category), so a typo'd value (``'EtOH'`` vs ``'etoh'``) must
            fail loudly here — direct engine callers bypass the MCP intake
            validation.
    """
    if param.type == ParameterType.CATEGORICAL:
        if param.categories is None:
            msg = f"Categorical parameter '{param.name}' has no categories defined"
            raise ValueError(msg)
        if value not in param.categories:
            msg = (
                f"Unknown category {value!r} for parameter '{param.name}'; "
                f"expected one of {list(param.categories)}."
            )
            raise ValueError(msg)
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


def stack_encoded_values(
    value_dicts: list[dict[str, Any]],
    spec: OptimizationSpec,
) -> Tensor:
    """Encode multiple parameter-value dicts and stack into a 2-D tensor.

    This is a convenience wrapper around :func:`encode_categorical` +
    ``torch.stack`` that lets callers avoid importing ``torch`` directly.

    Returns tensor of shape ``(len(value_dicts), n_dims)``.
    """
    tensors = [encode_categorical(v, spec) for v in value_dicts]
    return torch.stack(tensors)


def _decode_param_value(
    param: ParameterSpec, tensor: Tensor, idx: int
) -> tuple[int | float | str, int]:
    """Decode a single parameter value from its tensor representation.

    For continuous parameters, extracts a float.
    For discrete parameters with an explicit ``values`` grid, snaps to the
    nearest allowed value; bounds-only discrete parameters round to the
    nearest integer.
    For categorical parameters, decodes one-hot via ``argmax``.

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
        raw = tensor[idx].item()
        if param.values:
            # The optimizer relaxes discrete dimensions to a continuous box
            # spanning min/max of the grid (see ``_get_param_bounds``), so
            # the result can land between allowed values. Integer rounding
            # would leave a fractional grid entirely (e.g. every point of
            # ``values=[0.1, 0.2, 0.5]`` rounds to 0) — snap to the nearest
            # declared value instead. ``min`` resolves equidistant ties to
            # the earlier entry, deterministically.
            return min(param.values, key=lambda v: abs(v - raw)), idx + 1
        return round(raw), idx + 1

    # ParameterType.CATEGORICAL — pick the argmax over the one-hot block.
    # Softmax is monotone, so ``softmax(x).argmax() == x.argmax()``; the
    # extra transform was a no-op that misled readers about the semantics
    # (no probabilistic sampling happens here) and cost an unnecessary
    # exp/normalize per decode. ``torch.argmax`` resolves ties to the
    # lowest index deterministically — matching the encoder's category
    # ordering — so callers see the canonical category name on ties.
    if param.categories is None:
        msg = f"Categorical parameter '{param.name}' has no categories defined"
        raise ValueError(msg)
    n_cats = len(param.categories)
    cat_values = tensor[idx : idx + n_cats]
    best_cat_idx = int(torch.argmax(cat_values).item())
    return param.categories[best_cat_idx], idx + n_cats


def snap_discrete_columns(points: Tensor, spec: OptimizationSpec | None) -> Tensor:
    """Snap the numeric-discrete columns of encoded points onto their grids.

    Mirrors :func:`_decode_param_value` in tensor form: discrete parameters
    with an explicit ``values`` grid snap to the nearest declared value
    (``argmin`` resolves equidistant ties to the earlier entry, matching the
    decoder), bounds-only discrete parameters round to the nearest integer.
    Continuous and categorical columns pass through unchanged.  With no spec
    the points are returned as-is.

    Args:
        points: Encoded points of shape ``(..., n_dims)``.
        spec: Optimization specification defining the column layout, or None.

    Returns:
        Tensor of the same shape with discrete columns canonicalized.
    """
    if spec is None or not any(p.type == ParameterType.DISCRETE for p in spec.parameters):
        return points
    snapped = points.clone()
    idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            idx += len(param.categories or [])
            continue
        if param.type == ParameterType.DISCRETE:
            column = snapped[..., idx]
            if param.values:
                grid = torch.tensor(param.values, dtype=points.dtype, device=points.device)
                nearest = (column.unsqueeze(-1) - grid).abs().argmin(dim=-1)
                snapped[..., idx] = grid[nearest]
            else:
                snapped[..., idx] = column.round()
        idx += 1
    return snapped


def _numeric_discrete_axis(param: ParameterSpec) -> list[float] | None:
    """Return the finite value axis of one numeric-discrete parameter.

    ``values`` grids are intersected with the declared bounds (a spec may
    carry both, and only in-bounds values are executable candidates);
    bounds-only discrete parameters span the integers inside their bounds
    (the same set decode rounding can produce). ``None`` marks an axis that
    cannot be enumerated. A bounds-only axis wider than
    ``DISCRETE_ENUMERATION_MAX_POINTS`` is reported unenumerable *before*
    the value list is materialized — no consumer may use such an axis
    anyway, so building a huge list first would be pure waste.
    """
    if param.values:
        axis = [float(v) for v in param.values]
        if param.bounds is not None:
            lower, upper = param.bounds
            axis = [v for v in axis if lower <= v <= upper]
        return axis or None
    if param.bounds is not None:
        lower, upper = param.bounds
        axis_size = math.floor(upper) - math.ceil(lower) + 1
        if axis_size <= 0 or axis_size > DISCRETE_ENUMERATION_MAX_POINTS:
            return None
        return [float(v) for v in range(math.ceil(lower), math.floor(upper) + 1)]
    return None


def numeric_discrete_axes(spec: OptimizationSpec) -> list[tuple[int, list[float]]] | None:
    """Return ``(encoded column index, finite axis)`` per numeric-discrete parameter.

    Column indices account for one-hot categorical blocks so they are valid
    for any encoded tensor layout. Returns ``None`` when any discrete
    parameter has no enumerable axis (see :func:`_numeric_discrete_axis`).

    Args:
        spec: Optimization specification.

    Returns:
        List of (column index, axis values) pairs, or None.
    """
    axes: list[tuple[int, list[float]]] = []
    idx = 0
    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            idx += len(param.categories or [])
            continue
        if param.type == ParameterType.DISCRETE:
            axis = _numeric_discrete_axis(param)
            if axis is None:
                return None
            axes.append((idx, axis))
        idx += 1
    return axes


def enumerate_numeric_discrete_grid(spec: OptimizationSpec | None) -> Tensor | None:
    """Enumerate the full Cartesian grid of an all-numeric-discrete spec.

    Returns ``None`` when the grid cannot be enumerated — no spec, any
    non-discrete parameter, an axis without a finite value set, or a product
    larger than ``DISCRETE_ENUMERATION_MAX_POINTS`` — so callers fall back to
    sampling. A returned tensor enumerates the *entire* space, which lets
    callers prove exhaustion instead of inferring it from sample coverage.

    Args:
        spec: Optimization specification, or None.

    Returns:
        Tensor of shape (n_grid_points, n_params), or None.
    """
    if spec is None or not spec.parameters:
        return None
    if any(p.type != ParameterType.DISCRETE for p in spec.parameters):
        return None
    axes: list[list[float]] = []
    total = 1
    for param in spec.parameters:
        axis = _numeric_discrete_axis(param)
        if axis is None:
            return None
        total *= len(axis)
        if total > DISCRETE_ENUMERATION_MAX_POINTS:
            return None
        axes.append(axis)
    rows = list(itertools.product(*axes))
    return torch.tensor(rows, dtype=get_dtype(), device=get_device())


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
    return (x - lower) / (upper - lower + NUMERICAL_EPSILON)


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
    std = torch.where(std < SAFE_DIVISION_EPSILON, torch.ones_like(std), std)
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
                msg = f"Categorical parameter '{param.name}' has no categories defined"
                raise ValueError(msg)
            n_dims += len(param.categories)
        else:
            n_dims += 1
    return n_dims
