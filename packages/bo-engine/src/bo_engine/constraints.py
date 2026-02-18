"""Constraint handling for Bayesian Optimization."""

from collections.abc import Callable

import torch
from torch import Tensor

from bo_engine.types import ConstraintSpec, ConstraintType, OptimizationSpec, ParameterType


def create_constraint_callable(
    constraint: ConstraintSpec,
    spec: OptimizationSpec,
) -> Callable[[Tensor], Tensor]:
    """Create a constraint callable for BoTorch optimization.

    BoTorch constraints expect a callable that returns a tensor where:
    - values >= 0 indicate constraint satisfaction
    - values < 0 indicate constraint violation

    Args:
        constraint: Constraint definition
        spec: Campaign specification

    Returns:
        Callable that takes candidates and returns constraint values
    """
    # Get indices of parameters involved in constraint
    param_indices = _get_parameter_indices(constraint.parameters, spec)

    if constraint.type == ConstraintType.SUM_EQUALS:

        def sum_equals(x: Tensor) -> Tensor:
            # x has shape (..., n_dims)
            selected = x[..., param_indices]
            total = selected.sum(dim=-1)
            # Constraint: sum = value, so |sum - value| <= epsilon
            # Return negative of absolute difference (satisfied when >= 0)
            epsilon = 1e-4
            return epsilon - torch.abs(total - constraint.value)

        return sum_equals

    elif constraint.type == ConstraintType.SUM_LESS_THAN:

        def sum_less_than(x: Tensor) -> Tensor:
            selected = x[..., param_indices]
            total = selected.sum(dim=-1)
            # Constraint: sum < value
            return constraint.value - total

        return sum_less_than

    elif constraint.type == ConstraintType.SUM_GREATER_THAN:

        def sum_greater_than(x: Tensor) -> Tensor:
            selected = x[..., param_indices]
            total = selected.sum(dim=-1)
            # Constraint: sum > value
            return total - constraint.value

        return sum_greater_than

    elif constraint.type == ConstraintType.LINEAR:

        def linear_constraint(x: Tensor) -> Tensor:
            selected = x[..., param_indices]
            coeffs = torch.tensor(constraint.coefficients, dtype=x.dtype, device=x.device)
            # Linear combination
            weighted_sum = (selected * coeffs).sum(dim=-1)
            # Constraint: weighted_sum <= value
            return constraint.value - weighted_sum

        return linear_constraint

    else:
        msg = f"Unknown constraint type: {constraint.type}"
        raise ValueError(msg)


def _get_parameter_indices(
    param_names: list[str],
    spec: OptimizationSpec,
) -> list[int]:
    """Get tensor indices for named parameters.

    Accounts for one-hot encoding of categorical parameters.
    """
    indices = []
    current_idx = 0

    for param in spec.parameters:
        if param.type == ParameterType.CATEGORICAL:
            assert param.categories is not None
            n_cats = len(param.categories)
            if param.name in param_names:
                # Include all one-hot indices for this categorical
                indices.extend(range(current_idx, current_idx + n_cats))
            current_idx += n_cats
        else:
            if param.name in param_names:
                indices.append(current_idx)
            current_idx += 1

    return indices


def apply_sum_constraint(
    candidates: Tensor,
    param_indices: list[int],
    target_sum: float,
) -> Tensor:
    """Apply sum constraint by normalizing selected parameters.

    Projects candidates to satisfy sum(x[param_indices]) = target_sum.

    Args:
        candidates: Candidate tensor of shape (..., n_dims)
        param_indices: Indices of parameters that must sum to target
        target_sum: Target sum value

    Returns:
        Candidates with constraint applied
    """
    result = candidates.clone()
    selected = result[..., param_indices]

    # Normalize to sum to target. For near-zero rows, fall back to
    # an equal split so the constraint is still satisfied.
    current_sum = selected.sum(dim=-1, keepdim=True)
    near_zero = current_sum.abs() < 1e-10
    safe_sum = torch.where(near_zero, torch.ones_like(current_sum), current_sum)
    projected = selected * (target_sum / safe_sum)

    if near_zero.any():
        equal_split = torch.full_like(selected, target_sum / len(param_indices))
        projected = torch.where(near_zero.expand_as(selected), equal_split, projected)

    result[..., param_indices] = projected

    return result


def create_constraints_list(
    spec: OptimizationSpec,
) -> list[Callable[[Tensor], Tensor]]:
    """Create list of constraint callables from campaign spec.

    Args:
        spec: Campaign specification with constraints

    Returns:
        List of constraint callables for BoTorch
    """
    return [create_constraint_callable(c, spec) for c in spec.constraints]
