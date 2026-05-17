"""Constraint handling for Bayesian Optimization."""

from collections.abc import Callable

import torch
from torch import Tensor

from bo_engine.spec_ir import ConstraintTargetClass, classify_constraint_target
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


def build_botorch_linear_constraints(
    spec: OptimizationSpec,
) -> tuple[
    list[tuple[Tensor, Tensor, float]],
    list[tuple[Tensor, Tensor, float]],
    list[ConstraintSpec],
]:
    """Convert ConstraintSpecs to BoTorch's native linear constraint format.

    BoTorch's optimize_acqf accepts:
    - inequality_constraints: list of (indices, coefficients, rhs) where Ax <= b
    - equality_constraints: list of (indices, coefficients, rhs) where Ax = b

    Constraints involving categorical (one-hot) parameters cannot be expressed
    as simple linear constraints and are returned separately for post-hoc projection.

    Args:
        spec: Optimization specification with constraints

    Returns:
        Tuple of (inequality_constraints, equality_constraints, projection_constraints)
        where projection_constraints need post-hoc handling.

    Reference:
        https://botorch.org/api/optim.html#botorch.optim.optimize.optimize_acqf
    """
    inequality_constraints: list[tuple[Tensor, Tensor, float]] = []
    equality_constraints: list[tuple[Tensor, Tensor, float]] = []
    projection_constraints: list[ConstraintSpec] = []

    for constraint in spec.constraints:
        indices = _get_parameter_indices(constraint.parameters, spec)

        # Any constraint that touches a categorical parameter cannot be
        # expressed as a native BoTorch linear constraint (the parameter
        # is one-hot encoded). The shared classifier returns CATEGORICAL
        # for both all-categorical and mixed-with-categorical cases —
        # the same dispatch BayBE uses to refuse those constraints.
        if (
            classify_constraint_target(constraint, spec.parameters)
            == ConstraintTargetClass.CATEGORICAL
        ):
            projection_constraints.append(constraint)
            continue

        idx_tensor = torch.tensor(indices, dtype=torch.long)

        if constraint.type == ConstraintType.SUM_LESS_THAN:
            # sum(x[indices]) <= value  →  Ax <= b with A=1, b=value
            coeffs = torch.ones(len(indices), dtype=torch.double)
            inequality_constraints.append((idx_tensor, coeffs, constraint.value))

        elif constraint.type == ConstraintType.SUM_GREATER_THAN:
            # sum(x[indices]) >= value  →  -sum(x[indices]) <= -value
            coeffs = -torch.ones(len(indices), dtype=torch.double)
            inequality_constraints.append((idx_tensor, coeffs, -constraint.value))

        elif constraint.type == ConstraintType.SUM_EQUALS:
            # Equality: sum of selected params must equal the target value
            coeffs = torch.ones(len(indices), dtype=torch.double)
            equality_constraints.append((idx_tensor, coeffs, constraint.value))

        elif constraint.type == ConstraintType.LINEAR:
            # The intake-layer ``Constraint`` validator (bo_mcp_server.domain)
            # rejects ``type=linear`` without coefficients, so a missing
            # ``constraint.coefficients`` reaching this point indicates a
            # third-party caller bypassing intake. Surface that as a hard
            # error rather than silently rewriting the constraint into an
            # unweighted sum.
            if constraint.coefficients is None:
                msg = (
                    "Linear constraint reached the engine without "
                    "coefficients. Intake validation should have rejected "
                    "this; check that the caller is going through "
                    "CampaignSpec / IntakeData."
                )
                raise ValueError(msg)
            # coefficients @ x[indices] <= value
            coeffs = torch.tensor(constraint.coefficients, dtype=torch.double)
            inequality_constraints.append((idx_tensor, coeffs, constraint.value))

        else:
            projection_constraints.append(constraint)

    return inequality_constraints, equality_constraints, projection_constraints
