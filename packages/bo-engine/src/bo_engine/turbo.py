"""TuRBO (Trust Region Bayesian Optimization) implementation.

TuRBO manages a trust region that expands on success and contracts on failure,
enabling efficient optimization in high-dimensional spaces.

Reference: Eriksson et al., "Scalable Global Optimization via Local Bayesian
Optimization", NeurIPS 2019.
"""

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from bo_engine.constants import (
    IMPROVEMENT_TOLERANCE_ABSOLUTE,
    IMPROVEMENT_TOLERANCE_RELATIVE,
    TURBO_CONTRACTION_FACTOR,
    TURBO_EXPANSION_FACTOR,
    TURBO_INITIAL_LENGTH,
    TURBO_LENGTH_MAX,
    TURBO_LENGTH_MIN,
    TURBO_MIN_DIMENSIONS,
    TURBO_SUCCESS_TOLERANCE,
)

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP


@dataclass
class TurboState:
    """Trust region state for TuRBO optimization.

    Attributes:
        dim: Problem dimensionality
        batch_size: Number of suggestions per iteration
        length: Current trust region length (normalized [0,1] space)
        length_min: Minimum length before restart (default: 0.5^7 ~ 0.0078)
        length_max: Maximum length after expansion (default: 1.6)
        failure_counter: Consecutive iterations without improvement
        failure_tolerance: Failures before shrinking
        success_counter: Consecutive improving iterations
        success_tolerance: Successes before expansion (default: 10)
        best_value: Best objective value seen (for maximization)
        restart_triggered: True when length < length_min
    """

    dim: int
    batch_size: int
    length: float = TURBO_INITIAL_LENGTH
    length_min: float = TURBO_LENGTH_MIN
    length_max: float = TURBO_LENGTH_MAX
    failure_counter: int = 0
    failure_tolerance: int | None = None
    success_counter: int = 0
    success_tolerance: int = TURBO_SUCCESS_TOLERANCE
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        """Compute failure_tolerance if not provided."""
        if self.failure_tolerance is None:
            # Default: more tolerance for higher dimensions and smaller batches
            object.__setattr__(
                self,
                "failure_tolerance",
                math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size)),
            )


def create_turbo_state(
    dim: int,
    batch_size: int,
    initial_best_value: float | None = None,
) -> TurboState:
    """Create a new TuRBO state for a problem.

    Args:
        dim: Number of parameters
        batch_size: Batch size for suggestions
        initial_best_value: Best objective value seen so far (for maximization)

    Returns:
        Initialized TurboState
    """
    state = TurboState(dim=dim, batch_size=batch_size)
    if initial_best_value is not None:
        state = replace(state, best_value=initial_best_value)
    return state


def update_turbo_state(state: TurboState, y_next: Tensor) -> TurboState:
    """Update trust region based on new observations.

    The trust region expands after consecutive successes and contracts
    after consecutive failures. This adapts the search region based
    on optimization progress.

    Args:
        state: Current TuRBO state
        y_next: Objective values from latest batch (shape: [batch_size] or [batch_size, 1])
                Values should be for maximization (negate if minimizing)

    Returns:
        Updated TurboState with modified counters and length
    """
    # Get best value from new batch
    y_max = y_next.max().item()

    # Check improvement with tolerance for numerical stability
    tolerance = (
        IMPROVEMENT_TOLERANCE_RELATIVE * abs(state.best_value)
        if state.best_value != 0
        else IMPROVEMENT_TOLERANCE_ABSOLUTE
    )
    improved = y_max > state.best_value + tolerance

    # Update counters
    if improved:
        new_success_counter = state.success_counter + 1
        new_failure_counter = 0
    else:
        new_success_counter = 0
        new_failure_counter = state.failure_counter + 1

    # Initialize new state values
    new_length = state.length
    new_best_value = max(state.best_value, y_max)
    final_success_counter = new_success_counter
    final_failure_counter = new_failure_counter

    # Expand trust region on sustained success
    if new_success_counter >= state.success_tolerance:
        new_length = min(TURBO_EXPANSION_FACTOR * state.length, state.length_max)
        final_success_counter = 0

    # Contract trust region on sustained failure
    elif state.failure_tolerance is not None and new_failure_counter >= state.failure_tolerance:
        new_length = state.length / TURBO_CONTRACTION_FACTOR
        final_failure_counter = 0

    # Check for restart trigger
    restart_triggered = new_length < state.length_min

    return replace(
        state,
        length=new_length,
        success_counter=final_success_counter,
        failure_counter=final_failure_counter,
        best_value=new_best_value,
        restart_triggered=restart_triggered,
    )


def get_turbo_bounds(
    state: TurboState,
    train_x: Tensor,
    train_y: Tensor,
    model: "SingleTaskGP",
) -> tuple[Tensor, Tensor]:
    """Compute trust region bounds centered on best point.

    The trust region is centered on the best observed point and scaled
    by the lengthscales learned by the GP model. This focuses the
    search on promising regions while respecting the model's learned
    structure.

    Args:
        state: Current TuRBO state
        train_x: All training inputs (normalized to [0,1])
        train_y: All training outputs (for maximization)
        model: Fitted GP model (for lengthscale extraction)

    Returns:
        Tuple of (lower_bounds, upper_bounds) tensors of shape [dim]
    """
    # Center on best observed point
    best_idx = int(train_y.argmax().item())
    x_center = train_x[best_idx].clone()

    # Extract lengthscales from model
    # Handle different kernel structures (ScaleKernel wraps base_kernel, or direct RBF)
    covar = model.covar_module
    if hasattr(covar, "base_kernel"):
        # ScaleKernel case
        lengthscales = covar.base_kernel.lengthscale.squeeze().detach()
    else:
        # Direct RBF kernel case
        lengthscales = covar.lengthscale.squeeze().detach()

    # Handle scalar lengthscale (isotropic kernel)
    if lengthscales.dim() == 0:
        lengthscales = lengthscales.unsqueeze(0).expand(state.dim)

    # Normalize weights (geometric mean = 1)
    weights = lengthscales / lengthscales.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    # Compute bounds with lengthscale-weighted radius
    half_width = weights * state.length / 2.0
    tr_lb = torch.clamp(x_center - half_width, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + half_width, 0.0, 1.0)

    return tr_lb, tr_ub


def should_use_turbo(n_parameters: int, threshold: int = TURBO_MIN_DIMENSIONS) -> bool:
    """Determine if TuRBO should be used based on problem dimensionality.

    TuRBO is most beneficial for high-dimensional problems where standard
    BO struggles with the curse of dimensionality.

    Args:
        n_parameters: Number of parameters in the optimization problem
        threshold: Minimum dimensions for TuRBO (default: 20)

    Returns:
        True if TuRBO is recommended
    """
    return n_parameters >= threshold
