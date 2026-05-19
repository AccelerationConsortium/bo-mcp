"""TuRBO (Trust Region Bayesian Optimization) implementation.

TuRBO manages a trust region that expands on success and contracts on failure,
enabling efficient optimization in high-dimensional spaces.

Sign convention: ``update_turbo_state`` accepts **raw** objective values and
takes an explicit ``minimize`` flag; it then converts to TuRBO's internal
maximization convention (``best_value`` tracks the largest "better-is-higher"
value).  Every helper that consumes model-space data (``train_y``,
``best_value``) operates in that internal form.  See the canonical
minimization-form convention in :mod:`bo_engine.types` and note that this
module's internal form differs by a sign.

Scale assumption: every default tolerance in this module is calibrated to
**unit-standardized targets** — i.e. the BoTorch ``Standardize(m=1)`` outcome
transform leaves the GP's training targets at mean≈0 and std≈1. ``best_value``
itself stays on the raw (unstandardized) scale because users care about the
raw improvement story, but the improvement-detection threshold inside
``update_turbo_state`` is a small relative fraction of ``best_value`` and so
implicitly expects ``best_value`` to lie within a couple of orders of
magnitude of unit scale. Pair-with-1.42: if the user supplies targets several
orders of magnitude off unit scale, call :func:`assert_unit_scale_targets`
before constructing a ``TurboState`` to warn loudly instead of silently
mis-firing the expand/contract dynamics.

Reference: Eriksson et al., "Scalable Global Optimization via Local Bayesian
Optimization", NeurIPS 2019.
"""

import math
import warnings
from dataclasses import dataclass
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
    TURBO_MAX_FAILURE_TOLERANCE,
    TURBO_MIN_DIMENSIONS,
    TURBO_SUCCESS_TOLERANCE,
    TURBO_UNIT_SCALE_MEAN_ABS_MAX,
    TURBO_UNIT_SCALE_STD_MAX,
    TURBO_UNIT_SCALE_STD_MIN,
)

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP


@dataclass
class TurboState:
    """Trust region state for TuRBO optimization.

    Sign convention: ``best_value`` is tracked in TuRBO's internal
    maximization convention — larger is better, regardless of the
    user-facing ``minimize`` direction.  ``update_turbo_state`` performs
    the conversion from raw objective values using its ``minimize``
    argument, and ``get_turbo_bounds`` centres the trust region on
    ``train_y.argmax()`` in the same convention.  Callers that build a
    ``TurboState`` by hand (e.g. tests, state deserialization) must
    therefore seed ``best_value`` with an already-negated value when the
    user-facing objective is minimized.

    Scale assumption: ``success_tolerance``, ``failure_tolerance``,
    ``length_min``, and ``length_max`` are calibrated for **unit-standardized**
    objectives (BoTorch's ``Standardize(m=1)`` produces mean≈0, std≈1).
    Concretely:

    * ``success_tolerance`` (Eriksson et al., 2019, Algorithm 1): number of
      consecutive batches whose best raw observation exceeds
      ``best_value + IMPROVEMENT_TOLERANCE_RELATIVE * abs(best_value)`` before
      the trust region doubles. With unit-scale targets the threshold is
      ≈1e-3 absolute and the cadence matches the paper's reported behaviour;
      with a target scale of 1e6 the same relative threshold becomes 1e3 in
      raw units and "improvements" smaller than typical sensor noise are
      treated as successes, expanding the trust region prematurely.
    * ``failure_tolerance`` (Eriksson et al., 2019, §3.2): consecutive
      non-improving batches before the trust region halves. The default
      ``ceil(max(4/batch, dim/batch))`` scales with ``dim/batch`` so high-D
      campaigns get more rope before contracting (capped at
      ``TURBO_MAX_FAILURE_TOLERANCE`` to guarantee eventual restarts even at
      ``dim=1000``).
    * ``length_min`` (default ``0.5**7`` ≈ 7.8e-3) and ``length_max`` (default
      ``1.6``) live in normalized [0,1] input space, so they are independent
      of the **target** scale but do assume that the input transform
      (``Normalize``) has been applied. Override via :class:`TurboConfig` for
      campaigns whose input geometry differs from the unit hypercube.

    Use :func:`assert_unit_scale_targets` at construction time to warn when
    the supplied training targets are far from unit scale; the warning
    references this docstring and points at
    :class:`bo_engine.types.OutcomeTransformSpec` (paired with TODO 1.42).

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
        best_value: Best observed value in TuRBO's internal maximization
            convention (always "larger is better")
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
        """Compute failure_tolerance if not provided and enforce invariants.

        The invariants mirror :class:`bo_mcp_server.domain.TurboConfig`'s
        Pydantic validators so the engine refuses garbage even when callers
        construct ``TurboState`` directly (tests, state deserialization,
        non-MCP backends). Defense in depth: a bad ``length_min``/
        ``length_max`` pair silently breaks the expand/contract dynamics, so
        we'd rather raise at construction than chase the failure into the
        acquisition loop.

        Note on the ``length`` band: ``update_turbo_state`` legitimately
        produces a state with ``length < length_min`` after the final
        contraction step — that *is* the restart signal. We therefore
        enforce ``length <= length_max`` unconditionally (expansion is
        always clamped) but accept ``length < length_min`` only when
        ``restart_triggered`` is set, which is exactly the post-contraction
        case. A direct construction with ``length < length_min`` and
        ``restart_triggered=False`` is incoherent and rejected.
        """
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.length_min <= 0:
            raise ValueError(f"length_min must be positive, got {self.length_min}")
        if self.length_max <= 0:
            raise ValueError(f"length_max must be positive, got {self.length_max}")
        if self.length_min >= self.length_max:
            raise ValueError(
                f"length_min ({self.length_min}) must be strictly less than "
                f"length_max ({self.length_max})"
            )
        if self.length > self.length_max:
            raise ValueError(
                f"length ({self.length}) must not exceed length_max "
                f"({self.length_max}); expansion clamps at length_max."
            )
        if self.length < self.length_min and not self.restart_triggered:
            raise ValueError(
                f"length ({self.length}) is below length_min ({self.length_min}) "
                "but restart_triggered is False; the only legitimate "
                "below-min state is the post-contraction restart signal."
            )
        if self.success_tolerance < 1:
            raise ValueError(f"success_tolerance must be >= 1, got {self.success_tolerance}")

        if self.failure_tolerance is None:
            # Default: more tolerance for higher dimensions and smaller batches,
            # capped at TURBO_MAX_FAILURE_TOLERANCE to ensure restarts happen.
            raw = math.ceil(max(4.0 / self.batch_size, self.dim / self.batch_size))
            object.__setattr__(
                self,
                "failure_tolerance",
                min(raw, TURBO_MAX_FAILURE_TOLERANCE),
            )
        elif self.failure_tolerance < 1:
            raise ValueError(f"failure_tolerance must be >= 1, got {self.failure_tolerance}")


def assert_unit_scale_targets(train_y: Tensor) -> None:
    """Warn when ``train_y`` is far enough from unit scale to mis-fire TuRBO.

    See :class:`TurboState` for the rationale: every default expand/contract
    threshold in :func:`update_turbo_state` is calibrated assuming raw
    objective values sit within a couple of orders of magnitude of unit scale,
    matching what ``Standardize(m=1)`` produces. When the raw targets exceed
    those bounds the relative improvement threshold lands either deep in the
    sensor-noise floor (large scale) or above realistic step sizes (tiny
    scale).

    Emits :class:`UserWarning` with a recommendation to apply an outcome
    transform; the warning is non-fatal so existing campaigns continue to run.

    Args:
        train_y: Raw (unstandardized) objective values, shape ``(n,)`` or
            ``(n, 1)``. Empty or single-row tensors are ignored — TuRBO is
            built for the bulk-data regime and a 0/1-sample window is not
            informative enough to warn about.
    """
    if train_y.numel() < 2:
        return
    flat = train_y.detach().reshape(-1).to(dtype=torch.float64)
    mean_abs = float(flat.mean().abs().item())
    std = float(flat.std(unbiased=True).item())

    if math.isnan(mean_abs) or math.isnan(std):
        return

    mean_bad = mean_abs > TURBO_UNIT_SCALE_MEAN_ABS_MAX
    std_bad = std < TURBO_UNIT_SCALE_STD_MIN or std > TURBO_UNIT_SCALE_STD_MAX
    if not (mean_bad or std_bad):
        return

    warnings.warn(
        (
            "TuRBO defaults are calibrated for unit-standardized targets "
            f"(|mean|<={TURBO_UNIT_SCALE_MEAN_ABS_MAX}, "
            f"std in [{TURBO_UNIT_SCALE_STD_MIN}, {TURBO_UNIT_SCALE_STD_MAX}]) "
            f"but train_y has |mean|={mean_abs:.3g}, std={std:.3g}. The "
            "expand/contract tolerances may mis-fire — apply an outcome "
            "transform (BoTorch Standardize / log_standardize) or set explicit "
            "success_tolerance / failure_tolerance via TurboConfig. See "
            "TurboState's docstring for the scale assumption."
        ),
        UserWarning,
        stacklevel=2,
    )


def create_turbo_state(
    dim: int,
    batch_size: int,
    initial_best_value: float | None = None,
    *,
    initial_length: float = TURBO_INITIAL_LENGTH,
    length_min: float = TURBO_LENGTH_MIN,
    length_max: float = TURBO_LENGTH_MAX,
    success_tolerance: int = TURBO_SUCCESS_TOLERANCE,
    failure_tolerance: int | None = None,
) -> TurboState:
    """Create a new TuRBO state for a problem.

    Args:
        dim: Number of parameters
        batch_size: Batch size for suggestions
        initial_best_value: Best objective value seen so far in TuRBO's
            internal maximization convention (larger is better, regardless
            of user-facing direction; see :class:`TurboState`).
        initial_length: Initial trust region edge in normalized [0,1] space.
            Override only when ``TurboConfig.initial_length`` differs from
            the paper default.
        length_min: Minimum trust region edge before restart. See
            :class:`TurboState` for the scale assumption.
        length_max: Maximum trust region edge after expansion.
        success_tolerance: Consecutive improving batches before expansion.
        failure_tolerance: Consecutive failing batches before contraction.
            ``None`` triggers :class:`TurboState`'s dim/batch-size-aware
            default.

    Returns:
        Initialized TurboState
    """
    best = initial_best_value if initial_best_value is not None else float("-inf")
    return TurboState(
        dim=dim,
        batch_size=batch_size,
        length=initial_length,
        length_min=length_min,
        length_max=length_max,
        success_tolerance=success_tolerance,
        failure_tolerance=failure_tolerance,
        best_value=best,
    )


def update_turbo_state(
    state: TurboState,
    y_next: Tensor,
    *,
    minimize: bool,
) -> TurboState:
    """Update trust region based on new observations.

    The trust region expands after consecutive successes and contracts
    after consecutive failures. This adapts the search region based
    on optimization progress.

    **Batch-aware counter update.** Per Eriksson et al. (NeurIPS 2019) the
    original TuRBO algorithm counts a "success" per batch when any point
    in the batch improves the incumbent. The previous implementation
    incremented ``success_counter`` by exactly 1 regardless of how many
    points in the batch actually improved, which paced the trust-region
    expansion conservatively for batch sizes > 1. We now increment the
    success counter by the *number of distinct improving points in the
    batch* (clamped to ``batch_size``). The increment is at most one per
    point so batches with many tiny improvements do not skip over the
    expansion tolerance; the increment is at least one when any point
    improves, matching the paper's per-batch semantics.

    Args:
        state: Current TuRBO state
        y_next: Objective values from latest batch (shape: [batch_size] or [batch_size, 1])
                Raw objective values — negation for maximization is handled internally.
        minimize: User-facing objective direction.  Required keyword so the
            caller cannot silently pass the wrong sign convention; see
            :mod:`bo_engine.types` for the canonical rule.  If True, lower
            raw y is better; if False, higher raw y is better.

    Returns:
        Updated TurboState with modified counters and length
    """
    # Negate if minimizing so that "improvement" always means higher internal value
    # (TuRBO internally works in maximization convention)
    y_internal = -y_next if minimize else y_next
    y_flat = y_internal.detach().reshape(-1)
    y_max = float(y_flat.max().item())

    # Check improvement with tolerance for numerical stability
    tolerance = (
        IMPROVEMENT_TOLERANCE_RELATIVE * abs(state.best_value)
        if state.best_value != 0
        else IMPROVEMENT_TOLERANCE_ABSOLUTE
    )
    improvement_threshold = state.best_value + tolerance

    # Per-batch improvement count: each entry that strictly beats the
    # incumbent by more than the tolerance contributes one. Bounded by the
    # actual batch size so a degenerate larger-than-batch tensor cannot
    # over-credit.
    n_improving = int((y_flat > improvement_threshold).sum().item())
    n_improving = min(n_improving, max(int(state.batch_size), 1))
    improved = n_improving > 0

    # Update counters
    if improved:
        new_success_counter = state.success_counter + n_improving
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

    return TurboState(
        dim=state.dim,
        batch_size=state.batch_size,
        length=new_length,
        length_min=state.length_min,
        length_max=state.length_max,
        failure_counter=final_failure_counter,
        failure_tolerance=state.failure_tolerance,
        success_counter=final_success_counter,
        success_tolerance=state.success_tolerance,
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
        lengthscales = covar.base_kernel.lengthscale.squeeze().detach()  # ty: ignore[call-non-callable, unresolved-attribute]
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
