"""Dynamic Reference Point Computation for Multi-Objective Optimization.

This module provides adaptive reference point computation strategies for
hypervolume-based multi-objective optimization. The reference point is
critical for computing hypervolume - a suboptimal reference point can
underestimate hypervolume improvements.

v2.6: Initial implementation with dynamic adaptation strategies

References:
    - Ishibuchi et al. "Reference Point Specification in Inverted
      Generational Distance for EMO" (2015)
    - Blank & Deb "pymoo: Multi-Objective Optimization in Python" (2020)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from torch import Tensor

from bo_engine.constants import (
    MIN_OBJECTIVE_RANGE,
    REFERENCE_POINT_PADDING,
)
from bo_engine.device import ensure_device, to_device


class ReferencePointStrategy(Enum):
    """Strategy for computing the reference point."""

    STATIC = "static"  # Fixed worst + margin (current default)
    DYNAMIC = "dynamic"  # Adapts based on Pareto front progress
    NADIR = "nadir"  # Nadir point estimation
    USER_SPECIFIED = "user_specified"  # User provides reference point


@dataclass
class ReferencePointConfig:
    """Configuration for reference point computation.

    Attributes:
        strategy: Method for computing reference point
        margin: Margin factor for STATIC strategy (default 0.1 = 10% beyond worst)
        adaptation_rate: Rate of adaptation for DYNAMIC strategy (0-1)
        user_reference_point: User-specified reference point (if strategy=USER_SPECIFIED)
        min_margin: Minimum margin to maintain for numerical stability
        track_history: Whether to track reference point history
    """

    strategy: ReferencePointStrategy = ReferencePointStrategy.DYNAMIC
    margin: float = REFERENCE_POINT_PADDING
    adaptation_rate: float = 0.2
    user_reference_point: list[float] | None = None
    min_margin: float = 0.01
    track_history: bool = True


@dataclass
class ReferencePointState:
    """State tracking for adaptive reference point computation.

    Attributes:
        current_ref_point: Current reference point tensor
        history: Historical reference points (if tracking enabled)
        nadir_estimate: Current nadir point estimate
        ideal_estimate: Current ideal point estimate
        n_updates: Number of updates performed
    """

    current_ref_point: Tensor | None = None
    history: list[Tensor] = field(default_factory=list)
    nadir_estimate: Tensor | None = None
    ideal_estimate: Tensor | None = None
    n_updates: int = 0


def get_reference_point_dynamic(
    train_y: Tensor,
    minimize_mask: Tensor | None = None,
    config: ReferencePointConfig | None = None,
    state: ReferencePointState | None = None,
) -> tuple[Tensor, ReferencePointState]:
    """Compute reference point with dynamic adaptation.

    This function implements multiple strategies for reference point
    computation, with the default being dynamic adaptation based on
    Pareto front progress.

    Args:
        train_y: Training outputs of shape (n_samples, n_objectives)
        minimize_mask: Boolean tensor indicating which objectives to minimize.
            If None, all objectives are assumed to be minimized.
        config: Reference point configuration
        state: Previous state for adaptive strategies

    Returns:
        Tuple of (reference_point, updated_state) where:
        - reference_point has shape (n_objectives,)
        - updated_state contains the new state for future calls

    Example:
        >>> train_y = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 1.5]])
        >>> ref_point, state = get_reference_point_dynamic(train_y)
        >>> ref_point  # Reference point beyond worst values
    """
    train_y = to_device(train_y)

    if config is None:
        config = ReferencePointConfig()

    if state is None:
        state = ReferencePointState()

    n_objectives = train_y.shape[-1]

    # Handle minimize_mask
    if minimize_mask is None:
        minimize_mask = torch.ones(n_objectives, dtype=torch.bool)
    minimize_mask = to_device(minimize_mask)

    # Dispatch to appropriate strategy
    if config.strategy == ReferencePointStrategy.USER_SPECIFIED:
        if config.user_reference_point is None:
            raise ValueError("User-specified strategy requires user_reference_point")
        ref_point = torch.tensor(
            config.user_reference_point, dtype=train_y.dtype, device=train_y.device
        )
    elif config.strategy == ReferencePointStrategy.NADIR:
        ref_point = _compute_nadir_reference_point(train_y, minimize_mask, config, state)
    elif config.strategy == ReferencePointStrategy.DYNAMIC:
        ref_point = _compute_dynamic_reference_point(train_y, minimize_mask, config, state)
    else:  # STATIC
        ref_point = _compute_static_reference_point(train_y, minimize_mask, config)

    # Update state
    state.current_ref_point = ref_point.clone()
    state.n_updates += 1

    if config.track_history:
        state.history.append(ref_point.clone())

    return ref_point, state


def _compute_static_reference_point(
    train_y: Tensor,
    _minimize_mask: Tensor,
    config: ReferencePointConfig,
) -> Tensor:
    """Compute static reference point (current behavior).

    The reference point is set to worst + margin * range for each objective.
    For minimization objectives, worst = max.
    For maximization objectives (if any), worst = min (before negation).

    Args:
        train_y: Training outputs (assumed already negated for maximization)
        minimize_mask: Boolean mask (True = minimize)
        config: Configuration with margin setting

    Returns:
        Reference point tensor
    """
    # For BoTorch, all objectives should already be negated for maximization
    # So "worst" is always max
    worst = train_y.max(dim=0).values
    ranges = train_y.max(dim=0).values - train_y.min(dim=0).values
    ranges = torch.where(ranges < MIN_OBJECTIVE_RANGE, torch.ones_like(ranges), ranges)

    ref_point = worst + config.margin * ranges
    return ref_point


def _compute_nadir_reference_point(
    train_y: Tensor,
    minimize_mask: Tensor,
    config: ReferencePointConfig,
    state: ReferencePointState,
) -> Tensor:
    """Compute reference point based on nadir point estimation.

    The nadir point is the vector of worst objective values among
    all Pareto-optimal solutions. This provides a tighter reference
    point that better reflects the true dominated hypervolume.

    Args:
        train_y: Training outputs (already negated for maximization objectives)
        minimize_mask: Boolean mask (True = minimize)
        config: Configuration
        state: Previous state

    Returns:
        Reference point tensor
    """
    # Find Pareto-optimal points (BoTorch expects maximization, so negate)
    pareto_mask = is_non_dominated(-train_y)
    pareto_y = train_y[pareto_mask]

    if pareto_y.shape[0] == 0:
        # No Pareto points yet, fall back to static
        return _compute_static_reference_point(train_y, minimize_mask, config)

    # Nadir point: worst value among Pareto points for each objective
    nadir = pareto_y.max(dim=0).values

    # Update nadir estimate in state (keep worst seen)
    if state.nadir_estimate is not None:
        state.nadir_estimate = torch.max(nadir, state.nadir_estimate)
    else:
        state.nadir_estimate = nadir.clone()

    # Update ideal estimate (best seen)
    ideal = pareto_y.min(dim=0).values
    if state.ideal_estimate is not None:
        state.ideal_estimate = torch.min(ideal, state.ideal_estimate)
    else:
        state.ideal_estimate = ideal.clone()

    # Reference point = nadir + small margin
    ranges = state.nadir_estimate - state.ideal_estimate
    ranges = torch.where(ranges < MIN_OBJECTIVE_RANGE, torch.ones_like(ranges), ranges)

    ref_point = state.nadir_estimate + config.min_margin * ranges
    return ref_point


def _compute_dynamic_reference_point(
    train_y: Tensor,
    minimize_mask: Tensor,
    config: ReferencePointConfig,
    state: ReferencePointState,
) -> Tensor:
    """Compute dynamically adapting reference point.

    The dynamic strategy starts with a conservative reference point
    and gradually adapts it based on observed progress. This balances:
    1. Stability (not changing too rapidly)
    2. Accuracy (reflecting actual dominated region)

    The adaptation uses exponential moving average:
    ref_new = (1 - rate) * ref_old + rate * ref_computed

    Args:
        train_y: Training outputs
        minimize_mask: Boolean mask
        config: Configuration with adaptation_rate
        state: Previous state with current_ref_point

    Returns:
        Adapted reference point tensor
    """
    # Compute nadir-based reference point
    nadir_ref = _compute_nadir_reference_point(train_y, minimize_mask, config, state)

    if state.current_ref_point is None:
        # First call - start with conservative static reference
        static_ref = _compute_static_reference_point(train_y, minimize_mask, config)
        # Use max of nadir and static for safety
        return torch.max(nadir_ref, static_ref)

    # Ensure devices match
    prev_ref = state.current_ref_point.to(train_y.device)

    # Exponential moving average adaptation
    # Only tighten the reference point, never expand it
    # (expanding could invalidate previously computed hypervolumes)
    adapted_ref = (1 - config.adaptation_rate) * prev_ref + config.adaptation_rate * nadir_ref

    # Ensure reference point never goes below current nadir estimate
    if state.nadir_estimate is not None:
        adapted_ref = torch.max(
            adapted_ref, state.nadir_estimate + config.min_margin * torch.ones_like(adapted_ref)
        )

    return adapted_ref


def compute_reference_point_quality(
    ref_point: Tensor,
    train_y: Tensor,
) -> dict[str, Any]:
    """Assess the quality of a reference point.

    Computes metrics to help understand if the reference point is
    well-suited for the current Pareto front.

    Args:
        ref_point: Reference point tensor
        train_y: Training outputs (all objectives in minimization form)

    Returns:
        Dictionary with quality metrics:
        - dominated_fraction: Fraction of points dominated by ref_point
        - margin_ratios: Ratio of margin to range for each objective
        - is_valid: Whether reference point dominates all observed points
        - recommendations: List of recommendations for improvement
    """
    ref_point, train_y = ensure_device(ref_point, train_y)

    # Check if reference point dominates all observed points
    # (it should, since ref_point should be worse than worst)
    dominated = (train_y <= ref_point).all(dim=-1)
    dominated_fraction = dominated.float().mean().item()
    is_valid = math.isclose(dominated_fraction, 1.0)

    # Compute margin ratios
    worst = train_y.max(dim=0).values
    best = train_y.min(dim=0).values
    ranges = worst - best
    ranges = torch.where(ranges < MIN_OBJECTIVE_RANGE, torch.ones_like(ranges), ranges)
    margins = ref_point - worst
    margin_ratios = (margins / ranges).tolist()

    # Generate recommendations
    recommendations = []

    if not is_valid:
        recommendations.append(
            "Reference point does not dominate all points. "
            "Consider using a more conservative (larger) reference point."
        )

    for i, ratio in enumerate(margin_ratios):
        if ratio > 0.5:
            recommendations.append(
                f"Objective {i}: margin ratio {ratio:.2f} is large. "
                "Consider tightening the reference point for better resolution."
            )
        elif ratio < 0.01:
            recommendations.append(
                f"Objective {i}: margin ratio {ratio:.3f} is very small. "
                "This may cause numerical instability."
            )

    return {
        "dominated_fraction": dominated_fraction,
        "margin_ratios": margin_ratios,
        "is_valid": is_valid,
        "recommendations": recommendations,
    }


def _select_best_strategy(
    ref_dynamic: Tensor,
    ref_nadir: Tensor,
    ref_static: Tensor,
    quality_dynamic: dict[str, Any],
    quality_nadir: dict[str, Any],
) -> tuple[ReferencePointStrategy, Tensor, list[str]]:
    """Select the best reference point strategy based on quality metrics.

    Prefers dynamic if valid, falls back to nadir then static.
    When both dynamic and nadir are valid, picks the tighter one.

    Args:
        ref_dynamic: Dynamic reference point
        ref_nadir: Nadir-based reference point
        ref_static: Static reference point
        quality_dynamic: Quality metrics for dynamic ref point
        quality_nadir: Quality metrics for nadir ref point

    Returns:
        Tuple of (best_strategy, best_ref_point, explanation_parts)
    """
    explanation_parts: list[str] = []

    if not quality_dynamic["is_valid"]:
        if quality_nadir["is_valid"]:
            explanation_parts.append("Using nadir-based reference point for tighter bounds.")
            return ReferencePointStrategy.NADIR, ref_nadir, explanation_parts
        explanation_parts.append("Using static reference point for stability.")
        return ReferencePointStrategy.STATIC, ref_static, explanation_parts

    # Dynamic is valid — check if nadir is also valid and tighter
    if quality_nadir["is_valid"]:
        nadir_tighter = all(
            nr < dr for nr, dr in zip(ref_nadir.tolist(), ref_dynamic.tolist(), strict=False)
        )
        if nadir_tighter:
            explanation_parts.append("Nadir-based reference is tighter and valid.")
            return ReferencePointStrategy.NADIR, ref_nadir, explanation_parts
        explanation_parts.append("Dynamic reference point adapts to Pareto progress.")
    else:
        explanation_parts.append("Dynamic reference point provides good balance.")

    return ReferencePointStrategy.DYNAMIC, ref_dynamic, explanation_parts


def _append_current_ref_point_advice(
    current_ref_point: Tensor,
    train_y: Tensor,
    explanation_parts: list[str],
) -> None:
    """Append advice about the current reference point to explanation_parts.

    Checks validity and margin ratios of the current reference point and
    adds actionable advice if improvements are possible.

    Args:
        current_ref_point: The user's current reference point
        train_y: Training outputs for quality evaluation
        explanation_parts: List to append explanation strings to (mutated in-place)
    """
    current_quality = compute_reference_point_quality(current_ref_point, train_y)
    if not current_quality["is_valid"]:
        explanation_parts.append(
            "Current reference point does not dominate all points - update recommended."
        )
    elif current_quality["margin_ratios"] and max(current_quality["margin_ratios"]) > 0.5:
        explanation_parts.append(
            "Current reference point has large margins - tightening may improve resolution."
        )


def recommend_reference_point(
    train_y: Tensor,
    current_ref_point: Tensor | None = None,
    minimize_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Recommend an optimal reference point for the current data.

    Analyzes the objective data and current reference point (if any)
    to recommend improvements.

    Args:
        train_y: Training outputs
        current_ref_point: Current reference point (optional)
        minimize_mask: Boolean mask for minimization

    Returns:
        Dictionary with:
        - recommended_ref_point: Suggested reference point
        - strategy: Recommended strategy
        - explanation: Human-readable explanation
    """
    train_y = to_device(train_y)
    n_objectives = train_y.shape[-1]

    if minimize_mask is None:
        minimize_mask = torch.ones(n_objectives, dtype=torch.bool)

    # Compute different reference points
    config_dynamic = ReferencePointConfig(strategy=ReferencePointStrategy.DYNAMIC)
    config_nadir = ReferencePointConfig(strategy=ReferencePointStrategy.NADIR)
    config_static = ReferencePointConfig(strategy=ReferencePointStrategy.STATIC)

    ref_dynamic, _ = get_reference_point_dynamic(train_y, minimize_mask, config_dynamic)
    ref_nadir, _ = get_reference_point_dynamic(train_y, minimize_mask, config_nadir)
    ref_static, _ = get_reference_point_dynamic(train_y, minimize_mask, config_static)

    # Evaluate quality of each
    quality_dynamic = compute_reference_point_quality(ref_dynamic, train_y)
    quality_nadir = compute_reference_point_quality(ref_nadir, train_y)

    # Choose based on validity and margin balance
    best_strategy, best_ref, explanation_parts = _select_best_strategy(
        ref_dynamic, ref_nadir, ref_static, quality_dynamic, quality_nadir
    )

    # Check current reference point if provided
    if current_ref_point is not None:
        _append_current_ref_point_advice(current_ref_point, train_y, explanation_parts)

    return {
        "recommended_ref_point": best_ref.tolist(),
        "strategy": best_strategy.value,
        "explanation": " ".join(explanation_parts),
        "alternatives": {
            "dynamic": ref_dynamic.tolist(),
            "nadir": ref_nadir.tolist(),
            "static": ref_static.tolist(),
        },
    }


# Backward compatibility - maintain existing API
def get_reference_point(
    train_y: Tensor,
    minimize_mask: Tensor,
    margin: float = 0.1,
) -> Tensor:
    """Compute reference point for hypervolume (backward compatible).

    This function maintains backward compatibility with the existing API
    while using the new dynamic reference point computation.

    Args:
        train_y: Training outputs of shape (n_samples, n_objectives)
        minimize_mask: Boolean tensor indicating which objectives to minimize
        margin: Margin factor (e.g., 0.1 = 10% worse)

    Returns:
        Reference point tensor of shape (n_objectives,)
    """
    config = ReferencePointConfig(
        strategy=ReferencePointStrategy.STATIC,  # Use static for backward compat
        margin=margin,
    )
    ref_point, _ = get_reference_point_dynamic(train_y, minimize_mask, config)
    return ref_point
