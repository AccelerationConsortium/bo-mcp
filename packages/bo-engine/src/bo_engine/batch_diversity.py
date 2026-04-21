"""Batch diversity enforcement for suggestion generation.

Section 1.5 of Implementation Plan: Ensures batch suggestions are diverse
to maximize experimental efficiency.

References:
- Local Penalization in BO: https://arxiv.org/abs/1505.08052
- Batch Bayesian Optimization: https://proceedings.mlr.press/v51/gonzalez16a.html
- BoTorch batch optimization: https://botorch.org/docs/batched_bayesian_optimization/
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from bo_engine.constants import (
    BATCH_DIVERSITY_MAX_ATTEMPTS,
    BATCH_DIVERSITY_MIN_DISTANCE,
)


@dataclass
class DiversityMetrics:
    """Metrics describing batch diversity.

    Attributes:
        min_pairwise_distance: Minimum distance between any two points
        mean_pairwise_distance: Mean distance between all pairs
        diversity_score: Normalized score from 0 (clustered) to 1 (diverse)
        is_diverse: Whether batch meets minimum diversity threshold
        pairs_below_threshold: Number of pairs below minimum distance
    """

    min_pairwise_distance: float
    mean_pairwise_distance: float
    diversity_score: float
    is_diverse: bool
    pairs_below_threshold: int


def compute_batch_diversity(
    candidates: Tensor,
    bounds: Tensor,
    min_distance: float = BATCH_DIVERSITY_MIN_DISTANCE,
) -> DiversityMetrics:
    """Compute diversity metrics for a batch of candidates.

    Args:
        candidates: Candidate points of shape (batch_size, n_dims)
        bounds: Parameter bounds of shape (2, n_dims)
        min_distance: Minimum normalized distance threshold

    Returns:
        DiversityMetrics with computed values

    Reference:
        Section 1.5 of Implementation Plan - Batch Diversity Enforcement
    """
    batch_size = candidates.shape[0]

    if batch_size < 2:
        return DiversityMetrics(
            min_pairwise_distance=float("inf"),
            mean_pairwise_distance=float("inf"),
            diversity_score=1.0,
            is_diverse=True,
            pairs_below_threshold=0,
        )

    # Normalize candidates to [0, 1] space
    ranges = bounds[1] - bounds[0]
    ranges = torch.where(ranges < 1e-10, torch.ones_like(ranges), ranges)
    normalized = (candidates - bounds[0]) / ranges

    # Compute pairwise distances
    distances = torch.cdist(normalized, normalized, p=2)

    # Extract upper triangular (excluding diagonal)
    n = batch_size
    upper_tri_indices = torch.triu_indices(n, n, offset=1)
    pairwise_distances = distances[upper_tri_indices[0], upper_tri_indices[1]]

    if pairwise_distances.numel() == 0:
        return DiversityMetrics(
            min_pairwise_distance=float("inf"),
            mean_pairwise_distance=float("inf"),
            diversity_score=1.0,
            is_diverse=True,
            pairs_below_threshold=0,
        )

    min_dist = pairwise_distances.min().item()
    mean_dist = pairwise_distances.mean().item()

    # Count pairs below threshold
    pairs_below = (pairwise_distances < min_distance).sum().item()

    # Compute diversity score based on min distance and expected distance
    # Expected distance in unit hypercube is sqrt(n_dims/6) for uniform distribution
    n_dims = candidates.shape[1]
    expected_distance = (n_dims / 6) ** 0.5
    diversity_score = min(1.0, min_dist / expected_distance)

    is_diverse = min_dist >= min_distance and pairs_below == 0

    return DiversityMetrics(
        min_pairwise_distance=min_dist,
        mean_pairwise_distance=mean_dist,
        diversity_score=diversity_score,
        is_diverse=is_diverse,
        pairs_below_threshold=int(pairs_below),
    )


def enforce_diversity(
    candidates: Tensor,
    bounds: Tensor,
    min_distance: float = BATCH_DIVERSITY_MIN_DISTANCE,
    max_attempts: int = BATCH_DIVERSITY_MAX_ATTEMPTS,
) -> tuple[Tensor, bool]:
    """Enforce minimum diversity by perturbing close candidates.

    When candidates are too close together, applies small perturbations
    to spread them apart while staying within bounds.

    Args:
        candidates: Candidate points of shape (batch_size, n_dims)
        bounds: Parameter bounds of shape (2, n_dims)
        min_distance: Minimum normalized distance to enforce
        max_attempts: Maximum perturbation attempts

    Returns:
        Tuple of (modified_candidates, success) where success indicates
        whether minimum diversity was achieved

    Reference:
        Section 1.5 of Implementation Plan - Batch Diversity Enforcement
    """
    batch_size = candidates.shape[0]

    if batch_size < 2:
        return candidates, True

    # Normalize to [0, 1] space for distance computation
    ranges = bounds[1] - bounds[0]
    ranges = torch.where(ranges < 1e-10, torch.ones_like(ranges), ranges)

    result = candidates.clone()

    for _attempt in range(max_attempts):
        # Normalize current candidates
        normalized = (result - bounds[0]) / ranges

        # Find closest pair
        distances = torch.cdist(normalized, normalized, p=2)
        # Set diagonal to inf to ignore self-distances
        distances.fill_diagonal_(float("inf"))

        min_dist = distances.min().item()
        if min_dist >= min_distance:
            return result, True

        # Find the closest pair
        flat_idx = distances.argmin()
        i = flat_idx // batch_size
        j = flat_idx % batch_size

        # Push points apart by adding small perturbation
        direction = normalized[i] - normalized[j]
        direction_norm = direction.norm()

        if direction_norm > 1e-10:
            direction = direction / direction_norm
        else:
            # Random direction if points are identical
            direction = torch.randn_like(direction)
            direction = direction / direction.norm()

        # Push both points in opposite directions
        push_amount = (min_distance - min_dist) / 2 + 0.01
        perturbation = direction * push_amount

        # Apply perturbation in normalized space, then convert back
        new_normalized_i = normalized[i] + perturbation
        new_normalized_j = normalized[j] - perturbation

        # Clamp to [0, 1] in normalized space
        new_normalized_i = torch.clamp(new_normalized_i, 0, 1)
        new_normalized_j = torch.clamp(new_normalized_j, 0, 1)

        # Convert back to original space
        result[i] = new_normalized_i * ranges + bounds[0]
        result[j] = new_normalized_j * ranges + bounds[0]

    # Check final diversity
    final_metrics = compute_batch_diversity(result, bounds, min_distance)
    return result, final_metrics.is_diverse


def apply_local_penalization(
    acq_values: Tensor,
    candidates: Tensor,
    selected_points: Tensor,
    bounds: Tensor,
    lengthscale: float = 0.1,
) -> Tensor:
    """Apply local penalization to acquisition values for diversity.

    Reduces acquisition values near already-selected points to encourage
    diverse batch generation.

    Args:
        acq_values: Acquisition function values of shape (n_candidates,)
        candidates: Candidate points of shape (n_candidates, n_dims)
        selected_points: Already selected points of shape (n_selected, n_dims)
        bounds: Parameter bounds
        lengthscale: Lengthscale for penalization kernel

    Returns:
        Penalized acquisition values

    Reference:
        "Batch Bayesian Optimization via Local Penalization"
        https://arxiv.org/abs/1505.08052
    """
    if selected_points.shape[0] == 0:
        return acq_values

    # Normalize
    ranges = bounds[1] - bounds[0]
    ranges = torch.where(ranges < 1e-10, torch.ones_like(ranges), ranges)

    norm_candidates = (candidates - bounds[0]) / ranges
    norm_selected = (selected_points - bounds[0]) / ranges

    # Compute distances to selected points
    distances = torch.cdist(norm_candidates, norm_selected, p=2)

    # Compute penalization factor (Gaussian kernel)
    min_distances = distances.min(dim=1).values
    penalization = torch.exp(-(min_distances**2) / (2 * lengthscale**2))

    # Apply penalization (multiply by (1 - penalization))
    penalized_acq = acq_values * (1 - penalization)

    return penalized_acq


def _is_diverse_from_selected(
    candidate_norm: Tensor,
    selected_normalized: list[Tensor],
    min_distance: float,
) -> bool:
    """Check if a candidate is sufficiently far from all already-selected points."""
    for sel_norm in selected_normalized:
        if torch.norm(candidate_norm - sel_norm, dim=-1).item() < min_distance:
            return False
    return True


def _greedy_diverse_selection(
    normalized: Tensor,
    sorted_indices: Tensor,
    batch_size: int,
    min_distance: float,
) -> list[int]:
    """Greedily select diverse candidates by acquisition value, enforcing min distance."""
    selected_indices: list[int] = []
    selected_normalized: list[Tensor] = []

    for idx in sorted_indices:
        idx_val = int(idx.item())
        if _is_diverse_from_selected(normalized[idx_val], selected_normalized, min_distance):
            selected_indices.append(idx_val)
            selected_normalized.append(normalized[idx_val])
        if len(selected_indices) >= batch_size:
            break

    return selected_indices


def filter_diverse_candidates(
    candidates: Tensor,
    acq_values: Tensor,
    bounds: Tensor,
    batch_size: int,
    min_distance: float = BATCH_DIVERSITY_MIN_DISTANCE,
) -> tuple[Tensor, Tensor]:
    """Select diverse subset from candidates greedily.

    Greedily selects candidates with highest acquisition values while
    maintaining minimum pairwise distance.

    Args:
        candidates: Candidate points of shape (n_candidates, n_dims)
        acq_values: Acquisition values of shape (n_candidates,)
        bounds: Parameter bounds
        batch_size: Number of candidates to select
        min_distance: Minimum normalized distance between selected points

    Returns:
        Tuple of (selected_candidates, selected_acq_values)
    """
    if candidates.shape[0] <= batch_size:
        return candidates, acq_values

    ranges = bounds[1] - bounds[0]
    ranges = torch.where(ranges < 1e-10, torch.ones_like(ranges), ranges)
    normalized = (candidates - bounds[0]) / ranges
    sorted_indices = torch.argsort(acq_values, descending=True)

    selected_indices = _greedy_diverse_selection(
        normalized,
        sorted_indices,
        batch_size,
        min_distance,
    )

    # Fill with best remaining if not enough diverse candidates found
    if len(selected_indices) < batch_size:
        selected_set = set(selected_indices)
        for idx in sorted_indices:
            idx_val = int(idx.item())
            if idx_val not in selected_set:
                selected_indices.append(idx_val)
                selected_set.add(idx_val)
            if len(selected_indices) >= batch_size:
                break

    selected_indices_tensor = torch.tensor(
        selected_indices, dtype=torch.long, device=candidates.device
    )
    return candidates[selected_indices_tensor], acq_values[selected_indices_tensor]
