"""Pending point handling for suggestion generation.

Section 1.6 of Implementation Plan: Handles pending/in-progress experiments
when generating new suggestions to avoid overlap.

Active strategy: BoTorch's ``X_pending`` conditioning is the production
path. ``bo_engine.suggestions._encode_pending_points`` encodes the
in-flight batch and forwards it to ``optimize_acquisition`` as
``X_pending`` so the qNEI / qNEHVI sampler treats those points as
already-acquired and pushes new candidates away from them. The
Kriging-Believer fantasy-model approach is documented in the reference
below for context, but is **not** wired into the pipeline; we removed
the dead ``get_pending_as_fantasy_model_input`` helper to keep the
module honest.

References:
- BoTorch pending points: https://botorch.org/docs/batched_bayesian_optimization/
- Kriging Believer: https://arxiv.org/abs/1012.2599
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import torch
from torch import Tensor

from bo_engine.constants import PENDING_SUGGESTION_MAX_AGE_HOURS


@dataclass
class PendingPoint:
    """Represents a pending/in-progress experimental point.

    Attributes:
        parameter_values: Dictionary of parameter name to value
        created_at: When the suggestion was created
        age_hours: Age of the pending point in hours
        is_stale: Whether the pending point is considered stale
    """

    parameter_values: dict[str, float]
    created_at: datetime
    age_hours: float
    is_stale: bool


def compute_pending_age(
    created_at: datetime,
    now: datetime | None = None,
) -> float:
    """Compute age of a pending point in hours.

    Args:
        created_at: When the point was created
        now: Current time (defaults to UTC now)

    Returns:
        Age in hours
    """
    if now is None:
        now = datetime.now(UTC)

    # Ensure created_at is timezone-aware
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    delta = now - created_at
    return delta.total_seconds() / 3600


def filter_pending_points(
    pending_params: list[dict[str, float]],
    pending_created_at: list[datetime],
    max_age_hours: float = PENDING_SUGGESTION_MAX_AGE_HOURS,
    now: datetime | None = None,
) -> tuple[list[dict[str, float]], list[PendingPoint]]:
    """Filter pending points by age, returning non-stale ones.

    Args:
        pending_params: Parameter values for each pending suggestion
        pending_created_at: Creation times for each pending suggestion
        max_age_hours: Maximum age in hours before a point is stale
        now: Current time for age computation

    Returns:
        Tuple of (valid_params, pending_point_info) where valid_params
        contains only non-stale points

    Reference:
        Section 1.6 of Implementation Plan - Pending Point Handling
    """
    if now is None:
        now = datetime.now(UTC)

    valid_params: list[dict[str, float]] = []
    pending_info: list[PendingPoint] = []

    for params, created_at in zip(pending_params, pending_created_at, strict=True):
        age = compute_pending_age(created_at, now)
        is_stale = age > max_age_hours

        point = PendingPoint(
            parameter_values=params,
            created_at=created_at,
            age_hours=age,
            is_stale=is_stale,
        )
        pending_info.append(point)

        if not is_stale:
            valid_params.append(params)

    return valid_params, pending_info


def encode_pending_points(
    pending_params: list[dict[str, float]],
    param_names: list[str],
    bounds: Tensor,
) -> Tensor | None:
    """Encode pending points to tensor format for acquisition.

    Args:
        pending_params: Parameter values for each pending point
        param_names: Names of parameters in order
        bounds: Parameter bounds of shape (2, n_dims)

    Returns:
        Tensor of shape (n_pending, n_dims) or None if no pending points
    """
    if not pending_params:
        return None

    device = bounds.device
    dtype = bounds.dtype

    encoded = []
    for params in pending_params:
        values = []
        for name in param_names:
            val = params.get(name)
            if val is not None:
                try:
                    values.append(float(val))
                except (TypeError, ValueError):
                    # Handle categorical or non-numeric values
                    # For now, skip - in full implementation would encode
                    return None
            else:
                # Missing parameter - can't encode
                return None
        encoded.append(values)

    if not encoded:
        return None

    return torch.tensor(encoded, device=device, dtype=dtype)


def compute_pending_distance(
    candidates: Tensor,
    pending_x: Tensor,
    bounds: Tensor,
) -> Tensor:
    """Compute minimum normalized distance from candidates to pending points.

    Args:
        candidates: Candidate points of shape (n_candidates, n_dims)
        pending_x: Pending points of shape (n_pending, n_dims)
        bounds: Parameter bounds

    Returns:
        Tensor of shape (n_candidates,) with minimum distances to any pending point
    """
    if pending_x.shape[0] == 0:
        return torch.full((candidates.shape[0],), float("inf"), device=candidates.device)

    # Normalize
    ranges = bounds[1] - bounds[0]
    ranges = torch.where(ranges < 1e-10, torch.ones_like(ranges), ranges)

    norm_candidates = (candidates - bounds[0]) / ranges
    norm_pending = (pending_x - bounds[0]) / ranges

    # Compute distances
    distances = torch.cdist(norm_candidates, norm_pending, p=2)

    # Return minimum distance to any pending point
    return distances.min(dim=1).values


def penalize_near_pending(
    acq_values: Tensor,
    candidates: Tensor,
    pending_x: Tensor,
    bounds: Tensor,
    exclusion_radius: float = 0.05,
) -> Tensor:
    """Penalize acquisition values near pending points.

    Reduces acquisition values for candidates that are close to pending
    experiments to encourage exploration of new regions.

    Args:
        acq_values: Acquisition values of shape (n_candidates,)
        candidates: Candidate points of shape (n_candidates, n_dims)
        pending_x: Pending points of shape (n_pending, n_dims)
        bounds: Parameter bounds
        exclusion_radius: Normalized radius around pending points to penalize

    Returns:
        Penalized acquisition values

    Reference:
        Section 1.6 of Implementation Plan - Pending Point Handling
    """
    if pending_x is None or pending_x.shape[0] == 0:
        return acq_values

    # Compute distances to pending points
    min_distances = compute_pending_distance(candidates, pending_x, bounds)

    # Apply soft penalization (Gaussian decay)
    penalization = torch.exp(-(min_distances**2) / (2 * exclusion_radius**2))

    # Points very close to pending are strongly penalized
    return acq_values * (1 - penalization)


class PendingPointTracker:
    """Tracks pending points across suggestion generation calls.

    This class maintains state about pending experiments and provides
    methods for integrating them into the suggestion generation process.
    """

    def __init__(
        self,
        max_age_hours: float = PENDING_SUGGESTION_MAX_AGE_HOURS,
    ):
        """Initialize tracker.

        Args:
            max_age_hours: Maximum age before points are considered stale
        """
        self.max_age_hours = max_age_hours
        self._pending: list[tuple[dict[str, float], datetime]] = []

    def add_pending(
        self,
        params: dict[str, float],
        created_at: datetime | None = None,
    ) -> None:
        """Add a pending point.

        Args:
            params: Parameter values
            created_at: Creation time (defaults to now)
        """
        if created_at is None:
            created_at = datetime.now(UTC)
        self._pending.append((params, created_at))

    def remove_pending(
        self,
        params: dict[str, float],
        tolerance: float = 1e-6,
    ) -> bool:
        """Remove a pending point (e.g., when result is submitted).

        Args:
            params: Parameter values to remove
            tolerance: Tolerance for parameter matching

        Returns:
            Whether a point was removed
        """
        for i, (existing_params, _) in enumerate(self._pending):
            is_match = True
            for key, val in params.items():
                existing_val = existing_params.get(key)
                if existing_val is None:
                    is_match = False
                    break
                try:
                    if abs(float(val) - float(existing_val)) > tolerance:
                        is_match = False
                        break
                except (TypeError, ValueError):
                    if val != existing_val:
                        is_match = False
                        break
            if is_match:
                self._pending.pop(i)
                return True
        return False

    def get_valid_pending(
        self,
        now: datetime | None = None,
    ) -> list[dict[str, float]]:
        """Get non-stale pending points.

        Args:
            now: Current time for age computation

        Returns:
            List of parameter value dicts for valid pending points
        """
        params = [p for p, _ in self._pending]
        created_times = [t for _, t in self._pending]
        valid, _ = filter_pending_points(params, created_times, self.max_age_hours, now)
        return valid

    def clear_stale(
        self,
        now: datetime | None = None,
    ) -> int:
        """Remove stale pending points.

        Args:
            now: Current time for age computation

        Returns:
            Number of points removed
        """
        if now is None:
            now = datetime.now(UTC)

        original_count = len(self._pending)
        self._pending = [
            (p, t) for p, t in self._pending if compute_pending_age(t, now) <= self.max_age_hours
        ]
        return original_count - len(self._pending)

    @property
    def count(self) -> int:
        """Number of pending points."""
        return len(self._pending)
