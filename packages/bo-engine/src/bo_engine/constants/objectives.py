"""Multi-objective reference-point computation (static and dynamic)."""

__all__ = [
    "MIN_OBJECTIVE_RANGE",
    "MIN_OBSERVATIONS_FOR_HYPERVOLUME",
    "REFERENCE_POINT_ADAPTATION_RATE",
    "REFERENCE_POINT_MIN_MARGIN",
    "REFERENCE_POINT_PADDING",
    "REFERENCE_POINT_RELATIVE_TOLERANCE",
]


# =============================================================================
# Reference Point Computation (Multi-objective)
# =============================================================================

# Padding factor for reference point beyond worst observed
REFERENCE_POINT_PADDING = 0.1

# Minimum range to avoid numerical issues
MIN_OBJECTIVE_RANGE = 1e-6

# Minimum number of observations before the observed hypervolume is defined.
# Below this a multi-objective campaign has no Pareto front yet, so every
# backend's ``compute_hypervolume`` returns ``0.0`` (vs. ``None`` for a
# single-objective campaign, where hypervolume is undefined). Shared by
# ``compute_observed_hypervolume`` so the contract cannot diverge per backend.
MIN_OBSERVATIONS_FOR_HYPERVOLUME = 2

# Floor for the per-objective margin window expressed as a fraction of
# ``abs(worst)``. The static reference point uses
# ``ref = worst + margin * max(range, abs(worst) * RELATIVE_TOLERANCE)`` so that
# objectives whose observed range collapses near zero still keep a margin
# proportional to their absolute scale; without it large-magnitude objectives
# with a tiny spread would receive an effectively zero offset and degenerate
# hypervolume.
REFERENCE_POINT_RELATIVE_TOLERANCE = 0.01

# =============================================================================
# Dynamic Reference Point (Section 2.1)
# =============================================================================

# Adaptation rate for dynamic reference point (0-1, higher = faster adaptation)
REFERENCE_POINT_ADAPTATION_RATE = 0.2

# Minimum margin to maintain for numerical stability
REFERENCE_POINT_MIN_MARGIN = 0.01
