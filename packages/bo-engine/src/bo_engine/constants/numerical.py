"""Numerical-stability epsilons, random-seed bounds, and confidence-interval z-scores."""

__all__ = [
    "CI_95_Z_SCORE",
    "IMPROVEMENT_TOLERANCE_ABSOLUTE",
    "IMPROVEMENT_TOLERANCE_RELATIVE",
    "MAX_RANDOM_SEED",
    "NUMERICAL_EPSILON",
    "SAFE_DIVISION_EPSILON",
    "STAGNATION_TOLERANCE_RELATIVE",
    "is_zero",
]


# =============================================================================
# Numerical Stability
# =============================================================================

# General-purpose epsilon for division guards and near-zero checks.
# Use for denominators, range checks, and absolute-value comparisons.
NUMERICAL_EPSILON = 1e-10


def is_zero(value: float, tol: float = NUMERICAL_EPSILON) -> bool:
    """Return ``True`` when ``value`` is within ``tol`` of zero.

    Use in place of bare ``x == 0`` / ``x != 0`` checks on computed
    floats so finite-precision artifacts (a result of order 1e-16
    instead of an exact zero) do not flip the branch.
    """
    return abs(value) <= tol


# Epsilon for clamping standard deviations and values before log().
# Slightly larger than NUMERICAL_EPSILON to avoid log-space underflow
# while remaining negligible relative to any realistic objective scale.
SAFE_DIVISION_EPSILON = 1e-6

# Tolerance for improvement detection (relative)
IMPROVEMENT_TOLERANCE_RELATIVE = 1e-3

# Tolerance for improvement detection (absolute, for zero values)
IMPROVEMENT_TOLERANCE_ABSOLUTE = 1e-6

# Fraction of the improvement-history's robust scale (see
# ``convergence._history_scale``) below which a step delta counts as "no
# improvement" for the single-objective stagnation counter. Calibrated so a
# unit-scale trajectory keeps the historical ``IMPROVEMENT_TOLERANCE_ABSOLUTE``
# behavior while micro-/macro-scale objectives get the same verdict after a
# units change (scale invariance; same class as the convergence-detector fix).
STAGNATION_TOLERANCE_RELATIVE = 1e-6

# =============================================================================
# Random Seeds
# =============================================================================

# Maximum random seed value (2^31 - 1 for compatibility)
MAX_RANDOM_SEED = 2**31 - 1

# =============================================================================
# Z-Scores for Confidence Intervals
# =============================================================================

# Z-score for 95 % confidence interval (normal distribution)
CI_95_Z_SCORE = 1.96
