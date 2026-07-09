"""Convergence detection, batch-diversity enforcement, and pending-point handling."""

__all__ = [
    "BATCH_DIVERSITY_MAX_ATTEMPTS",
    "BATCH_DIVERSITY_MIN_DISTANCE",
    "CONVERGENCE_IMPROVEMENT_THRESHOLD",
    "CONVERGENCE_MIN_OBSERVATIONS",
    "CONVERGENCE_WINDOW_SIZE",
    "ESTIMATE_REMAINING_MAX_ITERATIONS",
    "PENDING_SUGGESTION_MAX_AGE_HOURS",
]


# =============================================================================
# Early Stopping / Convergence Detection (Section 1.4)
# =============================================================================

# Window size for convergence detection
CONVERGENCE_WINDOW_SIZE = 5

# Relative improvement threshold below which convergence is detected
CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.01

# Minimum number of observations for convergence detection
CONVERGENCE_MIN_OBSERVATIONS = 10

# Upper bound on the horizon reported by ``estimate_remaining_iterations``.
# The estimate ``target_improvement / avg_improvement`` is unbounded as the
# improvement rate approaches zero, so it is capped to keep the reported
# horizon actionable rather than astronomically large.
ESTIMATE_REMAINING_MAX_ITERATIONS = 100

# =============================================================================
# Batch Diversity Enforcement (Section 1.5)
# =============================================================================

# Minimum normalized distance between batch suggestions
BATCH_DIVERSITY_MIN_DISTANCE = 0.05

# Maximum attempts to generate diverse batch
BATCH_DIVERSITY_MAX_ATTEMPTS = 10

# =============================================================================
# Pending Point Handling (Section 1.6)
# =============================================================================

# Maximum age (in hours) for pending suggestions to be considered
PENDING_SUGGESTION_MAX_AGE_HOURS = 24
