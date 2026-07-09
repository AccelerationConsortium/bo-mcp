"""TuRBO trust-region configuration."""

__all__ = [
    "TURBO_CONTRACTION_FACTOR",
    "TURBO_EXPANSION_FACTOR",
    "TURBO_INITIAL_LENGTH",
    "TURBO_LENGTH_MAX",
    "TURBO_LENGTH_MIN",
    "TURBO_MAX_FAILURE_TOLERANCE",
    "TURBO_SUCCESS_TOLERANCE",
    "TURBO_UNIT_SCALE_MEAN_ABS_MAX",
    "TURBO_UNIT_SCALE_STD_MAX",
    "TURBO_UNIT_SCALE_STD_MIN",
]


# =============================================================================
# TuRBO Configuration
# =============================================================================

# Initial trust region length (normalized [0,1] space)
TURBO_INITIAL_LENGTH = 0.8

# Minimum trust region length before restart
TURBO_LENGTH_MIN = 0.5**7  # ~0.0078

# Maximum trust region length after expansion
TURBO_LENGTH_MAX = 1.6

# Maximum failure tolerance to prevent TuRBO from never restarting
# in very high-dimensional problems. Without this cap,
# dim=1000 / batch_size=1 → failure_tolerance=1000.
TURBO_MAX_FAILURE_TOLERANCE = 20

# Number of consecutive successes before expanding trust region
TURBO_SUCCESS_TOLERANCE = 10

# Trust region expansion factor (multiply by this on success)
TURBO_EXPANSION_FACTOR = 2.0

# Trust region contraction factor (divide by this on failure)
TURBO_CONTRACTION_FACTOR = 2.0

# TuRBO's expand/contract logic compares improvement against
# ``IMPROVEMENT_TOLERANCE_RELATIVE * abs(best_value)`` (see
# ``update_turbo_state``). That cadence is calibrated to unit-standardized
# targets — ``Standardize(m=1)`` keeps ``train_Y`` at mean≈0, std≈1, so the
# improvement tolerance lands in the noise floor rather than at a fraction of
# the natural objective scale. ``TURBO_UNIT_SCALE_*`` bound the *unstandardized*
# training targets we accept without warning: ``|mean| > MAX`` or ``std`` outside
# ``[MIN, MAX]`` triggers a warning recommending an outcome transform.
TURBO_UNIT_SCALE_MEAN_ABS_MAX = 10.0
TURBO_UNIT_SCALE_STD_MIN = 0.05
TURBO_UNIT_SCALE_STD_MAX = 20.0
