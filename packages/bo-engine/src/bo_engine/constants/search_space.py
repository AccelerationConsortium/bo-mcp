"""Search-space sizing: dimensionality thresholds, discrete/mixed enumeration, initial design."""

__all__ = [
    "DISCRETE_ENUMERATION_MAX_POINTS",
    "HIGH_DIMENSION_WARNING_THRESHOLD",
    "INITIAL_DESIGN_MULTIPLIER",
    "MIXED_CATEGORICAL_COMBO_THRESHOLD",
    "SAASBO_MIN_DIMENSIONS",
    "TURBO_MIN_DIMENSIONS",
]


# =============================================================================
# High-Dimensional Optimization Thresholds
# =============================================================================

# Minimum number of parameters to consider TuRBO
TURBO_MIN_DIMENSIONS = 20

# Minimum number of parameters to consider SAASBO
SAASBO_MIN_DIMENSIONS = 50

# Number of parameters that triggers high-dimensional warnings
HIGH_DIMENSION_WARNING_THRESHOLD = 20

# =============================================================================
# Discrete / Mixed Search Space Optimization
# =============================================================================

# Maximum number of categorical combinations for the optimize_acqf_mixed path.
# Each combination triggers a separate L-BFGS-B run over the continuous dims.
# 100 runs complete in seconds for typical problem sizes; increase if you have
# more computational budget or reduce if continuous dimensionality is high.
MIXED_CATEGORICAL_COMBO_THRESHOLD = 100

# Maximum number of discrete points for exhaustive enumeration in
# optimize_acqf_discrete. Above this, a ValueError is raised to prevent
# silent memory issues from loading a massive choices tensor.
DISCRETE_ENUMERATION_MAX_POINTS = 10_000

# =============================================================================
# Initial Design
# =============================================================================

# Default multiplier for initial design size: 2 * n_params + 1
INITIAL_DESIGN_MULTIPLIER = 2
