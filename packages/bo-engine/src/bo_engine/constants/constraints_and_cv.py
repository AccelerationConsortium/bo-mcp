"""Outcome-constraint modeling, cross-validation, and model-selection thresholds."""

__all__ = [
    "CONSTRAINT_CALIBRATION_N_BINS",
    "CONSTRAINT_CALIBRATION_WARN_THRESHOLD",
    "CONSTRAINT_PROBABILITY_THRESHOLD",
    "CONSTRAINT_VIOLATION_WEIGHT",
    "CV_APPROXIMATE_THRESHOLD",
    "CV_CACHE_TTL",
    "CV_DEFAULT_K_FOLDS",
    "MODEL_SELECTION_CRITERION",
    "MODEL_SELECTION_MIN_IMPROVEMENT",
]


# =============================================================================
# Outcome Constraint Modeling (Section 2.3)
# =============================================================================

# Default probability threshold for constraint satisfaction
CONSTRAINT_PROBABILITY_THRESHOLD = 0.5

# Weight for expected constraint violation in acquisition
CONSTRAINT_VIOLATION_WEIGHT = 1.0

# Calibration error (mean absolute deviation between predicted feasibility
# probability and realized binary feasibility) above which the outcome
# constraint model is considered miscalibrated. Surfaced by
# ``assess_constraint_model_quality`` and lifted into ``get_diagnostics`` so
# agents can react to overconfident feasibility predictions before scheduling
# expensive experiments.
CONSTRAINT_CALIBRATION_WARN_THRESHOLD = 0.1

# Number of equal-width probability bins used to compute the expected
# calibration error (ECE) for outcome constraint models. Ten bins is the
# textbook default (Guo et al., 2017; Naeini et al., 2015).
CONSTRAINT_CALIBRATION_N_BINS = 10

# =============================================================================
# Cross-Validation Optimization (Section 2.4)
# =============================================================================

# Dataset size threshold above which to use approximate LOO-CV
CV_APPROXIMATE_THRESHOLD = 100

# Default number of folds for K-fold CV
CV_DEFAULT_K_FOLDS = 5

# Cache TTL for CV results (seconds)
CV_CACHE_TTL = 300.0

# =============================================================================
# Model Selection (Section 2.5)
# =============================================================================

# Minimum relative improvement required to prefer complex model
MODEL_SELECTION_MIN_IMPROVEMENT = 0.05

# Default selection criterion ("cv_rmse", "cv_r2", "lml", "bic")
MODEL_SELECTION_CRITERION = "cv_rmse"
