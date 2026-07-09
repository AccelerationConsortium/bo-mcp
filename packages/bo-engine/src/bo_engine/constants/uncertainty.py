"""Confidence/sensitivity/calibration reporting, posterior checks, Thompson sampling, what-if."""

__all__ = [
    "CALIBRATION_CONFIDENCE_LEVELS",
    "CALIBRATION_GOOD_THRESHOLD",
    "CALIBRATION_POOR_THRESHOLD",
    "COMMON_CONFIDENCE_LEVELS",
    "CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD",
    "CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD",
    "PAREGO_AUGMENTED_RHO",
    "POSTERIOR_CHECK_KURTOSIS_THRESHOLD",
    "POSTERIOR_CHECK_NORMALITY_ALPHA",
    "POSTERIOR_CHECK_SKEWNESS_THRESHOLD",
    "PREDICTION_INTERVAL_DEFAULT_LEVELS",
    "PREDICTION_INTERVAL_EPSILON",
    "SENSITIVITY_HIGH_THRESHOLD",
    "SENSITIVITY_MEDIUM_THRESHOLD",
    "SENSITIVITY_OUTPUT_SCALE_MIN",
    "SENSITIVITY_PERTURBATION_FRACTION",
    "THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE",
    "THOMPSON_CANDIDATES_PER_DIM",
    "THOMPSON_MAX_CANDIDATES",
    "THOMPSON_MIN_CANDIDATES",
    "WHATIF_DEFAULT_NUM_SUGGESTIONS",
    "WHATIF_VOI_HIGH_THRESHOLD",
    "WHATIF_VOI_HYPERVOLUME_WEIGHT",
    "WHATIF_VOI_LOW_THRESHOLD",
    "WHATIF_VOI_MODERATE_THRESHOLD",
    "WHATIF_VOI_PARETO_WEIGHT",
    "WHATIF_VOI_SHIFT_WEIGHT",
    "WHATIF_VOI_UNCERTAINTY_WEIGHT",
]


# =============================================================================
# Confidence Level Thresholds
# =============================================================================
#
# These thresholds are *relative*: they apply to the candidate posterior std
# after it has been divided by the model's ``Standardize.stdvs`` (the
# training-data scale). A value of 0.1 therefore means "the predictive std is
# 10% of the objective's spread", independent of the objective's units. The
# raw posterior std is reported on the user's scale because ``Standardize``
# un-transforms it, so comparing it directly against absolute cutoffs would
# make every suggestion "low" confidence for a large-scale objective and
# "high" for a tiny-scale one. See ``suggestions_common._normalize_uncertainty``.

# Relative uncertainty below this value is "high" confidence
CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD = 0.1

# Relative uncertainty below this value is "medium" confidence (above is "low")
CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD = 0.3

# =============================================================================
# Sensitivity Analysis (Section 3.1)
# =============================================================================

# Perturbation fraction for sensitivity computation
SENSITIVITY_PERTURBATION_FRACTION = 0.01

# Threshold for high sensitivity classification
SENSITIVITY_HIGH_THRESHOLD = 0.5

# Threshold for medium sensitivity classification
SENSITIVITY_MEDIUM_THRESHOLD = 0.2

# Floor for the output-side normalization scale (std of observed y).
# Guards the division when all observations are (near-)identical, where
# no finite spread exists to express the gradient against.
SENSITIVITY_OUTPUT_SCALE_MIN = 1e-8

# =============================================================================
# Shared Confidence Levels
# =============================================================================

# Canonical confidence levels used by prediction-interval and calibration
# diagnostics.  Defined once so the two consumers cannot drift apart silently.
COMMON_CONFIDENCE_LEVELS = [0.5, 0.9, 0.95]

# =============================================================================
# Prediction Intervals (Section 3.2)
# =============================================================================

# Default confidence levels for prediction intervals
PREDICTION_INTERVAL_DEFAULT_LEVELS = COMMON_CONFIDENCE_LEVELS

# Epsilon for numerical stability in PI computations
PREDICTION_INTERVAL_EPSILON = 1e-8

# =============================================================================
# Model Calibration Diagnostics (Section 3.3)
# =============================================================================

# Confidence levels to check for calibration
CALIBRATION_CONFIDENCE_LEVELS = COMMON_CONFIDENCE_LEVELS

# Threshold for "good" calibration (mean error below this)
CALIBRATION_GOOD_THRESHOLD = 0.1

# Threshold for "poor" calibration (error above this triggers warning)
CALIBRATION_POOR_THRESHOLD = 0.2

# =============================================================================
# Posterior Predictive Checks (Section 3.4)
# =============================================================================

# Significance level for normality tests
POSTERIOR_CHECK_NORMALITY_ALPHA = 0.05

# Skewness threshold for normality warning
POSTERIOR_CHECK_SKEWNESS_THRESHOLD = 1.0

# Kurtosis threshold for normality warning
POSTERIOR_CHECK_KURTOSIS_THRESHOLD = 2.0

# =============================================================================
# Thompson Sampling (Section 3.5)
# =============================================================================

# Discrete Thompson Sampling optimizes over a finite Sobol candidate
# cloud, so the cloud must grow with dimension: a fixed 1000 points in
# >=10-d is so sparse that the argmax is nearly model-independent. The
# count follows the TuRBO tutorial's sizing
# ``min(MAX, max(MIN, PER_DIM * d))``
# (https://botorch.org/tutorials/turbo_1/).
THOMPSON_CANDIDATES_PER_DIM = 200
THOMPSON_MIN_CANDIDATES = 2000
THOMPSON_MAX_CANDIDATES = 5000

# Minimum distance for diverse batch in Thompson Sampling
THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE = 0.05

# Augmentation weight ``rho`` for the augmented Tchebycheff (ParEGO)
# scalarization ``max_k(w_k y_k) + rho * sum_k(w_k y_k)``. The small linear
# term keeps the scalarization strictly monotone (Pareto-compliant) without
# materially shifting the Tchebycheff optimum. Default from Knowles (2006).
PAREGO_AUGMENTED_RHO = 0.05

# =============================================================================
# What-If Analysis (Section 3.8)
# =============================================================================

# Default number of suggestions to compute in what-if analysis
WHATIF_DEFAULT_NUM_SUGGESTIONS = 4

# Value-of-information component weights. Each component is normalized to a
# dimensionless ~[0, 1] scale before weighting (fraction of variance removed,
# normalized suggestion shift, relative hypervolume gain, Pareto indicator), so
# the weighted sum is a unit-free score that does not change when the objective
# is rescaled. The weights express relative importance, not units.
WHATIF_VOI_UNCERTAINTY_WEIGHT = 0.5
WHATIF_VOI_SHIFT_WEIGHT = 0.1
WHATIF_VOI_HYPERVOLUME_WEIGHT = 0.3
WHATIF_VOI_PARETO_WEIGHT = 0.2

# Recommendation cutoffs applied to the dimensionless VOI score.
WHATIF_VOI_HIGH_THRESHOLD = 0.5
WHATIF_VOI_MODERATE_THRESHOLD = 0.2
WHATIF_VOI_LOW_THRESHOLD = 0.1
