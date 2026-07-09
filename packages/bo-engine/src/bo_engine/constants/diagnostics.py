"""Campaign-diagnostics thresholds: health/progress status, trend detection, data-quality checks."""

__all__ = [
    "DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS",
    "DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING",
    "DIAGNOSTICS_MIN_RESULTS",
    "DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING",
    "DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL",
    "DIAGNOSTICS_MODEL_CORRELATION_WARNING",
    "DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS",
    "DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS",
    "DUPLICATE_DETECTION_TOLERANCE",
    "EXPECTED_DISTANCE_HYPERCUBE_DIVISOR",
    "EXPLOITATION_HEAVY_THRESHOLD",
    "EXPLORATION_EXPLOITATION_OFFSET",
    "EXPLORATION_HEAVY_THRESHOLD",
    "EXPLORATION_RATIO_MULTIPLIER",
    "FALLBACK_HYPERVOLUME_IMPROVEMENT",
    "HYPERVOLUME_STABILITY_THRESHOLD",
    "MAX_ITERATIONS_NO_IMPROVEMENT",
    "MAX_OBJECTIVES_VISUAL",
    "MAX_PARAMS_DESIGN_WARNING",
    "MIN_HYPERVOLUME_IMPROVEMENT",
    "MIN_IMPROVEMENT_RATE",
    "MIN_MODEL_CORRELATION",
    "OUTLIER_DETECTION_MIN_OBSERVATIONS",
    "OUTLIER_DETECTION_SIGMA_THRESHOLD",
    "PROGRESS_IMPROVING_MULTIPLIER",
    "PROGRESS_IMPROVING_THRESHOLD",
    "PROGRESS_REGRESSING_MULTIPLIER",
    "PROGRESS_REGRESSING_THRESHOLD",
    "SATISFACTION_TREND_THRESHOLD",
    "UNCERTAINTY_TREND_SLOPE_THRESHOLD",
]


# =============================================================================
# Health Status Thresholds
# =============================================================================

# Minimum hypervolume improvement rate to consider "healthy"
MIN_HYPERVOLUME_IMPROVEMENT = 0.01

# Minimum model correlation for acceptable fit
MIN_MODEL_CORRELATION = 0.5

# Maximum iterations without improvement before warning
MAX_ITERATIONS_NO_IMPROVEMENT = 5

# Minimum improvement rate for single-objective "improving" status
MIN_IMPROVEMENT_RATE = 0.1

# =============================================================================
# Diagnostics Thresholds (determine_health_status)
# =============================================================================

# Minimum results required before running diagnostics
DIAGNOSTICS_MIN_RESULTS = 3

# Iterations without improvement that triggers critical status
DIAGNOSTICS_CRITICAL_STAGNATION_ITERATIONS = 5

# Model correlation below this value triggers warning (with enough data)
DIAGNOSTICS_MODEL_CORRELATION_WARNING = 0.1

# Hypervolume decrease beyond this triggers warning
DIAGNOSTICS_HYPERVOLUME_DECREASE_WARNING = -0.05

# Minimum results required for critical low-correlation status
DIAGNOSTICS_MIN_RESULTS_FOR_CRITICAL = 10

# Model correlation below this triggers warning status
DIAGNOSTICS_MODEL_CORRELATION_WARNING_STATUS = 0.3

# Iterations without improvement that triggers warning status
DIAGNOSTICS_WARNING_STAGNATION_ITERATIONS = 3

# Minimum results required for low-correlation warning
DIAGNOSTICS_MIN_RESULTS_FOR_CORRELATION_WARNING = 5

# =============================================================================
# Progress Status Thresholds (determine_progress_status)
# =============================================================================

# Relative change threshold for "improving" progress status
PROGRESS_IMPROVING_THRESHOLD = 0.02

# Relative change threshold for "regressing" progress status
PROGRESS_REGRESSING_THRESHOLD = -0.02

# Multiplier for recent value comparison (considered improving if > 1.01)
PROGRESS_IMPROVING_MULTIPLIER = 1.01

# Multiplier for recent value comparison (considered regressing if < 0.99)
PROGRESS_REGRESSING_MULTIPLIER = 0.99

# =============================================================================
# Hypervolume Trajectory Thresholds (analyze_hypervolume_history)
# =============================================================================

# Relative change below this counts as a "no-improvement" step when scanning
# the tail of the hypervolume history for stagnation.
HYPERVOLUME_STABILITY_THRESHOLD = 0.001

# Synthetic improvement value used when only a single hypervolume reading is
# available but a non-zero hypervolume has been observed. Keeps single-step
# multi-objective campaigns out of "critical" while genuine stagnation is
# still detectable once a second sample arrives.
FALLBACK_HYPERVOLUME_IMPROVEMENT = 0.1

# =============================================================================
# Data Quality Warnings
# =============================================================================

# Threshold for number of objectives before visualization warning
MAX_OBJECTIVES_VISUAL = 4

# Threshold for number of parameters before design size warning
MAX_PARAMS_DESIGN_WARNING = 20

# =============================================================================
# Duplicate Detection Thresholds (Section 1.2)
# =============================================================================

# Tolerance for detecting duplicate parameter values
DUPLICATE_DETECTION_TOLERANCE = 1e-6

# =============================================================================
# Outlier Detection Thresholds (Section 1.3)
# =============================================================================

# Number of standard deviations for outlier detection
OUTLIER_DETECTION_SIGMA_THRESHOLD = 3.0

# Minimum observations required for outlier detection
OUTLIER_DETECTION_MIN_OBSERVATIONS = 5

# =============================================================================
# Exploration / Exploitation Metrics (diagnostics)
# =============================================================================

# Regularisation offset to prevent division by zero in exploration ratio
EXPLORATION_EXPLOITATION_OFFSET = 0.1

# Divisor for estimating expected pairwise distance in a unit hypercube
# (based on the average distance between random points in [0,1]^d)
EXPECTED_DISTANCE_HYPERCUBE_DIVISOR = 6

# Exploration ratio multiplier (maps average uncertainty to [0, 1])
EXPLORATION_RATIO_MULTIPLIER = 2

# Thresholds for classifying the exploration/exploitation balance
EXPLORATION_HEAVY_THRESHOLD = 0.65
EXPLOITATION_HEAVY_THRESHOLD = 0.35

# =============================================================================
# Uncertainty Trend Detection (diagnostics)
# =============================================================================

# Absolute relative-slope threshold for "increasing" / "decreasing" trend
UNCERTAINTY_TREND_SLOPE_THRESHOLD = 0.05

# =============================================================================
# Constraint Satisfaction Trend (diagnostics)
# =============================================================================

# Change in satisfaction rate beyond which a trend is detected
SATISFACTION_TREND_THRESHOLD = 0.1
