"""Constants for Bayesian Optimization configuration.

This module centralizes magic numbers and threshold values used throughout
the bo-engine package, making them easy to understand, tune, and override.
"""

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
# Model Training Thresholds
# =============================================================================

# Minimum observations before training a model (fallback to initial design)
MIN_OBSERVATIONS_FOR_MODEL = 2

# Minimum observations for meaningful LOO cross-validation
MIN_OBSERVATIONS_FOR_LOO_CV = 5

# Multiplier for minimum data: requires n_params * this factor
MIN_DATA_PARAM_MULTIPLIER = 2

# Minimum observations regardless of parameters
MIN_DATA_ABSOLUTE = 3

# =============================================================================
# Confidence Level Thresholds
# =============================================================================

# Uncertainty below this value is "high" confidence
CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD = 0.1

# Uncertainty below this value is "medium" confidence (above is "low")
CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD = 0.3

# =============================================================================
# TuRBO Configuration
# =============================================================================

# Initial trust region length (normalized [0,1] space)
TURBO_INITIAL_LENGTH = 0.8

# Minimum trust region length before restart
TURBO_LENGTH_MIN = 0.5**7  # ~0.0078

# Maximum trust region length after expansion
TURBO_LENGTH_MAX = 1.6

# Number of consecutive successes before expanding trust region
TURBO_SUCCESS_TOLERANCE = 10

# Trust region expansion factor (multiply by this on success)
TURBO_EXPANSION_FACTOR = 2.0

# Trust region contraction factor (divide by this on failure)
TURBO_CONTRACTION_FACTOR = 2.0

# =============================================================================
# Acquisition Optimization
# =============================================================================

# Number of random restarts for L-BFGS-B optimization
NUM_RESTARTS = 10

# Number of raw samples for initial candidates
RAW_SAMPLES = 512

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

# =============================================================================
# Reference Point Computation (Multi-objective)
# =============================================================================

# Padding factor for reference point beyond worst observed
REFERENCE_POINT_PADDING = 0.1

# Minimum range to avoid numerical issues
MIN_OBJECTIVE_RANGE = 1e-6

# =============================================================================
# Numerical Stability
# =============================================================================

# Tolerance for improvement detection (relative)
IMPROVEMENT_TOLERANCE_RELATIVE = 1e-3

# Tolerance for improvement detection (absolute, for zero values)
IMPROVEMENT_TOLERANCE_ABSOLUTE = 1e-6

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
# Random Seeds
# =============================================================================

# Maximum random seed value (2^31 - 1 for compatibility)
MAX_RANDOM_SEED = 2**31 - 1

# =============================================================================
# Data Quality Warnings
# =============================================================================

# Threshold for number of objectives before visualization warning
MAX_OBJECTIVES_VISUAL = 4

# Threshold for number of parameters before design size warning
MAX_PARAMS_DESIGN_WARNING = 20

# =============================================================================
# Model Validation Thresholds (Section 1.1)
# =============================================================================

# Minimum lengthscale ratio (lengthscale / parameter_range) before warning
MODEL_VALIDATION_MIN_LENGTHSCALE_RATIO = 0.01

# Maximum lengthscale ratio before warning
MODEL_VALIDATION_MAX_LENGTHSCALE_RATIO = 10.0

# Maximum noise-to-signal ratio (noise_variance / data_variance) before warning
MODEL_VALIDATION_MAX_NOISE_RATIO = 1.0

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
# Early Stopping / Convergence Detection (Section 1.4)
# =============================================================================

# Window size for convergence detection
CONVERGENCE_WINDOW_SIZE = 5

# Relative improvement threshold below which convergence is detected
CONVERGENCE_IMPROVEMENT_THRESHOLD = 0.01

# Minimum number of observations for convergence detection
CONVERGENCE_MIN_OBSERVATIONS = 10

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

# =============================================================================
# Dynamic Reference Point (Section 2.1)
# =============================================================================

# Adaptation rate for dynamic reference point (0-1, higher = faster adaptation)
REFERENCE_POINT_ADAPTATION_RATE = 0.2

# Minimum margin to maintain for numerical stability
REFERENCE_POINT_MIN_MARGIN = 0.01

# =============================================================================
# RGPE Transfer Learning (Section 2.2)
# =============================================================================

# Default number of samples for RGPE ranking computation
RGPE_NUM_SAMPLES = 512

# Minimum weight threshold for considering a prior task helpful
RGPE_HELPFUL_WEIGHT_THRESHOLD = 0.1

# =============================================================================
# Transfer Learning Similarity Weights
# =============================================================================

# Weights for computing overall similarity between campaigns for transfer learning.
# Higher weights = more influence on the overall score.
TRANSFER_WEIGHT_PARAMETER = 0.4
TRANSFER_WEIGHT_OBJECTIVE = 0.3
TRANSFER_WEIGHT_BOUNDS = 0.2
TRANSFER_WEIGHT_DATA_RICHNESS = 0.1

# =============================================================================
# Outcome Constraint Modeling (Section 2.3)
# =============================================================================

# Default probability threshold for constraint satisfaction
CONSTRAINT_PROBABILITY_THRESHOLD = 0.5

# Weight for expected constraint violation in acquisition
CONSTRAINT_VIOLATION_WEIGHT = 1.0

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

# =============================================================================
# Sensitivity Analysis (Section 3.1)
# =============================================================================

# Perturbation fraction for sensitivity computation
SENSITIVITY_PERTURBATION_FRACTION = 0.01

# Threshold for high sensitivity classification
SENSITIVITY_HIGH_THRESHOLD = 0.5

# Threshold for medium sensitivity classification
SENSITIVITY_MEDIUM_THRESHOLD = 0.2

# =============================================================================
# Prediction Intervals (Section 3.2)
# =============================================================================

# Default confidence levels for prediction intervals
PREDICTION_INTERVAL_DEFAULT_LEVELS = [0.5, 0.9, 0.95]

# Epsilon for numerical stability in PI computations
PREDICTION_INTERVAL_EPSILON = 1e-8

# =============================================================================
# Model Calibration Diagnostics (Section 3.3)
# =============================================================================

# Confidence levels to check for calibration
CALIBRATION_CONFIDENCE_LEVELS = [0.5, 0.9, 0.95]

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

# Number of posterior samples for Thompson Sampling (5+ recommended
# for better exploration-exploitation tradeoff)
THOMPSON_NUM_POSTERIOR_SAMPLES = 5

# Number of candidates to consider
THOMPSON_NUM_CANDIDATES = 1000

# Minimum distance for diverse batch in Thompson Sampling
THOMPSON_BATCH_DIVERSITY_MIN_DISTANCE = 0.05

# =============================================================================
# What-If Analysis (Section 3.8)
# =============================================================================

# Default number of suggestions to compute in what-if analysis
WHATIF_DEFAULT_NUM_SUGGESTIONS = 4
