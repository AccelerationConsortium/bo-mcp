"""Constants for Bayesian Optimization configuration.

This module centralizes magic numbers and threshold values used throughout
the bo-engine package, making them easy to understand, tune, and override.
"""

# =============================================================================
# SAASBO Active-Dimension Threshold
# =============================================================================

# Median per-dimension lengthscale above which the SAAS prior considers a
# dimension "inactive". Eriksson & Jankowiak (UAI 2021, §3.3 & Appendix B)
# report that truly inactive dimensions converge to lengthscales of order
# 1e2-1e3 under the sparsity-inducing half-Cauchy prior; values below ~10
# remain attached to the signal. We use 10.0 as the conservative cut-off so
# the boolean mask reports an "inactive" dimension only when the prior has
# clearly pushed it past the noise threshold, in keeping with the paper's
# recommendation.
SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD = 10.0

# Order-of-magnitude width (in log10 units) above which a SAASBO posterior
# interval is considered "wide" and the corresponding dimension is flagged
# with ``confident=False``. Early in a campaign the half-Cauchy posterior
# can span several orders of magnitude and the median importance is
# unreliable for pruning decisions; pinning the threshold at 1 order of
# magnitude follows Eriksson & Jankowiak's UAI-2021 recommendation in
# §3.3 / Appendix B.
SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD = 1.0

# =============================================================================
# GP Noise Prior (likelihood)
# =============================================================================

# Default ``GammaPrior`` shape/rate for the GP observation-noise hyperparameter.
# Mildly informative; matches the BoTorch single-task tutorial defaults
# (Eriksson & Jankowiak, UAI 2021; BoTorch reference implementation
# https://botorch.org). Operates on standardized (unit-variance) targets — see
# ``models.create_single_task_model`` for the standardization convention.
NOISE_PRIOR_GAMMA_CONCENTRATION = 1.1
NOISE_PRIOR_GAMMA_RATE = 0.05

# Lower bound applied to the inferred noise hyperparameter via ``GreaterThan``
# to keep the GP Cholesky factor well conditioned under noisy / multi-scale
# objectives.
NOISE_PRIOR_MIN_INFERRED = 1e-4

# Default LogNormal-prior parameters for the Kumaraswamy warp concentrations
# (see ``models.create_input_transform``). Calibrated to BoTorch's stock prior
# under the post-Normalize unit-cube domain — the input transform is ordered
# ``normalize → warp`` and the prior assumes a [0, 1] input space.
WARP_PRIOR_LOC = 0.0
WARP_PRIOR_SCALE = 0.75

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

# Tolerances for verifying output standardization (see
# models.verify_standardization). BoTorch's `Standardize(m=1)` uses `nanstd`
# (ddof=1 / sample stdv) to normalize targets, so the post-transform *unbiased*
# variance is exactly 1.0 up to float jitter; these tolerances bound numerical
# noise, not statistical slack.
STANDARDIZATION_MEAN_TOLERANCE = 1e-5
STANDARDIZATION_VAR_TOLERANCE = 1e-4

# Minimum standard deviation applied to a sub-model whose raw training targets
# are (near-)constant. BoTorch's ``Standardize(m=1)`` divides by an empirical
# stddev that can collapse to ~0 on replicate / constant data, which inflates
# posterior variance and produces spurious acquisition spikes. The verifier in
# ``models.verify_standardization`` floors the *floor used for the unit-variance
# assertion*; the model factory floors the post-standardize stddev attribute
# itself so subsequent posterior evaluation stays numerically stable.
STANDARDIZATION_STD_FLOOR = 1e-4

# Minimum observations for meaningful LOO cross-validation
MIN_OBSERVATIONS_FOR_LOO_CV = 5

# Multiplier for minimum data: requires n_params * this factor
MIN_DATA_PARAM_MULTIPLIER = 2

# Minimum observations regardless of parameters
MIN_DATA_ABSOLUTE = 3

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

# =============================================================================
# Acquisition Optimization
# =============================================================================

# Base counts used by ``AcquisitionOptimizationConfig.for_dimension`` to scale
# restart count and raw-sample budget with problem dimensionality. The
# defaults follow the BoTorch tutorial guidance to grow restart density with
# acquisition multimodality (Balandat et al., 2020, §6.1) instead of leaving a
# single fixed value that under-performs in SAASBO / high-D campaigns.
#
# Effective values per call:
#   num_restarts = NUM_RESTARTS_BASE + NUM_RESTARTS_PER_DIM * d
#   raw_samples  = max(RAW_SAMPLES_MIN, RAW_SAMPLES_PER_DIM * d)
NUM_RESTARTS_BASE = 10
NUM_RESTARTS_PER_DIM = 2

RAW_SAMPLES_MIN = 512
RAW_SAMPLES_PER_DIM = 32

# Hard upper bounds so very-high-D campaigns do not blow up CPU budget.
# Picked at ~4× the linear-extrapolation value for d=20 (the SAASBO threshold)
# so the cap only bites well beyond the typical campaign size.
NUM_RESTARTS_MAX = 200
RAW_SAMPLES_MAX = 8192

# Relative gap between the best and the median restart acquisition value
# below which ``optimize_acquisition`` emits a "widespread local minima"
# warning. The check guards against silent restart collapse — when most
# restarts converge to acquisition values close to the best, multi-start is
# no longer probing distinct basins and the BO algorithm is likely trapped.
RESTART_WARN_TOLERANCE = 0.05

# Legacy compatibility aliases. Existing callers still reference these names;
# they resolve to the base counts used by the dimension-adaptive formula.
NUM_RESTARTS = NUM_RESTARTS_BASE
RAW_SAMPLES = RAW_SAMPLES_MIN

# Floor applied to the cost model's predicted expected cost before inverse-cost
# weighting (EIpu = EI / cost). A GP cost posterior can dip to (near-)zero or
# slightly negative in extrapolation; clamping keeps the division well-defined
# and satisfies BoTorch's strictly-positive-cost requirement for
# ``InverseCostWeightedUtility``.
COST_AWARE_MIN_EXPECTED_COST = 1e-6

# =============================================================================
# Multi-Fidelity Optimization (qMFKG)
# =============================================================================

# Default ``AffineFidelityCostModel`` parameters: cost(x) = fixed_cost +
# cost_weight * fidelity. The fixed base cost dominates at low fidelity so
# the cost-aware utility favours cheap exploratory evaluations early; the
# weight scales the marginal cost of moving toward the target fidelity. These
# are the single source of truth for both ``FidelitySpec`` and
# ``create_cost_model`` so the two default sites cannot drift apart.
MF_DEFAULT_FIXED_COST = 5.0
MF_DEFAULT_COST_WEIGHT = 1.0

# ``optimize_acqf`` options for the qMFKG inner optimization. ``batch_limit``
# caps how many restart candidates L-BFGS-B optimizes simultaneously (memory
# vs. throughput), and ``maxiter`` bounds the L-BFGS-B iterations per restart.
# Matches the BoTorch multi-fidelity tutorial defaults (Balandat et al., 2020).
MF_ACQF_BATCH_LIMIT = 5
MF_ACQF_MAXITER = 200

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
# Random Seeds
# =============================================================================

# Maximum random seed value (2^31 - 1 for compatibility)
MAX_RANDOM_SEED = 2**31 - 1

# Per-iteration offset for deriving reproducible acquisition seeds from
# ``spec.random_seed``. Each call to ``generate_next_batch`` at a different
# iteration gets a distinct but deterministic seed via
# ``(spec.random_seed + iteration * SEED_ITERATION_OFFSET) % MAX_RANDOM_SEED``.
# The value is a large odd prime that is coprime to ``MAX_RANDOM_SEED`` so the
# modular stride visits every residue before cycling, spreading consecutive
# iterations across the seed range instead of clustering near adjacent values.
SEED_ITERATION_OFFSET = 1_000_003

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

# Minimum number of target observations required before rank-based
# weighting is meaningful; below this every model gets a uniform weight
# (Feurer et al. 2018: LOO models need >= 2 points, "we start the
# weighting procedure only when we have gathered three observations").
RGPE_MIN_TARGET_OBSERVATIONS = 3

# Weight-dilution prevention (Feurer et al. 2018, v1 percentile rule):
# a base model is discarded when the RGPE_DILUTION_BASE_QUANTILE of its
# ranking-loss samples is >= the RGPE_DILUTION_TARGET_QUANTILE of the
# target model's ranking-loss samples. The paper uses the base median
# vs the target 95th percentile and reports the 95 threshold as
# non-critical in a sensitivity analysis.
RGPE_DILUTION_BASE_QUANTILE = 0.5
RGPE_DILUTION_TARGET_QUANTILE = 0.95

# Lengthscale (normalized [0,1] input space) of the Gaussian local
# penalizer that conditions RGPE acquisition values on pending batch
# points (Gonzalez et al. 2016, "Batch BO via Local Penalization").
RGPE_PENDING_PENALTY_LENGTHSCALE = 0.1

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

# Number of candidates to consider
THOMPSON_NUM_CANDIDATES = 1000

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

# =============================================================================
# Z-Scores for Confidence Intervals
# =============================================================================

# Z-score for 95 % confidence interval (normal distribution)
CI_95_Z_SCORE = 1.96
