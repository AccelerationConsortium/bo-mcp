"""GP model fitting: SAASBO priors, noise/kernel priors, training/validation thresholds."""

import math

__all__ = [
    "CV_CACHE_MAX_ENTRIES",
    "HAMMING_LENGTHSCALE_PRIOR_LOC",
    "HAMMING_LENGTHSCALE_PRIOR_SCALE",
    "KERNEL_LENGTHSCALE_FLOOR",
    "MIN_DATA_ABSOLUTE",
    "MIN_DATA_PARAM_MULTIPLIER",
    "MIN_OBSERVATIONS_FOR_LOO_CV",
    "MIN_OBSERVATIONS_FOR_MODEL",
    "MODEL_VALIDATION_MAX_LENGTHSCALE_RATIO",
    "MODEL_VALIDATION_MAX_NOISE_RATIO",
    "MODEL_VALIDATION_MIN_LENGTHSCALE_RATIO",
    "NOISE_PRIOR_GAMMA_CONCENTRATION",
    "NOISE_PRIOR_GAMMA_RATE",
    "NOISE_PRIOR_MIN_INFERRED",
    "SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD",
    "SAASBO_WIDE_INTERVAL_LOG10_THRESHOLD",
    "STANDARDIZATION_MEAN_TOLERANCE",
    "STANDARDIZATION_STD_FLOOR",
    "STANDARDIZATION_VAR_TOLERANCE",
    "WARP_PRIOR_LOC",
    "WARP_PRIOR_SCALE",
    "resolve_initial_design_size",
]


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
# Kernel Lengthscale Regularization (mixed categorical kernel)
# =============================================================================

# Lower bound on kernel lengthscales, matching the ``GreaterThan(2.5e-2)``
# constraint BoTorch installs on ``SingleTaskGP``'s stock dimension-scaled
# kernel (``get_covar_module_with_dim_scaled_prior``); keeps lengthscales from
# collapsing in the small-n / high-d regime.
KERNEL_LENGTHSCALE_FLOOR = 2.5e-2

# LogNormal-prior parameters for the Hamming categorical kernel's single
# shared lengthscale. Uses the Hvarfner et al. 2024 dimension-scaled form
# ``loc = sqrt(2) + 0.5 * log(d)`` with ``d = 1`` (a category change is
# Hamming distance 1, so the block behaves as one effective dimension) and
# the stock ``scale = sqrt(3)``.
HAMMING_LENGTHSCALE_PRIOR_LOC = math.sqrt(2)
HAMMING_LENGTHSCALE_PRIOR_SCALE = math.sqrt(3)

# =============================================================================
# Model Training Thresholds
# =============================================================================

# Minimum observations before training a model (fallback to initial design)
MIN_OBSERVATIONS_FOR_MODEL = 2


def resolve_initial_design_size(n_parameters: int, requested: int | None) -> int:
    """Minimum observations to collect before switching from initial design to a fitted model.

    The GP kernel needs more data points than lengthscale hyperparameters to
    estimate, so at least ``n_parameters + 1`` observations are required,
    floored at :data:`MIN_OBSERVATIONS_FOR_MODEL`. A caller-supplied
    ``initial_design_size`` can raise this floor further (e.g. to explore
    more of the space before the first fit) but never lowers it — a request
    for fewer points than the kernel needs still waits for the floor.
    """
    floor = max(MIN_OBSERVATIONS_FOR_MODEL, n_parameters + 1)
    if requested is None:
        return floor
    return max(floor, requested)


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

# Hard size cap for the module-level CV result cache. Entries are small
# (a CVMetrics dataclass per distinct dataset/config), but a long-lived
# server process computes CV for many campaigns; the cap bounds memory
# while comfortably covering the handful of campaigns active at once.
CV_CACHE_MAX_ENTRIES = 128

# Multiplier for minimum data: requires n_params * this factor
MIN_DATA_PARAM_MULTIPLIER = 2

# Minimum observations regardless of parameters
MIN_DATA_ABSOLUTE = 3

# =============================================================================
# Model Validation Thresholds (Section 1.1)
# =============================================================================

# Minimum lengthscale ratio (lengthscale / parameter_range) before warning
MODEL_VALIDATION_MIN_LENGTHSCALE_RATIO = 0.01

# Maximum lengthscale ratio before warning
MODEL_VALIDATION_MAX_LENGTHSCALE_RATIO = 10.0

# Maximum noise-to-signal ratio (noise_variance / data_variance) before warning
MODEL_VALIDATION_MAX_NOISE_RATIO = 1.0
