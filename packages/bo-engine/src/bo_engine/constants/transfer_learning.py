"""RGPE transfer learning and cross-campaign similarity weighting."""

__all__ = [
    "RGPE_DILUTION_BASE_QUANTILE",
    "RGPE_DILUTION_TARGET_QUANTILE",
    "RGPE_HELPFUL_WEIGHT_THRESHOLD",
    "RGPE_MIN_TARGET_OBSERVATIONS",
    "RGPE_NUM_SAMPLES",
    "RGPE_PENDING_PENALTY_LENGTHSCALE",
    "TRANSFER_WEIGHT_BOUNDS",
    "TRANSFER_WEIGHT_DATA_RICHNESS",
    "TRANSFER_WEIGHT_OBJECTIVE",
    "TRANSFER_WEIGHT_PARAMETER",
]


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
