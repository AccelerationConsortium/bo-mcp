"""Acquisition-function optimization: restart/sample budgets, multi-fidelity, cost-awareness."""

__all__ = [
    "ACQF_LBFGS_BATCH_LIMIT",
    "ACQF_LBFGS_MAXITER",
    "COST_AWARE_MIN_EXPECTED_COST",
    "DEFAULT_UCB_BETA",
    "MF_ACQF_BATCH_LIMIT",
    "MF_ACQF_MAXITER",
    "MF_DEFAULT_COST_WEIGHT",
    "MF_DEFAULT_FIXED_COST",
    "NUM_RESTARTS_BASE",
    "NUM_RESTARTS_MAX",
    "NUM_RESTARTS_PER_DIM",
    "RAW_SAMPLES_MAX",
    "RAW_SAMPLES_MIN",
    "RAW_SAMPLES_PER_DIM",
    "RESTART_WARN_TOLERANCE",
]


# =============================================================================
# Acquisition Optimization
# =============================================================================

# Base counts used by ``AcquisitionOptimizationConfig.resolve`` to scale
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

# Per-restart L-BFGS-B inner-loop budget passed to ``optimize_acqf`` via its
# ``options`` dict. ``batch_limit`` caps how many restarts run in one batched
# gradient step (memory/parallelism trade-off); ``maxiter`` caps the L-BFGS-B
# iterations per restart and materially affects convergence quality in high-D
# campaigns. Defaults follow the BoTorch tutorial guidance (Balandat et al.,
# 2020); centralized here (rather than inlined per call site) so continuous
# and mixed acquisition paths cannot drift.
ACQF_LBFGS_BATCH_LIMIT = 5
ACQF_LBFGS_MAXITER = 200

# Floor applied to the cost model's predicted expected cost before inverse-cost
# weighting (EIpu maximizes EI / cost, computed in log space as
# ``log EI - log cost``). A GP cost posterior can dip to (near-)zero or
# slightly negative in extrapolation; clamping keeps the log-cost term finite
# and satisfies BoTorch's strictly-positive-cost requirement for
# ``InverseCostWeightedUtility``.
COST_AWARE_MIN_EXPECTED_COST = 1e-6

# Default exploration weight for the UCB acquisition family
# (``alpha(x) = mu(x) + beta * sigma(x)``) when the spec leaves
# ``acquisition_beta`` unset. Matches BayBE's own
# ``UpperConfidenceBound.beta`` default so an unset beta produces the same
# acquisition on both backends.
DEFAULT_UCB_BETA = 0.2

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
