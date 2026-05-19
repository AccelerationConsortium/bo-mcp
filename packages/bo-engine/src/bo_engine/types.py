"""Internal types for bo-engine.

These types define the interface between bo-engine and higher-level packages.
They are simple dataclasses/TypedDicts to keep bo-engine independent of
external Pydantic models or other frameworks.

---------------------------------------------------------------------------
Canonical sign convention (internal = minimization)
---------------------------------------------------------------------------

Every public factory and diagnostic helper in :mod:`bo_engine` operates on
objective data that has been pre-transformed to **minimization form**, i.e.
*lower is always better*.  Concretely:

* For an objective declared ``ObjectiveSpec(minimize=True)`` the caller
  passes ``train_y`` unchanged.
* For an objective declared ``ObjectiveSpec(minimize=False)`` (maximization)
  the caller must pre-negate the data: ``train_y_internal = -train_y_raw``.
* For multi-objective problems the same rule is applied column-wise using
  the ``minimize_mask`` built from the spec.

The convention applies to every tensor that carries objective values into
or out of the engine's internals, including:

* ``train_y`` / ``best_f`` arguments to acquisition factories.
* ``pareto_y`` and ``ref_point`` passed to
  :func:`bo_engine.diagnostics.compute_hypervolume`.
* ``train_y`` passed to :func:`bo_engine.reference_point.get_reference_point`.

Every factory that straddles this boundary accepts an explicit
``minimize: bool`` (or ``minimize_mask: Tensor``) keyword so the intended
direction is part of the call site instead of an implicit contract.  Helper
functions whose outputs are reported back to users (e.g.
``predicted_objectives`` in :class:`SuggestionResult`) undo the negation for
maximization objectives before returning, so the user never sees internal
form.

If you add a new helper that consumes or produces objective values,
document the convention you follow **and** require an explicit direction
argument rather than inferring it from the spec.  Silent sign flips are the
single highest-risk correctness hazard in BO code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from torch import Tensor

    from bo_engine.saasbo import SAASBOConfig

    GPModel = SingleTaskGP | ModelListGP
    OutcomeConstraintModel = tuple[SingleTaskGP, float]


class ParameterType(StrEnum):
    """Type of input parameter."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class ConstraintType(StrEnum):
    """Type of constraint."""

    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    LINEAR = "linear"


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a single input parameter.

    ``parameter_options`` carries per-backend metadata that has no neutral
    cross-backend equivalent (BayBE encoding choice, active task values,
    candidate-table mode, etc.). Keys are concrete backend names — a
    backend that does not recognize its slot must silently ignore it.
    """

    name: str
    type: ParameterType
    bounds: tuple[float, float] | None = None  # For continuous/discrete
    values: list[float] | None = None  # For discrete (explicit values; fractional ok)
    categories: list[str] | None = None  # For categorical
    parameter_options: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single objective.

    ``log_transform`` opts the objective into a ``Log → Standardize``
    outcome stack inside :mod:`bo_engine.models`. Enable it for
    multi-decade objectives (e.g. reaction rates spanning 10⁻³ … 10²)
    whose raw scale would otherwise dominate the GP's lengthscale
    fit; the model un-applies both stages on the posterior so callers
    still see results in the user's original scale.

    **Constraints (enforced at model-fit time):**

    * Requires strictly positive ``train_y`` for this objective.
      Zero or negative observations raise ``ValueError`` from the
      model factory; pre-shift the target (or drop the row) if
      non-positive outcomes can occur.
    * Requires ``minimize=True``. The maximize path negates targets
      to enforce BoTorch's internal minimization convention, which
      flips positive raw values to negative and makes the subsequent
      ``Log`` step ill-defined. Suggestion generation raises a
      ``ValueError`` for ``log_transform=True`` + ``minimize=False``.
    """

    name: str
    minimize: bool = True
    log_transform: bool = False


@dataclass(frozen=True)
class ConstraintSpec:
    """Specification for a constraint."""

    type: ConstraintType
    parameters: list[str]  # Parameter names involved
    value: float  # Constraint value (e.g., sum equals this value)
    coefficients: list[float] | None = None  # For linear constraints


class AcquisitionMethod(StrEnum):
    """Acquisition function method.

    Values are backend-agnostic semantic names. The mapping to concrete
    BoTorch classes lives inside ``bo_engine.acquisition``.
    """

    AUTO = "auto"
    NOISY_EI = "noisy_expected_improvement"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    HYPERVOLUME_IMPROVEMENT = "hypervolume_improvement"
    SCALARIZED_MULTI_OBJ = "scalarized_multi_objective"
    COST_WEIGHTED_EI = "cost_weighted_ei"
    MULTI_FIDELITY_KG = "multi_fidelity_kg"


# Maps legacy BoTorch class-name values to current semantic names.
# Used for backward compatibility with stored campaign specs.
LEGACY_ACQUISITION_VALUES: dict[str, AcquisitionMethod] = {
    "qLogNEI": AcquisitionMethod.NOISY_EI,
    "qLogEI": AcquisitionMethod.EXPECTED_IMPROVEMENT,
    "qLogNEHVI": AcquisitionMethod.HYPERVOLUME_IMPROVEMENT,
    "qLogNParEGO": AcquisitionMethod.SCALARIZED_MULTI_OBJ,
    "EIpu": AcquisitionMethod.COST_WEIGHTED_EI,
    "qMFKG": AcquisitionMethod.MULTI_FIDELITY_KG,
    "SAASBO": AcquisitionMethod.NOISY_EI,  # SAASBO is a model strategy, not acq
}


@dataclass(frozen=True)
class OutcomeConstraintSpec:
    """Specification for an outcome constraint learned from data.

    Outcome constraints define feasibility thresholds on objectives.
    A constraint GP is trained to predict P(feasible | x).
    """

    objective_name: str  # Which objective to constrain
    threshold: float  # Constraint value
    greater_than: bool = True  # True: obj >= threshold, False: obj <= threshold
    feasibility_threshold: float = 0.5  # P(feasible) cutoff (0-1)


@dataclass(frozen=True)
class FidelityParameterSpec:
    """Specification for a fidelity parameter (v2.0 Multi-Fidelity BO).

    Fidelity parameters control the approximation level of evaluations.
    Lower fidelity = cheaper but less accurate.
    """

    name: str  # Name of the fidelity parameter
    bounds: tuple[float, float]  # (min_fidelity, max_fidelity)
    target: float  # Target fidelity for final optimization (usually max)
    cost_weight: float = 1.0  # Cost scaling factor for fidelity
    fixed_cost: float = 0.0  # Fixed base cost (set > 0 if evaluations have overhead)


@dataclass(frozen=True)
class TransferLearningSpec:
    """Specification for transfer learning from prior campaigns (v2.0 RGPE).

    Allows leveraging data from prior optimization campaigns to
    accelerate optimization on a new but related task.
    """

    prior_campaign_ids: list[str]  # IDs of prior campaigns to transfer from
    num_ranking_samples: int = 512  # Samples for rank computation
    temperature: float = 0.5  # RGPE softmax temperature for weight distribution


@dataclass(frozen=True)
class TurboConfig:
    """Configuration for TuRBO trust-region optimization.

    Present = use TuRBO, absent (None) = standard acquisition optimization.

    All defaults match the canonical TuRBO paper (Eriksson et al., NeurIPS
    2019) under the assumption of unit-standardized objectives — see
    :class:`bo_engine.turbo.TurboState` for the scale assumption and the
    meaning of each tolerance.

    Attributes:
        initial_length: Initial trust region edge in normalized [0,1] input
            space (paper Algorithm 1: ``L_init = 0.8``).
        length_min: Minimum trust region edge before a restart is triggered
            (paper §3.2: ``L_min = 0.5**7 ≈ 7.8e-3``).
        length_max: Maximum trust region edge after expansion (paper §3.2:
            ``L_max = 1.6``). Larger than 1.0 lets the trust region cover the
            entire normalized input box once expanded.
        success_tolerance: Consecutive improving batches before the trust
            region doubles (paper Algorithm 1: ``tau_s = 10``).
        failure_tolerance: Consecutive non-improving batches before the trust
            region halves. ``None`` (the default) re-derives the value at
            ``TurboState`` construction time as
            ``ceil(max(4/batch, dim/batch))`` capped at
            ``TURBO_MAX_FAILURE_TOLERANCE`` so high-dimensional campaigns get
            proportionally more rope before contracting; set an explicit
            integer to override.
    """

    initial_length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 10
    failure_tolerance: int | None = None


@dataclass(frozen=True)
class AcquisitionOptimizationConfig:
    """L-BFGS-B / multi-start budget for acquisition optimization.

    Restart and raw-sample budgets must scale with problem dimensionality so
    that SAASBO and other high-D campaigns do not return shallow local
    optima. The defaults are derived from :mod:`bo_engine.constants` and grow
    linearly with the number of acquisition-input dimensions; both values are
    capped to keep CPU budget bounded.

    Attributes:
        num_restarts: Override for restart count. ``None`` means use the
            dimension-adaptive default.
        raw_samples: Override for the raw-sample budget. ``None`` means use
            the dimension-adaptive default.
    """

    num_restarts: int | None = None
    raw_samples: int | None = None

    def resolve(self, n_dims: int) -> tuple[int, int]:
        """Return the effective ``(num_restarts, raw_samples)`` for ``n_dims``.

        Caller-provided overrides on the dataclass take precedence; otherwise
        the formula
        ``num_restarts = NUM_RESTARTS_BASE + NUM_RESTARTS_PER_DIM * d`` and
        ``raw_samples = max(RAW_SAMPLES_MIN, RAW_SAMPLES_PER_DIM * d)`` apply,
        each clamped by the corresponding ``*_MAX`` constant.
        """
        from bo_engine.constants import (
            NUM_RESTARTS_BASE,
            NUM_RESTARTS_MAX,
            NUM_RESTARTS_PER_DIM,
            RAW_SAMPLES_MAX,
            RAW_SAMPLES_MIN,
            RAW_SAMPLES_PER_DIM,
        )

        d = max(int(n_dims), 1)
        restarts = (
            int(self.num_restarts)
            if self.num_restarts is not None
            else NUM_RESTARTS_BASE + NUM_RESTARTS_PER_DIM * d
        )
        samples = (
            int(self.raw_samples)
            if self.raw_samples is not None
            else max(RAW_SAMPLES_MIN, RAW_SAMPLES_PER_DIM * d)
        )
        return min(restarts, NUM_RESTARTS_MAX), min(samples, RAW_SAMPLES_MAX)


@dataclass(frozen=True)
class OptimizationSpec:
    """Full specification for an optimization problem.

    This is the main input type for bo-engine functions.

    **Backend-specific fields:** Several optional fields (``turbo_config``,
    ``saasbo_config``, ``fidelity_parameter``, ``transfer_learning``,
    ``use_cost_aware``, ``use_input_warping``) are primarily consumed by
    the BoTorch backend.  Other backends **should silently ignore** fields
    they do not support and surface warnings via
    :meth:`BOBackend.validate_spec` rather than raising.
    """

    parameters: list[ParameterSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] = field(default_factory=list)
    batch_size: int = 1
    initial_design_size: int | None = None
    # Campaign-level seed. When set, the Sobol initial-design sequence is
    # reproducible across calls — consecutive ``generate_initial_design``
    # invocations ``fast_forward`` the same low-discrepancy sequence instead
    # of restarting it with a fresh scramble. When ``None`` the Sobol engine
    # draws an OS-level scramble on every construction (prior behavior).
    random_seed: int | None = None
    # v1.0.1: Acquisition method selection
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    # v1.1: Input warping for non-stationary objectives
    use_input_warping: bool = False
    # v1.2: TuRBO for high-dimensional optimization (None = disabled)
    turbo_config: TurboConfig | None = None
    # v1.3: Outcome constraints learned from data
    outcome_constraints: list[OutcomeConstraintSpec] = field(default_factory=list)
    # v1.3: Cost-aware optimization (EIpu)
    use_cost_aware: bool = False
    # v2.0: Multi-fidelity optimization
    fidelity_parameter: FidelityParameterSpec | None = None
    # v2.0: Transfer learning from prior campaigns
    transfer_learning: TransferLearningSpec | None = None
    # v2.0: SAASBO for high-dimensional optimization (None = disabled)
    saasbo_config: SAASBOConfig | None = None
    # Acquisition-optimizer restart / raw-sample budget. Defaults scale with
    # parameter dimensionality (see ``AcquisitionOptimizationConfig``); set
    # explicit fields here to override per campaign.
    acquisition_optimization: AcquisitionOptimizationConfig = field(
        default_factory=AcquisitionOptimizationConfig
    )
    # Budget / convergence-based automatic stopping. Each field is optional;
    # when set, the suggestion entry point uses the corresponding signal to
    # short-circuit further suggestion generation. ``max_iterations`` caps
    # the number of completed BO iterations; ``max_observations`` caps the
    # number of observed results irrespective of iteration grouping;
    # ``convergence_tolerance`` is forwarded to ``detect_convergence`` as the
    # relative-improvement threshold below which the campaign is considered
    # converged.
    max_iterations: int | None = None
    max_observations: int | None = None
    convergence_tolerance: float | None = None
    # Typed backend-native option surface. Outer keys are backend names
    # (``"botorch"``, ``"baybe"``); inner dicts hold options that have no
    # neutral cross-backend equivalent. Consumed by ``validate_capabilities``
    # and the per-backend converters; backends that do not recognize a key
    # must silently ignore it.
    backend_options: dict[str, dict[str, Any]] | None = None
    # Explicit caller-side acknowledgement that the chosen backend may
    # silently degrade these option fields. Backends classify
    # semantically load-bearing options (e.g. ``outcome_constraints`` on
    # BayBE) as UNSUPPORTED by default so misroutings fail loudly at
    # intake; passing the corresponding field name here downgrades the
    # report to IGNORED so the caller opts into the degraded run with a
    # warning. ``backend="auto"`` continues to route around backends that
    # require acknowledgement for active options.
    acknowledge_degradations: tuple[str, ...] = field(default_factory=tuple)
    # Modeling strategy for ``outcome_constraints``. ``"continuous"``
    # (default) fits a regression GP on the raw constrained-objective values
    # and converts posterior samples to signed distance-to-boundary so the
    # acquisition retains gradient information near the boundary;
    # Gaussian-CDF feasibility weighting is then exact in expectation
    # (Gardner et al. ICML 2014; Letham et al. ICML 2019). ``"binary"`` is
    # the legacy path that fits a GP on binary feasibility labels — kept as
    # an opt-in fallback for genuinely binary outcomes (e.g. pass/fail
    # quality gates) where the objective value is not informative.
    outcome_constraint_method: str = "continuous"
    # Categorical-aware GP kernel routing. ``False`` (default) keeps the
    # historical one-hot + RBF behaviour. ``True`` requests an additive
    # ``RBF(continuous_dims) + CategoricalKernel(one_hot_blocks)`` kernel —
    # Hamming-style similarity on the categorical dims instead of Euclidean
    # distance on their one-hot columns. The audit identified this as a
    # modeling-efficiency concern (slower fits and lower posterior quality
    # on high-cardinality categorical specs) rather than a correctness bug;
    # the acquisition path already enumerates one-hot combinations via
    # ``optimize_acqf_mixed`` so projection drift is not in scope.
    #
    # The flag is wired to ``models.create_input_transform`` /
    # ``models.create_single_task_model``; routing to BoTorch's
    # ``MixedSingleTaskGP`` (which requires *ordinal* integer encoding for
    # the categorical block) is a separate piece of work because it
    # propagates through every transform / acquisition call site.
    use_categorical_kernel: bool = False
    # Optional override for the GP observation-noise ``GammaPrior``
    # concentration and rate. ``None`` (default) uses the
    # bo_engine.constants values calibrated for standardized targets. In
    # data-starved regimes (small n on a high-noise problem) the inferred
    # noise hyperparameter is miscalibrated by the stock prior; passing
    # ``(concentration, rate)`` lets the caller tighten or relax the
    # prior to match an empirically known noise floor. The override is
    # only consumed when ``train_yvar`` is not supplied — the
    # heteroskedastic / known-uncertainty path bypasses the prior
    # entirely (see ``models._build_likelihood``).
    noise_prior_params: tuple[float, float] | None = None
    # Auto-shift toggle for objectives that declare ``log_transform=True``
    # but may have occasional non-positive observations. When set, the
    # log-transform path computes ``shift = -min(y) + epsilon`` once at
    # fit and applies it before training — see
    # :func:`bo_engine.models.create_single_task_model` for the
    # bookkeeping. Defaults to ``False`` so the strict positivity check
    # remains the default and only opt-in callers are subject to the
    # shifted-scale posterior contract.
    #
    # **Single-objective only.** The current shift bookkeeping lives on
    # the single-task GP factory and the single-objective suggestion
    # provenance; the multi-objective path
    # (``create_model`` / ``_generate_multi_objective_batch``) does not
    # thread it through and raises ``ValueError`` when the flag is set
    # alongside more than one objective. Per-objective shifts are
    # tracked separately (Phase H follow-up).
    auto_shift_for_log: bool = False

    @property
    def use_turbo(self) -> bool:
        """Backward-compatible check for TuRBO enabled."""
        return self.turbo_config is not None

    @property
    def use_saasbo(self) -> bool:
        """Backward-compatible check for SAASBO enabled."""
        return self.saasbo_config is not None

    @property
    def n_parameters(self) -> int:
        """Number of input parameters."""
        return len(self.parameters)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    def get_parameter(self, name: str) -> ParameterSpec | None:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


@dataclass
class SuggestionResult:
    """Result of suggestion generation.

    Contains parameter values and metadata about how the suggestion was generated.

    ``model_warnings`` (when non-empty) carries diagnostic strings raised by
    post-fit checks (e.g. failed standardization invariant on a constant
    objective). The warnings are batch-level rather than per-suggestion, but
    they are stamped onto every entry of the batch so a downstream wrapper
    that operates on the full list (e.g. :class:`bo_engine.backend.SuggestionBatch`)
    can de-duplicate and lift them onto its own ``warnings`` field without
    threading a separate channel through the generator API.
    """

    parameter_values: dict[str, Any]
    iteration: int
    batch_index: int
    generation_method: str  # "initial_design" or "bo"
    random_seed: int
    acquisition_value: float | None = None
    model_uncertainty: float | None = None
    acquisition_function: str | None = None
    model_type: str | None = None
    model_version: int | None = None
    confidence_level: str | None = None
    explanation: str | None = None
    predicted_objectives: dict[str, float] | None = None  # Posterior mean per objective
    predicted_std: dict[str, float] | None = None  # Posterior std per objective
    model_warnings: tuple[str, ...] = ()


@dataclass
class ObservationData:
    """Observed data point for optimization.

    Contains parameter values and their corresponding objective values.

    ``measurement_uncertainty`` holds per-objective measurement *standard
    deviations* (one entry per objective name). When supplied for every
    observation in a campaign, the bo-engine builds ``train_yvar`` as
    ``stddev**2`` and switches the GP to a ``FixedNoiseGaussianLikelihood``,
    so the user's measurement uncertainty is trusted instead of re-estimated
    by MLL. Partial coverage (some observations missing or with a missing
    objective key) is treated as "unknown for this batch" -- the GP falls
    back to its trainable-noise prior.
    """

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    cost: float | None = None  # v1.3: Cost for cost-aware optimization
    measurement_uncertainty: dict[str, float] | None = None  # per-objective stddev


# =============================================================================
# Internal Types for Suggestion Generation
# =============================================================================


@dataclass
class GenerationContext:
    """Context for batch suggestion generation.

    Bundles parameters for _generate_single_objective_batch and
    _generate_multi_objective_batch to reduce function parameter count.
    """

    # Required parameters
    spec: OptimizationSpec
    train_x: Any  # Tensor - avoid import cycle
    train_y: Any  # Tensor
    bounds: Any  # Tensor
    batch_size: int
    iteration: int
    random_seed: int

    # Optional parameters (single-objective specific)
    turbo_state: Any | None = None  # TurboState
    observations: list[ObservationData] | None = None
    train_costs: Any | None = None  # Tensor
    pending_x: Any | None = None  # Tensor of encoded pending / in-flight points
    # Per-objective measurement noise variance ``(n_obs, n_objectives)``
    # derived from ``ObservationData.measurement_uncertainty``. ``None`` when
    # any observation lacks uncertainty data so the GP keeps trainable noise.
    train_yvar: Any | None = None  # Tensor


@dataclass
class AcquisitionConfig:
    """Configuration for acquisition function creation.

    Bundles parameters for create_acquisition to reduce function parameter count.

    ``train_y``, ``ref_point`` and ``best_f``-derived quantities are
    consumed in minimization form.  Populate ``minimize`` for
    single-objective campaigns and ``minimize_mask`` for multi-objective
    campaigns so the call site records the user-facing direction
    explicitly (see :mod:`bo_engine.types` for the sign convention).
    """

    # Required parameters
    model: GPModel | Any  # ModelListGP | SingleTaskGP (Any at runtime)
    train_x: Tensor | Any  # Tensor (Any at runtime)
    train_y: Tensor | Any  # Tensor (Any at runtime)
    n_objectives: int = 2

    # Optional parameters
    ref_point: Tensor | Any | None = None  # Tensor
    # Sign-convention bookkeeping — populate the one matching n_objectives.
    minimize: bool | None = None  # single-objective campaigns
    minimize_mask: Tensor | Any | None = None  # multi-objective campaigns
    method: AcquisitionMethod = AcquisitionMethod.AUTO
    constraints: list[Any] | None = None
    outcome_constraint_models: list[OutcomeConstraintModel] | list[tuple[Any, float]] | None = None
    cost_model: SingleTaskGP | Any | None = None  # SingleTaskGP
