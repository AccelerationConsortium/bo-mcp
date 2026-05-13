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
    values: list[int] | None = None  # For discrete (explicit values)
    categories: list[str] | None = None  # For categorical
    parameter_options: dict[str, dict[str, Any]] | None = None


@dataclass(frozen=True)
class ObjectiveSpec:
    """Specification for a single objective."""

    name: str
    minimize: bool = True


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
    """

    initial_length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 10


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
