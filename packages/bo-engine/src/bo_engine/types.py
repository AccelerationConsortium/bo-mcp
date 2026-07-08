"""Internal types for bo-engine.

These types define the interface between bo-engine and higher-level packages.
They are simple dataclasses/TypedDicts to keep bo-engine independent of
external Pydantic models or other frameworks.

---------------------------------------------------------------------------
Canonical sign convention (internal = maximization)
---------------------------------------------------------------------------

The acquisition pipeline in :mod:`bo_engine` operates on objective data
that has been pre-transformed to **maximization form**, i.e. *higher is
always better*.  This matches BoTorch's native convention: the qLog*
acquisition family (``qLogEI`` / ``qLogNEI`` / ``qLogNEHVI`` /
``qLogNParEGO``) has no direction flag and always treats larger sampled
values as improvements.  Concretely:

* For an objective declared ``ObjectiveSpec(minimize=True)`` the caller
  must pre-negate the data: ``train_y_internal = -train_y_raw``.
* For an objective declared ``ObjectiveSpec(minimize=False)``
  (maximization) the caller passes ``train_y`` unchanged.
* For multi-objective problems the same rule is applied column-wise using
  the ``minimize`` flags from the spec.

The convention applies to every tensor that carries objective values into
the acquisition factories: ``train_y`` / ``best_f`` arguments and the
multi-objective hypervolume ``ref_point`` (which therefore lies *below*
every observed point in internal form).

Every factory that straddles this boundary accepts an explicit
``maximize: bool`` (or ``maximize_mask: Tensor``) keyword so the data form
is part of the call site instead of an implicit contract.  Helper
functions whose outputs are reported back to users (e.g.
``predicted_objectives`` in :class:`SuggestionResult`) undo the negation
for minimization objectives before returning, so the user never sees
internal form.

Two helper families document their own, different conventions and perform
any negation internally — do not pre-convert for them:

* :func:`bo_engine.diagnostics.compute_hypervolume` and
  :func:`bo_engine.reference_point.get_reference_point` consume
  minimization-form data (their docstrings state this explicitly).
* :func:`bo_engine.turbo.update_turbo_state` takes raw values plus an
  explicit ``minimize`` flag.

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
    """Type of constraint.

    Arithmetic families (``SUM_*`` / ``PRODUCT_*`` / ``LINEAR``) compare an
    aggregate of the referenced parameters against
    :attr:`ConstraintSpec.value`. ``CARDINALITY`` bounds the number of
    *nonzero* parameters (sparsity; ``min_cardinality`` /
    ``max_cardinality``). The set-based members constrain relationships
    between the referenced parameters' assigned values: distinct values
    within a batch row (``NO_LABEL_DUPLICATES``), identical values
    (``LINKED_PARAMETERS``), or order-invariance of the parameter group
    (``PERMUTATION_INVARIANCE``). Note that ``PERMUTATION_INVARIANCE``
    (as implemented by BayBE) additionally drops candidate rows where
    the group's slots hold *equal* values — the constraint keeps one
    canonical representative per multiset of values, so "two slots, same
    value" configurations are excluded by design, not only reordered
    duplicates.
    """

    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    LINEAR = "linear"
    PRODUCT_EQUALS = "product_equals"
    PRODUCT_LESS_THAN = "product_less_than"
    PRODUCT_GREATER_THAN = "product_greater_than"
    CARDINALITY = "cardinality"
    NO_LABEL_DUPLICATES = "no_label_duplicates"
    LINKED_PARAMETERS = "linked_parameters"
    PERMUTATION_INVARIANCE = "permutation_invariance"


# Constraint families grouped by shared field requirements. Kept next to the
# enum so intake validation, capability reports, and converters share one
# curation instead of re-deriving membership.
PRODUCT_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.PRODUCT_EQUALS,
        ConstraintType.PRODUCT_LESS_THAN,
        ConstraintType.PRODUCT_GREATER_THAN,
    }
)
SET_BASED_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.NO_LABEL_DUPLICATES,
        ConstraintType.LINKED_PARAMETERS,
        ConstraintType.PERMUTATION_INVARIANCE,
    }
)
ARITHMETIC_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.SUM_EQUALS,
        ConstraintType.SUM_LESS_THAN,
        ConstraintType.SUM_GREATER_THAN,
        ConstraintType.LINEAR,
        *PRODUCT_CONSTRAINT_TYPES,
    }
)
# Constraint types eligible for the interpoint (across-batch) flag: the
# continuous linear/sum family that maps to BayBE's
# ``ContinuousLinearConstraint(interpoint=True)``.
INTERPOINT_CONSTRAINT_TYPES: frozenset[ConstraintType] = frozenset(
    {
        ConstraintType.SUM_EQUALS,
        ConstraintType.SUM_LESS_THAN,
        ConstraintType.SUM_GREATER_THAN,
        ConstraintType.LINEAR,
    }
)


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


class TargetMode(StrEnum):
    """Optimization direction / goal of a single objective.

    ``MATCH`` targets a specific value (``ObjectiveSpec.target_value``)
    instead of a direction — the common lab ask "hit pH 7.4" — with the
    distance-to-target shape selected by :class:`MatchShape`.
    """

    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"
    MATCH = "match"


class MatchShape(StrEnum):
    """Distance-to-target shape for ``TargetMode.MATCH`` objectives.

    ``ABSOLUTE`` / ``QUADRATIC`` penalize the (squared) distance without
    extra parameters; ``BELL`` and ``TRIANGULAR`` are normalized kernels
    that additionally need a width (``ObjectiveSpec.match_scale``: the
    bell's sigma / the triangle's total base width).
    """

    ABSOLUTE = "absolute"
    QUADRATIC = "quadratic"
    BELL = "bell"
    TRIANGULAR = "triangular"


class ObjectiveTransformKind(StrEnum):
    """Typed target-transformation union (generalizes ``log_transform``)."""

    LOG = "log"
    CLAMP = "clamp"
    POWER = "power"
    SIGMOID = "sigmoid"


@dataclass(frozen=True)
class ObjectiveTransformSpec:
    """One target transformation applied to an objective's raw values.

    Field usage per :class:`ObjectiveTransformKind`:

    * ``LOG`` — no parameters; identical contract to the legacy
      ``log_transform=True`` boolean (strictly positive targets,
      minimize-only).
    * ``CLAMP`` — ``bounds`` required; values outside are clipped.
    * ``POWER`` — ``exponent`` (integer) required.
    * ``SIGMOID`` — ``center`` and ``steepness`` required; the normalized
      logistic ``1 / (1 + exp(-steepness * (x - center)))``.
    """

    kind: ObjectiveTransformKind
    bounds: tuple[float, float] | None = None
    exponent: int | None = None
    center: float | None = None
    steepness: float | None = None


class ScalarizationMode(StrEnum):
    """Multi-objective combination strategy.

    ``PARETO`` (default) optimizes the full front; ``DESIRABILITY``
    scalarizes normalized targets into a single figure of merit using the
    per-objective ``weight`` fields and the spec-level ``scalarizer``.
    """

    PARETO = "pareto"
    DESIRABILITY = "desirability"


class ScalarizerKind(StrEnum):
    """Weighted-mean flavor for ``ScalarizationMode.DESIRABILITY``."""

    MEAN = "mean"
    GEOM_MEAN = "geom_mean"


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
    * Requires ``minimize=True``. The minimize path's maximization-form
      negation is handled inside the model factory (a ``Negate`` stage
      un-negates before ``Log`` and re-negates the posterior); the
      maximize combination stays outside the supported contract and
      suggestion generation raises a ``ValueError`` for
      ``log_transform=True`` + ``minimize=False``.

    **Extended target surface** (validated at intake; currently honored
    by the BayBE backend, reported ``UNSUPPORTED`` by BoTorch so
    ``backend="auto"`` routes correctly):

    * ``target_mode`` — optional restatement of the optimization goal.
      ``MATCH`` mode drives the campaign toward ``target_value`` with
      the ``match_shape`` distance kernel (``match_scale`` sets the
      bell sigma / triangle width). For ``MINIMIZE``/``MAXIMIZE`` the
      resolution contract is :attr:`effective_mode` (``target_mode``
      wins over the boolean) — but note that only the BayBE converter
      resolves direction through ``effective_mode``; the BoTorch
      pipeline reads the boolean ``minimize`` field directly and its
      ``validate_capabilities`` reports ``UNSUPPORTED`` whenever
      ``target_mode`` disagrees with the boolean, so a direct-mode
      override never silently flips direction on one backend only.
      Callers building specs by hand should set ``minimize``
      consistently with ``target_mode`` (the served REST/MCP converter
      always does).
    * ``weight`` / ``normalization_bounds`` — desirability inputs: the
      scalarization weight and the raw-value range mapped onto [0, 1]
      (required for minimize/maximize objectives under
      ``ScalarizationMode.DESIRABILITY``; bell/triangular match targets
      are already normalized).
    * ``transform`` — typed target transformation
      (:class:`ObjectiveTransformSpec`); mutually exclusive with the
      legacy ``log_transform`` boolean, which remains supported and maps
      to the ``LOG`` kind.
    """

    name: str
    minimize: bool = True
    log_transform: bool = False
    target_mode: TargetMode | None = None
    target_value: float | None = None
    match_shape: MatchShape | None = None
    match_scale: float | None = None
    weight: float | None = None
    normalization_bounds: tuple[float, float] | None = None
    transform: ObjectiveTransformSpec | None = None

    @property
    def effective_mode(self) -> TargetMode:
        """Resolved target mode: explicit ``target_mode`` wins over ``minimize``."""
        if self.target_mode is not None:
            return self.target_mode
        return TargetMode.MINIMIZE if self.minimize else TargetMode.MAXIMIZE


@dataclass(frozen=True)
class ConstraintSpec:
    """Specification for a constraint.

    ``value`` is the arithmetic threshold (``SUM_*`` / ``PRODUCT_*`` /
    ``LINEAR``); it is unused by the cardinality and set-based families
    (which is why it now carries a neutral default). ``min_cardinality`` /
    ``max_cardinality`` bound the count of nonzero parameters for
    ``CARDINALITY`` constraints. ``is_interpoint`` switches a continuous
    linear/sum constraint from per-point to across-the-batch semantics
    (BayBE ``ContinuousLinearConstraint(interpoint=True)``): the aggregate
    is taken over all points of one recommended batch instead of holding
    for each point individually.
    """

    type: ConstraintType
    parameters: list[str]  # Parameter names involved
    value: float = 0.0  # Constraint value (e.g., sum equals this value)
    coefficients: list[float] | None = None  # For linear constraints
    min_cardinality: int | None = None  # CARDINALITY only
    max_cardinality: int | None = None  # CARDINALITY only
    is_interpoint: bool = False  # Continuous linear/sum only


class AcquisitionMethod(StrEnum):
    """Acquisition function method.

    Values are backend-agnostic semantic names. The mapping to concrete
    BoTorch classes lives inside ``bo_engine.acquisition``; the BayBE
    mapping lives in ``bo_engine_baybe.converters``. Not every member is
    expressible on every backend — each backend's
    ``validate_capabilities`` classifies unmappable members as
    ``UNSUPPORTED`` so ``backend="auto"`` routes to a backend that honors
    the request and a pinned incompatible backend fails loudly.

    Semantic families:

    * Improvement-based: ``NOISY_EI`` / ``EXPECTED_IMPROVEMENT`` (log
      variants, the defaults) and their explicit non-log siblings
      ``*_NONLOG`` for callers that need the classic formulation.
    * Exploration: ``UPPER_CONFIDENCE_BOUND`` (tunable ``acquisition_beta``)
      and ``POSTERIOR_STANDARD_DEVIATION`` (pure exploration).
    * Exploitation: ``POSTERIOR_MEAN`` and ``SIMPLE_REGRET`` (its
      Monte-Carlo counterpart).
    * Active learning: ``ACTIVE_LEARNING`` (negated integrated posterior
      variance, qNIPV).
    * Lookahead / randomized: ``KNOWLEDGE_GRADIENT``, ``THOMPSON_SAMPLING``.
    * Multi-objective: ``HYPERVOLUME_IMPROVEMENT`` (+ ``_NONLOG``) and
      ``SCALARIZED_MULTI_OBJ``.
    """

    AUTO = "auto"
    NOISY_EI = "noisy_expected_improvement"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    HYPERVOLUME_IMPROVEMENT = "hypervolume_improvement"
    SCALARIZED_MULTI_OBJ = "scalarized_multi_objective"
    COST_WEIGHTED_EI = "cost_weighted_ei"
    MULTI_FIDELITY_KG = "multi_fidelity_kg"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    SIMPLE_REGRET = "simple_regret"
    POSTERIOR_MEAN = "posterior_mean"
    POSTERIOR_STANDARD_DEVIATION = "posterior_standard_deviation"
    THOMPSON_SAMPLING = "thompson_sampling"
    KNOWLEDGE_GRADIENT = "knowledge_gradient"
    ACTIVE_LEARNING = "active_learning"
    EXPECTED_IMPROVEMENT_NONLOG = "expected_improvement_nonlog"
    NOISY_EI_NONLOG = "noisy_expected_improvement_nonlog"
    HYPERVOLUME_IMPROVEMENT_NONLOG = "hypervolume_improvement_nonlog"


# Acquisition methods that accept the ``acquisition_beta``
# exploration-weight parameter. ``beta`` on any other member is rejected
# at intake and at acquisition construction so the knob can never leak
# into an acquisition function that silently ignores it.
UCB_FAMILY_ACQUISITION: frozenset[AcquisitionMethod] = frozenset(
    {AcquisitionMethod.UPPER_CONFIDENCE_BOUND}
)


# Acquisition methods with single-objective semantics only — no
# hypervolume/scalarized counterpart exists, so a multi-objective spec
# requesting one falls back to the objective-family default acquisition
# on both backends. The dispatch fallback is sanctioned, but it must be
# *classified*: each backend's ``validate_capabilities`` reports the
# combination (UNSUPPORTED by default, IGNORED once
# ``'acquisition_method'`` is acknowledged) so the request and any
# attached ``acquisition_beta`` can never be discarded silently. The
# improvement-based members (EI / noisy-EI and their non-log siblings)
# are deliberately absent: they resolve *within* the objective family to
# their hypervolume analogues rather than being dropped.
SINGLE_OBJECTIVE_ONLY_ACQUISITION: frozenset[AcquisitionMethod] = frozenset(
    {
        AcquisitionMethod.UPPER_CONFIDENCE_BOUND,
        AcquisitionMethod.PROBABILITY_OF_IMPROVEMENT,
        AcquisitionMethod.SIMPLE_REGRET,
        AcquisitionMethod.POSTERIOR_MEAN,
        AcquisitionMethod.POSTERIOR_STANDARD_DEVIATION,
        AcquisitionMethod.THOMPSON_SAMPLING,
        AcquisitionMethod.KNOWLEDGE_GRADIENT,
        AcquisitionMethod.ACTIVE_LEARNING,
    }
)


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

    The former ``temperature`` field is gone: RGPE weights are the
    paper's ranking-loss argmin counts (see
    :mod:`bo_engine.transfer_learning`), which involve no softmax and
    therefore no temperature to tune.
    """

    prior_campaign_ids: list[str]  # IDs of prior campaigns to transfer from
    num_ranking_samples: int = 512  # Samples for rank computation


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
    # Exploration weight for the UCB acquisition family
    # (``alpha(x) = mu(x) + beta * sigma(x)``). Only valid together with a
    # member of ``UCB_FAMILY_ACQUISITION``; both backends validate the
    # combination at intake and the acquisition factories reject a stray
    # ``beta`` at construction time. ``None`` uses the backend default
    # (``bo_engine.constants.DEFAULT_UCB_BETA``, matching BayBE's own
    # ``UpperConfidenceBound.beta`` default).
    acquisition_beta: float | None = None
    # Multi-objective combination strategy. ``PARETO`` (default) keeps the
    # existing front-based behavior; ``DESIRABILITY`` scalarizes normalized
    # targets using per-objective ``ObjectiveSpec.weight`` values and the
    # ``scalarizer`` below. Honored by BayBE (``DesirabilityObjective``);
    # BoTorch reports it UNSUPPORTED so ``auto`` routes to BayBE.
    scalarization: ScalarizationMode = ScalarizationMode.PARETO
    # Weighted-mean flavor for desirability scalarization. ``None`` uses
    # the backend default (BayBE: geometric mean).
    scalarizer: ScalarizerKind | None = None
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
    # Optional durable cross-system identity. Backends that
    # serialise per-observation state (notably BayBE, which keeps an
    # ``observation_identity`` index) use this as the discriminator for
    # otherwise-identical replicate rows so a specific replicate can be
    # tied to the same storage row across reorder / restart cycles. The
    # MCP server populates it with ``Result.id`` (a UUID string) via
    # ``helpers.results_to_observations``; direct bo-engine callers that
    # do not need cross-system addressing can leave it ``None`` and the
    # backend falls back to a within-batch fingerprint as before.
    result_id: str | None = None


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
    consumed in maximization form.  Populate ``maximize`` for
    single-objective campaigns and ``maximize_mask`` for multi-objective
    campaigns so the call site records the data form explicitly (see
    :mod:`bo_engine.types` for the sign convention).
    """

    # Required parameters
    model: GPModel | Any  # ModelListGP | SingleTaskGP (Any at runtime)
    train_x: Tensor | Any  # Tensor (Any at runtime)
    train_y: Tensor | Any  # Tensor (Any at runtime)
    n_objectives: int = 2

    # Optional parameters
    ref_point: Tensor | Any | None = None  # Tensor
    # Sign-convention bookkeeping — populate the one matching n_objectives.
    maximize: bool | None = None  # single-objective campaigns
    maximize_mask: Tensor | Any | None = None  # multi-objective campaigns
    method: AcquisitionMethod = AcquisitionMethod.AUTO
    constraints: list[Any] | None = None
    outcome_constraint_models: list[OutcomeConstraintModel] | list[tuple[Any, float]] | None = None
    cost_model: SingleTaskGP | Any | None = None
    # UCB-family exploration weight; see ``OptimizationSpec.acquisition_beta``.
    beta: float | None = None
