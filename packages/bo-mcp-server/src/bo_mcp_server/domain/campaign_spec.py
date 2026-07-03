"""Campaign specification value object.

The neutral parameter, constraint, and acquisition-method enums live in
:mod:`bo_engine.types` (the lower-dependency package). They are re-exported
from this module so domain consumers keep the historic ``bo_mcp_server.domain``
import path; both names refer to the same enum object, so converters no
longer need a manual mapping layer.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from bo_engine.types import (
    LEGACY_ACQUISITION_VALUES as _LEGACY_ACQUISITION_VALUES,
)
from bo_engine.types import (
    AcquisitionMethod,
    ConstraintType,
    ParameterType,
)


def _freeze_option_value(value: object) -> object:
    """Recursively convert an option value into a deeply immutable shape.

    Mappings become :class:`types.MappingProxyType` over a freshly-copied
    dict whose values are themselves recursively frozen. Lists become
    tuples; tuples are walked recursively. Atoms (str / int / float /
    bool / None / enums / anything not list-or-mapping-like) pass through
    unchanged. The result is safe to expose from a frozen value object
    because every reachable container is read-only — neither
    ``param.parameter_options["baybe"]["nested"]["a"] = …`` nor
    ``param.parameter_options["baybe"]["items"].append(…)`` can land.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_option_value(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_option_value(v) for v in value)
    return value


def _freeze_backend_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Mapping[str, Any]] | None:
    """Deeply wrap a ``backend → option-dict`` mapping in read-only views.

    The outer mapping (keyed by backend name) and every nested
    mapping / sequence inside each backend's option dict is converted
    via :func:`_freeze_option_value`, so attempts like
    ``param.parameter_options["baybe"]["nested"]["a"] = …`` and
    ``param.parameter_options["baybe"]["items"].append(…)`` both raise
    instead of silently mutating shared state. Defensive copying ensures
    a held reference to the original source dict cannot mutate the
    frozen view either.

    The transformation runs in ``field_validator(mode="after")``; JSON
    round-trips go through :func:`_serialize_backend_options` which
    walks the frozen views back into plain dicts and lists.
    """
    if value is None:
        return None
    frozen = {k: cast("Mapping[str, Any]", _freeze_option_value(v)) for k, v in value.items()}
    return MappingProxyType(frozen)


def _thaw_option_value(value: object) -> object:
    """Recursively materialize a frozen option value back to plain dict / list.

    Inverse of :func:`_freeze_option_value`: ``MappingProxyType`` becomes
    ``dict``, ``tuple`` becomes ``list``. Used by the field serializers
    so JSON output never leaks ``mappingproxy`` (which ``json.dumps``
    cannot encode) and so external consumers see the same shape they
    submitted at construction time.
    """
    if isinstance(value, Mapping):
        return {k: _thaw_option_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_option_value(v) for v in value]
    return value


def _serialize_backend_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]] | None:
    """Convert a deeply frozen option mapping back to a plain nested dict."""
    if value is None:
        return None
    return {k: cast("dict[str, Any]", _thaw_option_value(v)) for k, v in value.items()}


def _hashable_option_value(value: object) -> object:
    """Recursively project an opaque option value into a hashable form.

    Lists / tuples become tuples; mappings become frozensets of
    ``(key, hashable_value)`` pairs. Atoms (str, int, float, bool, None,
    enums) pass through unchanged. Used by the custom ``__hash__`` on
    :class:`InputParameter` and :class:`CampaignSpec` so a value object
    that carries an option payload can still be hashed.
    """
    if isinstance(value, Mapping):
        return frozenset((k, _hashable_option_value(v)) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_option_value(v) for v in value)
    return value


def _hashable_backend_options(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[tuple[str, Any], ...] | None:
    """Project a ``backend → options`` mapping into a hashable tuple form."""
    if value is None:
        return None
    return tuple((k, _hashable_option_value(v)) for k, v in sorted(value.items()))


class Bounds(BaseModel):
    """Numeric lower/upper bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: float
    upper: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "Bounds":
        """Validate lower bound is less than upper bound."""
        if self.lower >= self.upper:
            msg = "Lower bound must be less than upper bound"
            raise ValueError(msg)
        return self


class InputParameter(BaseModel):
    """Input parameter definition.

    ``parameter_options`` carries per-backend metadata that has no neutral
    cross-backend equivalent (encoding choices, task-parameter active
    values, candidate-table mode). Outer keys are backend names; inner
    dicts are opaque to the neutral model. Backends ignore options
    addressed to other backends.

    Sequence fields (``values``, ``categories``) are typed as tuples so a
    frozen :class:`InputParameter` instance is also deeply immutable:
    ``param.categories.append(...)`` raises ``AttributeError`` instead of
    silently mutating shared state. JSON round-trips still produce
    arrays (Pydantic serializes tuples as JSON arrays).

    ``parameter_options`` is wrapped in nested :class:`types.MappingProxyType`
    views by ``field_validator(mode="after")`` so subscript assignment
    (``p.parameter_options["baybe"]["encoding"] = "x"``) raises
    ``TypeError`` instead of silently mutating the shared option dict.
    The custom :meth:`__hash__` projects the option mapping into a
    hashable form so instances with option payloads remain hashable for
    use as cache keys.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    type: ParameterType
    bounds: Bounds | None = None  # For continuous/discrete
    values: tuple[float, ...] | None = None  # For discrete (fractional values ok)
    categories: tuple[str, ...] | None = None  # For categorical
    description: str = ""
    parameter_options: Mapping[str, Mapping[str, Any]] | None = None

    @field_validator("bounds", mode="before")
    @classmethod
    def normalize_bounds(cls, value: object) -> object:
        """Accept legacy [lower, upper] payloads in addition to object form."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "Bounds must contain exactly 2 values"
                raise ValueError(msg)
            return {"lower": value[0], "upper": value[1]}
        return value

    @field_validator("parameter_options", mode="after")
    @classmethod
    def freeze_parameter_options(
        cls, value: Mapping[str, Mapping[str, Any]] | None
    ) -> Mapping[str, Mapping[str, Any]] | None:
        """Wrap the option mapping in read-only views (see module docstring)."""
        return _freeze_backend_options(value)

    @field_serializer("parameter_options", when_used="always")
    def _dump_parameter_options(
        self, value: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, dict[str, Any]] | None:
        """Materialize the frozen view back to a plain dict for JSON output."""
        return _serialize_backend_options(value)

    @model_validator(mode="after")
    def validate_parameter(self) -> "InputParameter":
        """Validate parameter has appropriate fields for its type."""
        if self.type == ParameterType.CONTINUOUS:
            if self.bounds is None:
                msg = "Continuous parameter requires bounds"
                raise ValueError(msg)
        elif self.type == ParameterType.DISCRETE and (self.values is None and self.bounds is None):
            msg = "Discrete parameter requires values or bounds"
            raise ValueError(msg)
        elif self.type == ParameterType.CATEGORICAL and (
            self.categories is None or len(self.categories) < 2
        ):
            msg = "Categorical parameter requires at least 2 categories"
            raise ValueError(msg)
        return self

    def __hash__(self) -> int:
        """Hash that handles the read-only ``parameter_options`` mapping.

        Pydantic's auto-generated ``__hash__`` would fail because the
        option mapping (even wrapped in ``MappingProxyType``) isn't
        natively hashable. Project the options into a tuple-of-items
        form so the instance remains usable as a dict / cache key.
        """
        return hash(
            (
                type(self).__name__,
                self.name,
                self.type,
                self.bounds,
                self.values,
                self.categories,
                self.description,
                _hashable_backend_options(self.parameter_options),
            )
        )


class Objective(BaseModel):
    """Optimization objective definition.

    ``log_transform`` opts a minimize objective into a ``Log → Standardize``
    outcome stack so multi-decade targets (e.g. concentrations or rates
    spanning several orders of magnitude) train against a roughly
    homoskedastic scale. Currently only valid for ``direction="minimize"``;
    enabling it on a maximize objective raises at the suggestion-generation
    boundary because BoTorch's ``Log`` transform requires strictly
    positive targets and negation flips positive raw values to negative.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1)
    direction: str = Field(..., pattern="^(minimize|maximize)$")
    unit: str = ""
    target: float | None = None  # Optional target value
    log_transform: bool = False

    @property
    def is_minimize(self) -> bool:
        """Check if objective should be minimized."""
        return self.direction == "minimize"


class Constraint(BaseModel):
    """Constraint definition.

    ``parameters`` and ``coefficients`` are tuples so a frozen instance
    is deeply immutable. JSON round-trips preserve these as arrays.

    Shape invariants per :attr:`type`:

    * ``LINEAR``: ``coefficients`` must be supplied and align one-to-one
      with ``parameters`` (same length, same order). The engine encodes
      the constraint as ``coefficients @ x[parameters] <= value``;
      missing coefficients used to be silently coerced into a sum
      constraint at the engine boundary, which produced unrelated
      semantics for a typo'd input. Reject the shape at intake so the
      failure is loud.
    * ``SUM_*``: ``coefficients`` must not be supplied (the constraint
      is unweighted by definition); supplying coefficients here is a
      sign the caller meant ``LINEAR`` and would otherwise be silently
      dropped on the SUM_* path.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: ConstraintType
    parameters: tuple[str, ...]  # Parameter names involved
    value: float  # Constraint value (e.g., sum equals this value)
    coefficients: tuple[float, ...] | None = None  # For linear constraints

    @model_validator(mode="after")
    def validate_constraint_shape(self) -> "Constraint":
        """Validate parameter references, coefficient cardinality and type-specific fields."""
        if not self.parameters:
            msg = f"Constraint of type {self.type.value} must reference at least one parameter"
            raise ValueError(msg)

        if self.type == ConstraintType.LINEAR:
            if self.coefficients is None:
                msg = (
                    f"Linear constraint requires coefficients (one per parameter); "
                    f"got coefficients=None for parameters={list(self.parameters)}."
                )
                raise ValueError(msg)
            if len(self.coefficients) != len(self.parameters):
                msg = (
                    f"Linear constraint requires one coefficient per parameter: "
                    f"got {len(self.coefficients)} coefficient(s) for "
                    f"{len(self.parameters)} parameter(s) "
                    f"({list(self.parameters)})."
                )
                raise ValueError(msg)
        elif self.coefficients is not None:
            msg = (
                f"Constraint of type {self.type.value} does not accept coefficients "
                f"(the constraint is unweighted by definition). Did you mean "
                f"type=linear?"
            )
            raise ValueError(msg)
        return self


class OutcomeConstraint(BaseModel):
    """Outcome constraint learned from data.

    Specifies a threshold on an objective that defines feasibility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    objective_name: str  # Which objective to constrain
    threshold: float  # Constraint value
    greater_than: bool = True  # obj >= threshold (True) or <= threshold
    feasibility_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # P(feasible) cutoff


class FidelityParameter(BaseModel):
    """Fidelity parameter for multi-fidelity optimization (v2.0).

    Fidelity parameters control the approximation level of evaluations.
    Lower fidelity = cheaper but less accurate.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    bounds: Bounds  # (min_fidelity, max_fidelity)
    target: float  # Target fidelity for final optimization (usually max)
    cost_weight: float = 1.0  # Cost scaling factor for fidelity
    fixed_cost: float = Field(default=0.0, ge=0.0)  # Fixed base cost

    @field_validator("bounds", mode="before")
    @classmethod
    def normalize_bounds(cls, value: object) -> object:
        """Accept legacy [lower, upper] payloads in addition to object form."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "Bounds must contain exactly 2 values"
                raise ValueError(msg)
            return {"lower": value[0], "upper": value[1]}
        return value


class TransferLearningConfig(BaseModel):
    """Configuration for transfer learning from prior campaigns (v2.0).

    Allows leveraging data from prior optimization campaigns. The
    ``prior_campaign_ids`` field is a tuple so a frozen config instance
    is deeply immutable.

    ``temperature`` is deprecated and has no effect: RGPE ensemble
    weights are computed from the paper's ranking loss (argmin counts
    over posterior samples), which involves no softmax. The field is
    kept only so previously stored specs and older clients keep
    validating; it is not forwarded to the engine.
    """

    model_config = ConfigDict(frozen=True)

    prior_campaign_ids: tuple[str, ...] = Field(..., min_length=1)
    num_ranking_samples: int = Field(default=512, ge=1)
    temperature: float = Field(
        default=0.5,
        gt=0.0,
        description=(
            "Deprecated, ignored: ranking-loss RGPE weights have no "
            "softmax temperature. Kept for stored-spec compatibility."
        ),
    )


class TurboConfig(BaseModel):
    """Configuration for TuRBO trust-region optimization.

    Present = use TuRBO, absent (None) = standard acquisition optimization.

    Defaults follow the canonical paper (Eriksson et al., NeurIPS 2019); see
    the bo-engine ``TurboState`` docstring for the unit-standardized-targets
    scale assumption and the meaning of each tolerance. ``failure_tolerance``
    defaults to ``None`` so the engine re-derives the dim/batch-size-aware
    default at construction time — set an integer to override.

    Invariants enforced at the schema boundary so garbage never reaches the
    engine: every length is strictly positive, ``length_min < length_max``,
    the initial trust region sits inside the operating band
    (``length_min <= initial_length <= length_max``), and the success /
    failure tolerances are at least one (the smallest value that still
    counts a single batch toward expand/contract).
    """

    model_config = ConfigDict(frozen=True)

    initial_length: float = Field(default=0.8, gt=0.0)
    length_min: float = Field(default=0.5**7, gt=0.0)
    length_max: float = Field(default=1.6, gt=0.0)
    success_tolerance: int = Field(default=10, ge=1)
    failure_tolerance: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_length_invariants(self) -> "TurboConfig":
        if self.length_min >= self.length_max:
            msg = (
                f"length_min ({self.length_min}) must be strictly less than "
                f"length_max ({self.length_max}); without a gap the trust "
                "region cannot expand or contract."
            )
            raise ValueError(msg)
        if not (self.length_min <= self.initial_length <= self.length_max):
            msg = (
                f"initial_length ({self.initial_length}) must lie in "
                f"[length_min, length_max] = [{self.length_min}, "
                f"{self.length_max}]; otherwise the trust region either "
                "triggers an immediate restart or starts above the cap."
            )
            raise ValueError(msg)
        return self


class SaasboConfig(BaseModel):
    """Configuration for SAASBO high-dimensional optimization.

    Present = use SAASBO, absent (None) = standard GP.
    """

    model_config = ConfigDict(frozen=True)

    warmup_steps: int = 256
    num_samples: int = 128
    thinning: int = 16


class AcquisitionOptimizationConfig(BaseModel):
    """Override L-BFGS-B restart count and raw-sample budget.

    Both fields are optional; ``None`` keeps the dimension-adaptive defaults
    from bo-engine. Use this only when calibrating against a benchmark or
    when the campaign has a known multi-modal acquisition surface that needs
    more aggressive exploration.
    """

    model_config = ConfigDict(frozen=True)

    num_restarts: int | None = Field(default=None, ge=1)
    raw_samples: int | None = Field(default=None, ge=1)


class CampaignSpec(BaseModel):
    """Immutable campaign specification.

    Created from validated intake and contains all resolved configuration.
    Collection fields (``parameters``, ``objectives``, ``constraints``,
    ``outcome_constraints``) are typed as tuples so the spec is deeply
    immutable: a caller cannot ``spec.parameters.append(...)`` and
    silently corrupt every consumer holding the same instance. Pydantic
    coerces lists from JSON / intake input into tuples automatically.

    The ``backend_options`` mapping is wrapped in nested read-only views
    (see :func:`_freeze_backend_options`) so subscript assignment fails
    the same way the sequence fields do, and the custom :meth:`__hash__`
    projects the option mapping into a hashable form so a spec carrying
    an options payload remains usable as a cache / dedup key.
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: tuple[InputParameter, ...] = Field(..., min_length=1)
    objectives: tuple[Objective, ...] = Field(..., min_length=1)
    constraints: tuple[Constraint, ...] = Field(default_factory=tuple)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    # Total observation cap. Counted across all iterations; reaching it short-
    # circuits ``generate_suggestions`` even mid-iteration.
    max_observations: int | None = Field(default=None, ge=1)
    # Relative-improvement threshold passed to ``detect_convergence``. When
    # set, the suggestion entry point reports ``CONVERGED`` once recent
    # improvement falls below this value.
    convergence_tolerance: float | None = Field(default=None, gt=0.0)
    initial_design_size: int | None = None
    random_seed: int | None = Field(
        default=None,
        description=(
            "Campaign-level RNG seed. Optional. When supplied, the Sobol "
            "initial design and acquisition multi-start are deterministic "
            "within a fixed (torch version, device, deterministic-algorithms "
            "setting) triple; suggestions are NOT byte-identical across "
            "different torch versions, CPU vs. CUDA, or backend swaps. Set "
            "torch.use_deterministic_algorithms(True) for strictest behavior."
        ),
    )
    # v1.0.1: Acquisition method selection
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    # v1.1: Input warping for non-stationary objectives
    use_input_warping: bool = False
    # v1.2: TuRBO for high-dimensional optimization (None = disabled)
    turbo_config: TurboConfig | None = None
    # v1.3: Outcome constraints learned from data
    outcome_constraints: tuple[OutcomeConstraint, ...] = Field(default_factory=tuple)
    # v1.3: Cost-aware optimization
    use_cost_aware: bool = False
    # v2.0: Multi-fidelity optimization
    fidelity_parameter: FidelityParameter | None = None
    # v2.0: Transfer learning from prior campaigns
    transfer_learning: TransferLearningConfig | None = None
    # v2.0: SAASBO for high-dimensional optimization (None = disabled)
    saasbo_config: SaasboConfig | None = None
    # Acquisition-optimizer restart / raw-sample budget. Defaults to None so
    # the bo-engine dimension-adaptive defaults apply.
    acquisition_optimization: AcquisitionOptimizationConfig | None = None
    # Original caller-side backend selector before "auto" is resolved. Stored
    # for provenance only; execution uses ``backend`` below.
    requested_backend: str | None = None
    # v3.0: Backend selection (default uses BO_BACKEND env var)
    backend: str = "botorch"
    # Typed backend-native option surface. Outer keys are backend names
    # (``"botorch"``, ``"baybe"``); inner dicts hold options that have no
    # neutral cross-backend equivalent. Backends ignore options addressed
    # to other backends.
    backend_options: Mapping[str, Mapping[str, Any]] | None = None
    # Caller-side acknowledgement that the chosen backend may silently
    # drop these option fields. Backends classify semantically
    # load-bearing options (e.g. ``outcome_constraints`` on BayBE) as
    # UNSUPPORTED by default so a misrouted spec fails at intake instead
    # of running the optimization without the requested semantics. Naming
    # the field here downgrades the corresponding report to IGNORED,
    # opting into a degraded run with a prominent warning.
    acknowledge_degradations: tuple[str, ...] = Field(default_factory=tuple)

    model_config = ConfigDict(frozen=True)

    @field_validator("acquisition_method", mode="before")
    @classmethod
    def normalize_acquisition_method(cls, value: object) -> object:
        """Accept legacy BoTorch class-name values for backward compat."""
        if isinstance(value, str) and value in _LEGACY_ACQUISITION_VALUES:
            return _LEGACY_ACQUISITION_VALUES[value]
        return value

    @field_validator("backend_options", mode="after")
    @classmethod
    def freeze_backend_options(
        cls, value: Mapping[str, Mapping[str, Any]] | None
    ) -> Mapping[str, Mapping[str, Any]] | None:
        """Wrap ``backend_options`` in nested read-only views (see module docstring)."""
        return _freeze_backend_options(value)

    @field_serializer("backend_options", when_used="always")
    def _dump_backend_options(
        self, value: Mapping[str, Mapping[str, Any]] | None
    ) -> dict[str, dict[str, Any]] | None:
        """Materialize the frozen view back to a plain dict for JSON output."""
        return _serialize_backend_options(value)

    @property
    def use_turbo(self) -> bool:
        """Backward-compatible check for TuRBO enabled."""
        return self.turbo_config is not None

    @property
    def use_saasbo(self) -> bool:
        """Backward-compatible check for SAASBO enabled."""
        return self.saasbo_config is not None

    @model_validator(mode="after")
    def validate_spec(self) -> "CampaignSpec":
        """Validate campaign specification."""
        param_names = {p.name for p in self.parameters}

        # Validate constraint references
        for constraint in self.constraints:
            for param_name in constraint.parameters:
                if param_name not in param_names:
                    msg = f"Constraint references unknown parameter: {param_name}"
                    raise ValueError(msg)

        # Outcome constraints reference an objective by name. A typo here
        # used to slip through intake validation and was caught silently at
        # the engine boundary (the constraint was disabled), so users got
        # an "unconstrained" campaign they believed was constrained. Reject
        # unknown ``objective_name`` values at intake so the failure mode is
        # loud rather than a silent correctness bug.
        objective_names = {o.name for o in self.objectives}
        for outcome_constraint in self.outcome_constraints:
            if outcome_constraint.objective_name not in objective_names:
                msg = (
                    f"Outcome constraint references unknown objective "
                    f"'{outcome_constraint.objective_name}'. "
                    f"Declared objectives: {sorted(objective_names)}."
                )
                raise ValueError(msg)

        # ``convergence_tolerance`` is wired to a single-objective running-best
        # trajectory. Hypervolume-based convergence for multi-objective specs
        # is tracked separately; reject the setting at create time instead of
        # silently using only the first objective at suggestion time.
        if self.convergence_tolerance is not None and len(self.objectives) > 1:
            msg = (
                "convergence_tolerance is only supported for single-objective "
                "campaigns; multi-objective campaigns must rely on hypervolume "
                "diagnostics instead."
            )
            raise ValueError(msg)

        return self

    @property
    def n_parameters(self) -> int:
        """Number of input parameters."""
        return len(self.parameters)

    @property
    def n_objectives(self) -> int:
        """Number of objectives."""
        return len(self.objectives)

    @property
    def is_multi_objective(self) -> bool:
        """Check if this is a multi-objective problem."""
        return self.n_objectives > 1

    def get_parameter(self, name: str) -> InputParameter | None:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump()

    def __hash__(self) -> int:
        """Hash that handles the read-only ``backend_options`` mapping.

        Pydantic's auto-generated ``__hash__`` would fail for the
        option mapping (even wrapped in ``MappingProxyType``) because
        ``Mapping`` is not natively hashable. We project the options
        into a tuple-of-items form so a spec with an options payload
        can still be hashed (e.g. for cache keys keyed on spec identity).
        """
        return hash(
            (
                type(self).__name__,
                self.name,
                self.description,
                self.parameters,
                self.objectives,
                self.constraints,
                self.batch_size,
                self.max_iterations,
                self.max_observations,
                self.convergence_tolerance,
                self.initial_design_size,
                self.random_seed,
                self.acquisition_method,
                self.use_input_warping,
                self.turbo_config,
                self.outcome_constraints,
                self.use_cost_aware,
                self.fidelity_parameter,
                self.transfer_learning,
                self.saasbo_config,
                self.acquisition_optimization,
                self.requested_backend,
                self.backend,
                _hashable_backend_options(self.backend_options),
                self.acknowledge_degradations,
            )
        )
