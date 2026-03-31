"""Campaign specification value object."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class ParameterType(StrEnum):
    """Type of input parameter."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class Bounds(BaseModel):
    """Numeric lower/upper bounds."""

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
    """Input parameter definition."""

    name: str = Field(..., min_length=1)
    type: ParameterType
    bounds: Bounds | None = None  # For continuous/discrete
    values: list[int] | None = None  # For discrete
    categories: list[str] | None = None  # For categorical
    description: str = ""

    @field_validator("bounds", mode="before")
    @classmethod
    def normalize_bounds(cls, value: Any) -> Any:
        """Accept legacy [lower, upper] payloads in addition to object form."""
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "Bounds must contain exactly 2 values"
                raise ValueError(msg)
            return {"lower": value[0], "upper": value[1]}
        return value

    @model_validator(mode="after")
    def validate_parameter(self) -> "InputParameter":
        """Validate parameter has appropriate fields for its type."""
        if self.type == ParameterType.CONTINUOUS:
            if self.bounds is None:
                msg = "Continuous parameter requires bounds"
                raise ValueError(msg)
        elif self.type == ParameterType.DISCRETE:
            if self.values is None and self.bounds is None:
                msg = "Discrete parameter requires values or bounds"
                raise ValueError(msg)
        elif self.type == ParameterType.CATEGORICAL:
            if self.categories is None or len(self.categories) < 2:
                msg = "Categorical parameter requires at least 2 categories"
                raise ValueError(msg)
        return self


class Objective(BaseModel):
    """Optimization objective definition."""

    name: str = Field(..., min_length=1)
    direction: str = Field(..., pattern="^(minimize|maximize)$")
    unit: str = ""
    target: float | None = None  # Optional target value

    @property
    def is_minimize(self) -> bool:
        """Check if objective should be minimized."""
        return self.direction == "minimize"


class ConstraintType(StrEnum):
    """Type of constraint."""

    SUM_EQUALS = "sum_equals"
    SUM_LESS_THAN = "sum_less_than"
    SUM_GREATER_THAN = "sum_greater_than"
    LINEAR = "linear"


class Constraint(BaseModel):
    """Constraint definition."""

    type: ConstraintType
    parameters: list[str]  # Parameter names involved
    value: float  # Constraint value (e.g., sum equals this value)
    coefficients: list[float] | None = None  # For linear constraints


class AcquisitionMethod(StrEnum):
    """Acquisition function method.

    Values are backend-agnostic semantic names.
    """

    AUTO = "auto"
    NOISY_EI = "noisy_expected_improvement"
    EXPECTED_IMPROVEMENT = "expected_improvement"
    HYPERVOLUME_IMPROVEMENT = "hypervolume_improvement"
    SCALARIZED_MULTI_OBJ = "scalarized_multi_objective"
    COST_WEIGHTED_EI = "cost_weighted_ei"
    MULTI_FIDELITY_KG = "multi_fidelity_kg"


# Maps legacy BoTorch class-name values to current semantic names.
_LEGACY_ACQUISITION_VALUES: dict[str, str] = {
    "qLogNEI": "noisy_expected_improvement",
    "qLogEI": "expected_improvement",
    "qLogNEHVI": "hypervolume_improvement",
    "qLogNParEGO": "scalarized_multi_objective",
    "EIpu": "cost_weighted_ei",
    "qMFKG": "multi_fidelity_kg",
    "SAASBO": "noisy_expected_improvement",
}


class OutcomeConstraint(BaseModel):
    """Outcome constraint learned from data.

    Specifies a threshold on an objective that defines feasibility.
    """

    objective_name: str  # Which objective to constrain
    threshold: float  # Constraint value
    greater_than: bool = True  # obj >= threshold (True) or <= threshold
    feasibility_threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # P(feasible) cutoff


class FidelityParameter(BaseModel):
    """Fidelity parameter for multi-fidelity optimization (v2.0).

    Fidelity parameters control the approximation level of evaluations.
    Lower fidelity = cheaper but less accurate.
    """

    name: str = Field(..., min_length=1)
    bounds: Bounds  # (min_fidelity, max_fidelity)
    target: float  # Target fidelity for final optimization (usually max)
    cost_weight: float = 1.0  # Cost scaling factor for fidelity
    fixed_cost: float = Field(default=0.0, ge=0.0)  # Fixed base cost

    @field_validator("bounds", mode="before")
    @classmethod
    def normalize_bounds(cls, value: Any) -> Any:
        """Accept legacy [lower, upper] payloads in addition to object form."""
        if isinstance(value, (list, tuple)):
            if len(value) != 2:
                msg = "Bounds must contain exactly 2 values"
                raise ValueError(msg)
            return {"lower": value[0], "upper": value[1]}
        return value


class TransferLearningConfig(BaseModel):
    """Configuration for transfer learning from prior campaigns (v2.0).

    Allows leveraging data from prior optimization campaigns.
    """

    prior_campaign_ids: list[str] = Field(..., min_length=1)
    num_ranking_samples: int = Field(default=512, ge=1)
    temperature: float = Field(default=0.5, gt=0.0)


class TurboConfig(BaseModel):
    """Configuration for TuRBO trust-region optimization.

    Present = use TuRBO, absent (None) = standard acquisition optimization.
    """

    initial_length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    success_tolerance: int = 10


class SaasboConfig(BaseModel):
    """Configuration for SAASBO high-dimensional optimization.

    Present = use SAASBO, absent (None) = standard GP.
    """

    warmup_steps: int = 256
    num_samples: int = 128
    thinning: int = 16


class CampaignSpec(BaseModel):
    """Immutable campaign specification.

    Created from validated intake and contains all resolved configuration.
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: list[InputParameter] = Field(..., min_length=1)
    objectives: list[Objective] = Field(..., min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    initial_design_size: int | None = None
    random_seed: int | None = None
    # v1.0.1: Acquisition method selection
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    # v1.1: Input warping for non-stationary objectives
    use_input_warping: bool = False
    # v1.2: TuRBO for high-dimensional optimization (None = disabled)
    turbo_config: TurboConfig | None = None
    # v1.3: Outcome constraints learned from data
    outcome_constraints: list[OutcomeConstraint] = Field(default_factory=list)
    # v1.3: Cost-aware optimization
    use_cost_aware: bool = False
    # v2.0: Multi-fidelity optimization
    fidelity_parameter: FidelityParameter | None = None
    # v2.0: Transfer learning from prior campaigns
    transfer_learning: TransferLearningConfig | None = None
    # v2.0: SAASBO for high-dimensional optimization (None = disabled)
    saasbo_config: SaasboConfig | None = None

    model_config = {"frozen": True}

    @field_validator("acquisition_method", mode="before")
    @classmethod
    def normalize_acquisition_method(cls, value: Any) -> Any:
        """Accept legacy BoTorch class-name values for backward compat."""
        if isinstance(value, str) and value in _LEGACY_ACQUISITION_VALUES:
            return _LEGACY_ACQUISITION_VALUES[value]
        return value

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
