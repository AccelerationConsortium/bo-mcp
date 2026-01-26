"""Campaign specification value object."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ParameterType(str, Enum):
    """Type of input parameter."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


class InputParameter(BaseModel):
    """Input parameter definition."""

    name: str = Field(..., min_length=1)
    type: ParameterType
    bounds: tuple[float, float] | None = None  # For continuous/discrete
    values: list[int] | None = None  # For discrete
    categories: list[str] | None = None  # For categorical
    description: str = ""

    @model_validator(mode="after")
    def validate_parameter(self) -> "InputParameter":
        """Validate parameter has appropriate fields for its type."""
        if self.type == ParameterType.CONTINUOUS:
            if self.bounds is None:
                msg = "Continuous parameter requires bounds"
                raise ValueError(msg)
            if self.bounds[0] >= self.bounds[1]:
                msg = "Lower bound must be less than upper bound"
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


class ConstraintType(str, Enum):
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


class AcquisitionMethod(str, Enum):
    """Acquisition function method."""

    AUTO = "auto"  # Automatic selection based on n_objectives
    QLOGNEI = "qLogNEI"  # Single-objective: Log Noisy Expected Improvement
    QLOGEI = "qLogEI"  # Single-objective: Log Expected Improvement (noiseless)
    QLOGNEHVI = "qLogNEHVI"  # Multi-objective: Log Noisy Expected Hypervolume Improvement
    QLOGPAREGO = "qLogNParEGO"  # Multi-objective: Parallel EGO with Chebyshev scalarization
    EIPU = "EIpu"  # Cost-aware: Expected Improvement per Unit cost
    # v2.0: Advanced acquisition methods
    QMFKG = "qMFKG"  # Multi-fidelity: Knowledge Gradient
    SAASBO = "SAASBO"  # High-dimensional: Sparse Axis-Aligned Subspace BO


class OutcomeConstraint(BaseModel):
    """Outcome constraint learned from data.

    Specifies a threshold on an objective that defines feasibility.
    """

    objective_name: str  # Which objective to constrain
    threshold: float  # Constraint value
    greater_than: bool = True  # True: obj >= threshold, False: obj <= threshold


class FidelityParameter(BaseModel):
    """Fidelity parameter for multi-fidelity optimization (v2.0).

    Fidelity parameters control the approximation level of evaluations.
    Lower fidelity = cheaper but less accurate.
    """

    name: str = Field(..., min_length=1)
    bounds: tuple[float, float]  # (min_fidelity, max_fidelity)
    target: float  # Target fidelity for final optimization (usually max)
    cost_weight: float = 1.0  # Cost scaling factor for fidelity
    fixed_cost: float = 5.0  # Fixed base cost


class TransferLearningConfig(BaseModel):
    """Configuration for transfer learning from prior campaigns (v2.0).

    Allows leveraging data from prior optimization campaigns.
    """

    prior_campaign_ids: list[str] = Field(..., min_length=1)
    num_ranking_samples: int = Field(default=512, ge=1)


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
    # v1.2: TuRBO for high-dimensional optimization
    use_turbo: bool = False
    # v1.3: Outcome constraints learned from data
    outcome_constraints: list[OutcomeConstraint] = Field(default_factory=list)
    # v1.3: Cost-aware optimization (EIpu)
    use_cost_aware: bool = False
    # v2.0: Multi-fidelity optimization
    fidelity_parameter: FidelityParameter | None = None
    # v2.0: Transfer learning from prior campaigns
    transfer_learning: TransferLearningConfig | None = None
    # v2.0: SAASBO for high-dimensional optimization (50+ params)
    use_saasbo: bool = False

    model_config = {"frozen": True}

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
