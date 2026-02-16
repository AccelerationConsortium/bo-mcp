"""Internal types for bo-engine.

These types define the interface between bo-engine and higher-level packages.
They are simple dataclasses/TypedDicts to keep bo-engine independent of
external Pydantic models or other frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from botorch.models import SingleTaskGP
    from botorch.models.model_list_gp_regression import ModelListGP
    from torch import Tensor

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
    """Specification for a single input parameter."""

    name: str
    type: ParameterType
    bounds: tuple[float, float] | None = None  # For continuous/discrete
    values: list[int] | None = None  # For discrete (explicit values)
    categories: list[str] | None = None  # For categorical


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


@dataclass(frozen=True)
class OutcomeConstraintSpec:
    """Specification for an outcome constraint learned from data.

    Outcome constraints define feasibility thresholds on objectives.
    A constraint GP is trained to predict P(feasible | x).
    """

    objective_name: str  # Which objective to constrain
    threshold: float  # Constraint value
    greater_than: bool = True  # True: obj >= threshold, False: obj <= threshold


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
    fixed_cost: float = 5.0  # Fixed base cost


@dataclass(frozen=True)
class TransferLearningSpec:
    """Specification for transfer learning from prior campaigns (v2.0 RGPE).

    Allows leveraging data from prior optimization campaigns to
    accelerate optimization on a new but related task.
    """

    prior_campaign_ids: list[str]  # IDs of prior campaigns to transfer from
    num_ranking_samples: int = 512  # Samples for rank computation


@dataclass(frozen=True)
class OptimizationSpec:
    """Full specification for an optimization problem.

    This is the main input type for bo-engine functions.
    """

    parameters: list[ParameterSpec]
    objectives: list[ObjectiveSpec]
    constraints: list[ConstraintSpec] = field(default_factory=list)
    batch_size: int = 1
    initial_design_size: int | None = None
    # v1.0.1: Acquisition method selection
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    # v1.1: Input warping for non-stationary objectives
    use_input_warping: bool = False
    # v1.2: TuRBO for high-dimensional optimization
    use_turbo: bool = False
    # v1.3: Outcome constraints learned from data
    outcome_constraints: list[OutcomeConstraintSpec] = field(default_factory=list)
    # v1.3: Cost-aware optimization (EIpu)
    use_cost_aware: bool = False
    # v2.0: Multi-fidelity optimization
    fidelity_parameter: FidelityParameterSpec | None = None
    # v2.0: Transfer learning from prior campaigns
    transfer_learning: TransferLearningSpec | None = None
    # v2.0: SAASBO for high-dimensional optimization (50+ params)
    use_saasbo: bool = False

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


@dataclass
class ObservationData:
    """Observed data point for optimization.

    Contains parameter values and their corresponding objective values.
    """

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    cost: float | None = None  # v1.3: Cost for cost-aware optimization


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


@dataclass
class AcquisitionConfig:
    """Configuration for acquisition function creation.

    Bundles parameters for create_acquisition to reduce function parameter count.
    """

    # Required parameters
    model: GPModel | Any  # ModelListGP | SingleTaskGP (Any at runtime)
    train_x: Tensor | Any  # Tensor (Any at runtime)
    train_y: Tensor | Any  # Tensor (Any at runtime)
    n_objectives: int = 2

    # Optional parameters
    ref_point: Tensor | Any | None = None  # Tensor
    method: AcquisitionMethod = AcquisitionMethod.AUTO
    constraints: list[Any] | None = None
    outcome_constraint_models: list[OutcomeConstraintModel] | list[tuple[Any, float]] | None = None
    cost_model: SingleTaskGP | Any | None = None  # SingleTaskGP
