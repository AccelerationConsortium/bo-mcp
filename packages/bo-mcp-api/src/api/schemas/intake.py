"""Intake data schemas.

The REST intake mirrors the MCP ``CampaignIntakeInput`` field set so the
same payload can drive either transport. Unknown extras are rejected
with ``model_config={"extra": "forbid"}`` so misspelled or
not-yet-supported keys fail loudly instead of disappearing silently —
that was the pre-1.66 behavior and it produced confused agents.

Parameter, objective, and constraint nested models are the canonical
domain types from :mod:`bo_mcp_server.domain`. This lets the REST handler
hand the validated nested instances straight to ``CampaignIntakeInput``
without a ``model_dump() -> model_validate()`` round-trip.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from api.limits import (
    MAX_INTAKE_CONSTRAINTS,
    MAX_INTAKE_OBJECTIVES,
    MAX_INTAKE_PARAMETERS,
)
from bo_engine.types import ScalarizationMode, ScalarizerKind
from bo_mcp_server.client import (
    AcquisitionMethod,
    AcquisitionOptimizationConfig,
    Constraint,
    FidelityParameter,
    InputParameter,
    Objective,
    OutcomeConstraint,
    SaasboConfig,
    TransferLearningConfig,
    TurboConfig,
)


class IntakeData(BaseModel):
    """Campaign intake data schema for the REST API.

    Field set mirrors ``bo_mcp_server.domain.CampaignIntakeInput`` so the
    same JSON payload works on either transport. The ``parameters``,
    ``objectives``, and ``constraints`` fields use the canonical domain
    types directly — when the REST handler forwards a validated
    ``IntakeData`` to ``CampaignIntakeInput`` it can pass the already-
    parsed nested instances through without re-dumping to a dict.

    The advanced cross-backend knobs (``turbo_config``, ``saasbo_config``,
    ``fidelity_parameter``, ``transfer_learning``,
    ``outcome_constraints``, ``acquisition_optimization``) use the same
    canonical domain config models as ``CampaignIntakeInput`` (they are
    neutral domain types, not backend-specific). This gives the REST
    OpenAPI the full typed shape of each knob — parity with the MCP tool
    schema — and rejects a malformed inner field at the request boundary
    with a 422 instead of an opaque ``object``. ``CampaignIntakeInput`` /
    ``CampaignSpec`` still re-validate downstream.
    """

    name: str = Field(..., min_length=1)
    description: str = Field(default="", description="Free-text human-readable note.")
    parameters: tuple[InputParameter, ...] = Field(
        ..., min_length=1, max_length=MAX_INTAKE_PARAMETERS
    )
    objectives: tuple[Objective, ...] = Field(..., min_length=1, max_length=MAX_INTAKE_OBJECTIVES)
    constraints: tuple[Constraint, ...] = Field(
        default_factory=tuple, max_length=MAX_INTAKE_CONSTRAINTS
    )
    batch_size: int = Field(
        default=1, ge=1, description="Number of suggestions generated per call."
    )
    max_iterations: int | None = Field(
        default=None,
        description=(
            "Cap on the number of completed BO iterations. Once reached, "
            "suggestion generation reports BUDGET_EXCEEDED instead of "
            "producing more suggestions."
        ),
    )
    max_observations: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cap on the total number of observed results, irrespective of "
            "iteration grouping. Reaching it short-circuits suggestion "
            "generation even mid-iteration."
        ),
    )
    convergence_tolerance: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Relative-improvement threshold below which the campaign is "
            "considered converged (single-objective campaigns only)."
        ),
    )
    initial_design_size: int | None = Field(
        default=None,
        description=(
            "Number of space-filling (Sobol/random) warmup points before "
            "switching to the model-driven acquisition phase. None uses a "
            "dimension-adaptive default (BoTorch) or switches after the "
            "first measurement (BayBE, unless overridden by "
            "backend_options['baybe'].recommender.switch_after, which "
            "takes precedence)."
        ),
    )
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
    acquisition_optimization: AcquisitionOptimizationConfig | None = None
    # ``Literal`` mirrors :class:`bo_mcp_server.domain.CampaignIntakeInput`
    # so the REST OpenAPI schema advertises an explicit ``enum`` constraint
    # and the route handler can pass the value straight through without a
    # ``ty: ignore`` widening cast.
    backend: Literal["auto", "botorch", "baybe"] = Field(
        default="auto",
        description=(
            "Optimization backend. 'auto' prefers the deployment's "
            "configured default backend (BO_BACKEND env var, typically "
            "'baybe'), but switches to whichever installed backend can "
            "run the spec without silently dropping an option — e.g. a "
            "spec using a BoTorch-only feature (TuRBO, SAASBO, "
            "multi-fidelity, RGPE transfer learning, cost-aware, input "
            "warping, outcome constraints) auto-selects 'botorch'. Pin "
            "explicitly to fail fast instead of silently switching."
        ),
    )
    backend_options: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Backend-native option surface, keyed by backend name "
            "(currently only 'baybe' has a typed schema — see "
            "BayBEBackendOptions/BayBEParameterOptions below). Options "
            "addressed to a non-selected backend are rejected at intake "
            "when `backend` is pinned to a concrete name."
        ),
    )
    # Typed as the ``AcquisitionMethod`` enum (mirroring
    # ``CampaignIntakeInput``) so the REST OpenAPI advertises the valid
    # values and an invalid method is rejected at the request boundary
    # rather than later in ``_coerce_intake``. ``AcquisitionMethod.AUTO``
    # serializes to ``"auto"``, so an omitted field behaves identically on
    # both transports.
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    # UCB-family exploration weight; only valid with
    # acquisition_method='upper_confidence_bound' (enforced by CampaignSpec).
    acquisition_beta: float | None = Field(
        default=None,
        description=(
            "UCB exploration weight. Only valid with "
            "acquisition_method='upper_confidence_bound'; rejected otherwise."
        ),
    )
    # Multi-objective combination strategy + desirability scalarizer flavor;
    # cross-field rules enforced by CampaignSpec.
    scalarization: ScalarizationMode = ScalarizationMode.PARETO
    scalarizer: ScalarizerKind | None = None
    use_input_warping: bool = Field(
        default=False,
        description=(
            "Input warping for non-stationary objectives. BoTorch-only — "
            "reported UNSUPPORTED on the BayBE backend by default (see "
            "`acknowledge_degradations`)."
        ),
    )
    use_cost_aware: bool = Field(
        default=False,
        description=(
            "Cost-aware acquisition (EIpu), weighting candidates by "
            "`fidelity_parameter` cost. BoTorch-only — reported "
            "UNSUPPORTED on the BayBE backend by default (see "
            "`acknowledge_degradations`)."
        ),
    )
    turbo_config: TurboConfig | None = None
    saasbo_config: SaasboConfig | None = None
    fidelity_parameter: FidelityParameter | None = None
    transfer_learning: TransferLearningConfig | None = None
    outcome_constraints: tuple[OutcomeConstraint, ...] = Field(default_factory=tuple)
    # Caller opt-in to "this backend may silently drop these option
    # fields". Mirrors :class:`CampaignIntakeInput.acknowledge_degradations`
    # (same ``tuple[str, ...]`` annotation) so the REST and MCP transports
    # accept and validate the identical shape.
    acknowledge_degradations: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "Opt-in list of attribute names (e.g. 'turbo_config', "
            "'outcome_constraints') whose BayBE-UNSUPPORTED status should "
            "downgrade to an IGNORED warning instead of rejecting the "
            "request, when running a BoTorch-only feature on "
            "backend='baybe'."
        ),
    )

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "name": "example-baybe-campaign",
                    "parameters": [
                        {"name": "x", "type": "continuous", "bounds": {"lower": 0.0, "upper": 1.0}}
                    ],
                    "objectives": [{"name": "y", "direction": "minimize"}],
                    "backend": "baybe",
                }
            ]
        },
    )
