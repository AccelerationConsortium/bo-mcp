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

from bo_mcp_server.client import (
    Constraint,
    InputParameter,
    Objective,
)
from pydantic import BaseModel, ConfigDict, Field

from api.limits import (
    MAX_INTAKE_CONSTRAINTS,
    MAX_INTAKE_OBJECTIVES,
    MAX_INTAKE_PARAMETERS,
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
    ``outcome_constraints``, ``acquisition_optimization``) stay typed as
    plain ``dict`` here so the REST schema does not couple to the
    backend-specific Pydantic configs; ``CampaignIntakeInput`` /
    ``CampaignSpec`` re-validate the inner shape and raise 422 on
    malformed payloads.
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: tuple[InputParameter, ...] = Field(
        ..., min_length=1, max_length=MAX_INTAKE_PARAMETERS
    )
    objectives: tuple[Objective, ...] = Field(..., min_length=1, max_length=MAX_INTAKE_OBJECTIVES)
    constraints: tuple[Constraint, ...] = Field(
        default_factory=tuple, max_length=MAX_INTAKE_CONSTRAINTS
    )
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    max_observations: int | None = Field(default=None, ge=1)
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
    acquisition_optimization: dict[str, Any] | None = None
    # ``Literal`` mirrors :class:`bo_mcp_server.domain.CampaignIntakeInput`
    # so the REST OpenAPI schema advertises an explicit ``enum`` constraint
    # and the route handler can pass the value straight through without a
    # ``ty: ignore`` widening cast.
    backend: Literal["auto", "botorch", "baybe"] = "auto"
    backend_options: dict[str, dict[str, Any]] | None = None
    # Default ``"auto"`` matches the MCP ``CampaignIntakeInput`` default
    # so an omitted field produces identical behavior on both transports.
    acquisition_method: str = "auto"
    use_input_warping: bool = False
    use_cost_aware: bool = False
    turbo_config: dict[str, Any] | None = None
    saasbo_config: dict[str, Any] | None = None
    fidelity_parameter: dict[str, Any] | None = None
    transfer_learning: dict[str, Any] | None = None
    outcome_constraints: list[dict[str, Any]] = Field(default_factory=list)
    # Caller opt-in to "this backend may silently drop these option
    # fields". Mirrors :class:`CampaignIntakeInput.acknowledge_degradations`
    # so the REST and MCP transports accept the same shape.
    acknowledge_degradations: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")
