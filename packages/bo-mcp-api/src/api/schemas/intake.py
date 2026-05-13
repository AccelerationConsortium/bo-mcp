"""Intake data schemas.

The REST intake mirrors the MCP ``CampaignIntakeInput`` field set so the
same payload can drive either transport. Unknown extras are rejected
with ``model_config={"extra": "forbid"}`` so misspelled or
not-yet-supported keys fail loudly instead of disappearing silently —
that was the pre-1.66 behavior and it produced confused agents.
"""

from typing import Any

from pydantic import BaseModel, Field


class ParameterInput(BaseModel):
    """Parameter definition input.

    ``parameter_options`` is a per-backend metadata dict keyed by backend
    name (BayBE encoding choice, task-parameter active values, etc.). It
    flows through to ``CampaignSpec.parameters[*].parameter_options``.
    """

    name: str = Field(..., min_length=1)
    type: str = Field(..., pattern="^(continuous|discrete|categorical)$")
    bounds: list[float] | None = None
    values: list[int] | None = None
    categories: list[str] | None = None
    description: str = ""
    parameter_options: dict[str, dict[str, Any]] | None = None

    model_config = {"extra": "forbid"}


class ObjectiveInput(BaseModel):
    """Objective definition input."""

    name: str = Field(..., min_length=1)
    direction: str = Field(..., pattern="^(minimize|maximize)$")
    unit: str = ""
    target: float | None = None

    model_config = {"extra": "forbid"}


class ConstraintInput(BaseModel):
    """Constraint definition input."""

    type: str = Field(..., pattern="^(sum_equals|sum_less_than|sum_greater_than|linear)$")
    parameters: list[str]
    value: float
    coefficients: list[float] | None = None

    model_config = {"extra": "forbid"}


class IntakeData(BaseModel):
    """Campaign intake data schema for the REST API.

    Field set mirrors ``bo_mcp_server.domain.CampaignIntakeInput`` so the
    same JSON payload works on either transport. Field-level validation
    (constraint references, backend_options key matching, ...) happens
    when the dict is passed to ``CampaignIntakeInput.model_validate(...)``
    in the route handler.
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: list[ParameterInput]
    objectives: list[ObjectiveInput]
    constraints: list[ConstraintInput] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    max_observations: int | None = Field(default=None, ge=1)
    convergence_tolerance: float | None = Field(default=None, gt=0.0)
    initial_design_size: int | None = None
    random_seed: int | None = None
    acquisition_optimization: dict[str, Any] | None = None
    backend: str = Field(default="auto", pattern="^(auto|botorch|baybe)$")
    backend_options: dict[str, dict[str, Any]] | None = None
    # Advanced spec knobs — typed as ``dict`` at the REST boundary so the
    # REST schema does not couple to backend-specific Pydantic models.
    # ``CampaignIntakeInput`` / ``CampaignSpec`` re-validate the inner
    # shape, raising 422 on malformed payloads.
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

    model_config = {"extra": "forbid"}
