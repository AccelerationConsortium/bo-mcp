"""Pydantic input models for MCP tool payloads."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from bo_mcp_server.domain.campaign_spec import Constraint, InputParameter, Objective


class CampaignIntakeInput(BaseModel):
    """Validated input payload for campaign intake-based tools."""

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: list[InputParameter] = Field(..., min_length=1)
    objectives: list[Objective] = Field(..., min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    initial_design_size: int | None = None
    random_seed: int | None = 42

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_constraint_parameter_refs(self) -> "CampaignIntakeInput":
        """Ensure constraints only reference declared parameters."""
        parameter_names = {p.name for p in self.parameters}
        invalid_params = []

        for constraint in self.constraints:
            for parameter in constraint.parameters:
                if parameter not in parameter_names:
                    invalid_params.append(parameter)

        if invalid_params:
            unique_invalid = list(dict.fromkeys(invalid_params))
            msg = f"Constraints reference unknown parameters: {', '.join(unique_invalid)}"
            raise ValueError(msg)

        return self


class ResultSubmissionInput(BaseModel):
    """Validated input payload for each submitted result."""

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    suggestion_id: str | None = None
    measurement_uncertainty: dict[str, float] | None = None  # Per-objective noise std
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}
