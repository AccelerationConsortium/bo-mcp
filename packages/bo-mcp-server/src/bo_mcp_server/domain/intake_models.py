"""Pydantic input models for MCP tool payloads."""

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
    random_seed: int | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_constraint_parameter_refs(self) -> "CampaignIntakeInput":
        """Ensure constraints only reference declared parameters."""
        parameter_names = {p.name for p in self.parameters}
        for constraint in self.constraints:
            for parameter in constraint.parameters:
                if parameter not in parameter_names:
                    msg = f"Constraint references unknown parameter '{parameter}'"
                    raise ValueError(msg)
        return self
