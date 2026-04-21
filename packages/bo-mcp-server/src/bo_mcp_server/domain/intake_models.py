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
    backend: str = Field(default="auto", pattern="^(auto|botorch|baybe)$")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_names_and_constraints(self) -> "CampaignIntakeInput":
        """Validate unique names and constraint parameter references."""
        # Duplicate parameter names
        param_names = [p.name for p in self.parameters]
        dup_params = {n for n in param_names if param_names.count(n) > 1}
        if dup_params:
            msg = f"Duplicate parameter names: {', '.join(sorted(dup_params))}"
            raise ValueError(msg)

        # Duplicate objective names
        obj_names = [o.name for o in self.objectives]
        dup_objs = {n for n in obj_names if obj_names.count(n) > 1}
        if dup_objs:
            msg = f"Duplicate objective names: {', '.join(sorted(dup_objs))}"
            raise ValueError(msg)

        # Constraints reference declared parameters
        parameter_name_set = set(param_names)
        invalid_params = []
        for constraint in self.constraints:
            for parameter in constraint.parameters:
                if parameter not in parameter_name_set:
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
