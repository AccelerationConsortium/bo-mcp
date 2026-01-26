"""Intake data schemas."""

from typing import Any

from pydantic import BaseModel, Field


class ParameterInput(BaseModel):
    """Parameter definition input."""

    name: str = Field(..., min_length=1)
    type: str = Field(..., pattern="^(continuous|discrete|categorical)$")
    bounds: list[float] | None = None
    values: list[int] | None = None
    categories: list[str] | None = None
    description: str = ""


class ObjectiveInput(BaseModel):
    """Objective definition input."""

    name: str = Field(..., min_length=1)
    direction: str = Field(..., pattern="^(minimize|maximize)$")
    unit: str = ""
    target: float | None = None


class ConstraintInput(BaseModel):
    """Constraint definition input."""

    type: str = Field(..., pattern="^(sum_equals|sum_less_than|sum_greater_than|linear)$")
    parameters: list[str]
    value: float
    coefficients: list[float] | None = None


class IntakeData(BaseModel):
    """Campaign intake data schema."""

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: list[ParameterInput]
    objectives: list[ObjectiveInput]
    constraints: list[ConstraintInput] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    initial_design_size: int | None = None
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for MCP tool."""
        return self.model_dump()
