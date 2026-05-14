"""Pydantic input models for MCP tool payloads."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from bo_mcp_server.domain.campaign_spec import (
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


class CampaignIntakeInput(BaseModel):
    """Validated input payload for campaign intake-based tools.

    Field set mirrors :class:`CampaignSpec` so every advanced spec
    attribute (TuRBO, SAASBO, fidelity, transfer learning, outcome
    constraints, cost-aware, input warping, acquisition method,
    backend-specific options) can flow API → domain → storage without
    losing information.
    """

    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: list[InputParameter] = Field(..., min_length=1)
    objectives: list[Objective] = Field(..., min_length=1)
    constraints: list[Constraint] = Field(default_factory=list)
    batch_size: int = Field(default=1, ge=1)
    max_iterations: int | None = None
    # Budget / convergence-based stopping (optional). Mirrors the fields on
    # ``CampaignSpec``; see :mod:`bo_engine.convergence.evaluate_stopping_decision`.
    max_observations: int | None = Field(default=None, ge=1)
    convergence_tolerance: float | None = Field(default=None, gt=0.0)
    initial_design_size: int | None = None
    # Default ``None`` so MCP and REST intake forms behave identically when
    # the caller omits the seed: a fresh OS-level scramble each iteration.
    # Callers that want deterministic Sobol sequences must opt in by
    # supplying an explicit seed (see TODO 1.66).
    random_seed: int | None = None
    # Per-campaign override for L-BFGS-B restart count / raw-sample budget.
    # Leave None to use the dimension-adaptive defaults in bo-engine.
    acquisition_optimization: AcquisitionOptimizationConfig | None = None
    backend: str = Field(default="auto", pattern="^(auto|botorch|baybe)$")
    # Typed backend-native option surface; see ``CampaignSpec.backend_options``.
    # Validation rejects options addressed to an explicit non-matching backend
    # so misrouted knobs surface at intake instead of silently disappearing.
    backend_options: dict[str, dict[str, Any]] | None = None
    # Advanced cross-backend knobs — passed through to ``CampaignSpec``
    # unchanged; each is honored only by backends that advertise the
    # corresponding capability via ``validate_capabilities``.
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO
    use_input_warping: bool = False
    use_cost_aware: bool = False
    turbo_config: TurboConfig | None = None
    saasbo_config: SaasboConfig | None = None
    fidelity_parameter: FidelityParameter | None = None
    transfer_learning: TransferLearningConfig | None = None
    outcome_constraints: list[OutcomeConstraint] = Field(default_factory=list)

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def validate_names_and_constraints(self) -> "CampaignIntakeInput":
        """Validate unique names, constraint references, and backend options."""
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

        # backend_options is a dict keyed by backend name. When the caller
        # pins ``backend`` to a concrete name (not "auto"), reject option
        # keys that target a different backend so misrouted knobs fail at
        # intake instead of silently disappearing during conversion.
        if self.backend_options and self.backend != "auto":
            wrong_keys = [k for k in self.backend_options if k != self.backend]
            if wrong_keys:
                msg = (
                    "backend_options keys must match the selected backend "
                    f"({self.backend!r}); got: {', '.join(sorted(wrong_keys))}"
                )
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
