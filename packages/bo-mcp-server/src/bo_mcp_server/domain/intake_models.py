"""Pydantic input models for MCP tool payloads."""

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

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
from bo_mcp_server.domain.result import ResultMetadata


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


_DEFS_KEY = "$defs"
_REF_KEY = "$ref"
_REF_PREFIX = f"#/{_DEFS_KEY}/"


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` pointers so the schema stands alone.

    Pydantic emits nested models as ``$defs`` references rooted at the
    outer schema. When we splice a sub-schema into another model's
    ``json_schema_extra`` those references dangle, so we walk the tree
    and substitute the referenced definitions in place. This keeps the
    metadata schema self-contained inside the splice point.
    """
    defs = schema.get(_DEFS_KEY, {})

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get(_REF_KEY)
            if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
                target = ref[len(_REF_PREFIX) :]
                if target in defs:
                    return walk(defs[target])
            return {k: walk(v) for k, v in node.items() if k != _DEFS_KEY}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(schema)


# Projected JSON schema for the metadata sub-blob. Generated lazily so
# tests / agents see a stable shape regardless of import order. The
# field below uses ``json_schema_extra`` to splice this directly into
# the generated tool schema (TODO 1.12 review): the runtime type is
# still ``dict[str, Any]`` to preserve compatibility with persisted
# rows, but agents see the documented key set in the MCP tool
# definition instead of the bare ``additionalProperties: true``.
_METADATA_SCHEMA = _inline_defs(ResultMetadata.model_json_schema())


class ResultSubmissionInput(BaseModel):
    """Validated input payload for each submitted result.

    The ``metadata`` blob is validated against :class:`ResultMetadata`
    so unknown keys fail at intake with a 422 / structured error envelope
    rather than being silently dropped on the way to storage. The
    consumed key set is documented on ``ResultMetadata``; submit a
    no-metadata batch by omitting the field or passing ``{}``.
    """

    parameter_values: dict[str, Any]
    objective_values: dict[str, float]
    suggestion_id: str | None = None
    measurement_uncertainty: dict[str, float] | None = None  # Per-objective noise std
    # Runtime type is dict[str, Any] for backward-compat with persisted
    # rows; the JSON schema is overridden to reference the
    # ResultMetadata key set so agents introspect the documented schema.
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional per-result metadata. Validated against the "
            "ResultMetadata schema (see properties below); unknown keys "
            "are rejected at intake."
        ),
        json_schema_extra={
            "properties": _METADATA_SCHEMA.get("properties", {}),
            "$defs": _METADATA_SCHEMA.get("$defs", {}),
            "additionalProperties": False,
        },
    )

    model_config = {"extra": "forbid"}

    @field_validator("metadata")
    @classmethod
    def _validate_metadata(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Reject unknown metadata keys at intake time.

        Round-trips through :class:`ResultMetadata` so the schema lives
        in one place. We re-dump (excluding unset fields) so the stored
        blob keeps its original shape — callers do not get
        ``{"cost": null, "operator": null, ...}`` filled in for keys
        they never sent.
        """
        if not value:
            return value
        typed = ResultMetadata.model_validate(value)
        return typed.model_dump(exclude_unset=True, mode="json")
