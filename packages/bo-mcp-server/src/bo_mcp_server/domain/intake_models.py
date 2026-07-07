"""Pydantic input models for MCP tool payloads."""

from typing import Annotated, Any, Literal, cast

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

# Objective measurements feed the surrogate's training targets. Pydantic's
# plain ``float`` accepts NaN/±inf (``json.loads`` admits the ``NaN``
# literal and ``1e999`` coerces to ``inf``); once such a value persists it
# fails every subsequent model fit and results cannot be deleted, so the
# schema rejects non-finite values at intake.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


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
    parameters: tuple[InputParameter, ...] = Field(..., min_length=1)
    objectives: tuple[Objective, ...] = Field(..., min_length=1)
    constraints: tuple[Constraint, ...] = Field(default_factory=tuple)
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
    # supplying an explicit seed.
    #
    # **Reproducibility bounds.** Supplying ``random_seed`` makes the Sobol
    # initial design and acquisition multi-start deterministic *within* a
    # fixed (torch version, device, ``torch.use_deterministic_algorithms``
    # setting) triple. Suggestions are NOT guaranteed byte-identical
    # across:
    #
    # * Different ``torch`` minor versions (kernel-fitting drift in
    #   gpytorch / fit_gpytorch_mll changes between releases).
    # * CPU vs. CUDA (and across CUDA driver versions) — float ordering in
    #   reductions differs between devices.
    # * ``torch.use_deterministic_algorithms(False)`` (default) — some
    #   CUDA kernels still use non-deterministic reductions.
    # * Backend swaps (``backend="botorch"`` vs ``backend="baybe"``).
    #
    # A nightly drift test in CI pins suggestions against a golden file for
    # a reference campaign on the production torch version; bumping the
    # torch pin requires regenerating the golden file.
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
    # Per-campaign override for L-BFGS-B restart count / raw-sample budget.
    # Leave None to use the dimension-adaptive defaults in bo-engine.
    acquisition_optimization: AcquisitionOptimizationConfig | None = None
    # ``Literal`` produces an explicit ``enum`` constraint in the
    # generated MCP tool schema, so agents discover the valid backend
    # selectors directly from the schema instead of by failing requests.
    backend: Literal["auto", "botorch", "baybe"] = "auto"
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
    outcome_constraints: tuple[OutcomeConstraint, ...] = Field(default_factory=tuple)
    # Per-spec opt-in to "this backend may silently drop these option
    # fields". Semantically load-bearing options (outcome_constraints,
    # turbo_config, …) are classified as UNSUPPORTED by default on
    # backends that cannot honor them; naming the field here downgrades
    # the rejection to an IGNORED warning so the caller accepts the
    # degraded run knowingly.
    acknowledge_degradations: tuple[str, ...] = Field(default_factory=tuple)

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
        invalid_params: list[str] = []
        for constraint in self.constraints:
            invalid_params.extend(
                parameter
                for parameter in constraint.parameters
                if parameter not in parameter_name_set
            )

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


def inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$ref`` pointers so the schema stands alone.

    Pydantic emits nested models as ``$defs`` references rooted at the
    outer schema. When we splice a sub-schema into another model's
    ``json_schema_extra`` those references dangle, so we walk the tree
    and substitute the referenced definitions in place. This keeps the
    spliced schema self-contained.
    """
    defs = schema.get(_DEFS_KEY, {})

    def walk(node: object) -> object:
        if isinstance(node, dict):
            node_dict = cast("dict[str, Any]", node)
            ref = node_dict.get(_REF_KEY)
            if isinstance(ref, str) and ref.startswith(_REF_PREFIX):
                target = ref[len(_REF_PREFIX) :]
                if target in defs:
                    return walk(defs[target])
            return {k: walk(v) for k, v in node_dict.items() if k != _DEFS_KEY}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return cast("dict[str, Any]", walk(schema))


# Back-compat private alias retained for any callers that imported the
# pre-rename symbol; new code should use :func:`inline_defs`.
_inline_defs = inline_defs


# Projected JSON schemas re-used by tools that accept the corresponding
# payload as a raw ``dict`` / ``list[dict]`` at the MCP boundary
# Keeping the runtime type loose lets the
# operation layer convert ``ValidationError`` into our
# ``field_errors`` envelope instead of letting FastMCP's pre-call
# validation raise an opaque ``ToolError``. ``json_schema_extra``
# splices the rich nested schema back into the tool definition so
# agent introspection still sees the documented fields.
_METADATA_SCHEMA = inline_defs(ResultMetadata.model_json_schema())
INTAKE_INPUT_JSON_SCHEMA = inline_defs(CampaignIntakeInput.model_json_schema())
RESULT_SUBMISSION_JSON_SCHEMA: dict[str, Any] = {}  # populated after ResultSubmissionInput


class ResultSubmissionInput(BaseModel):
    """Validated input payload for each submitted result.

    The ``metadata`` blob is validated against :class:`ResultMetadata`
    so unknown keys fail at intake with a 422 / structured error envelope
    rather than being silently dropped on the way to storage. The
    consumed key set is documented on ``ResultMetadata``; submit a
    no-metadata batch by omitting the field or passing ``{}``.
    """

    parameter_values: dict[str, Any]
    objective_values: dict[str, FiniteFloat]
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


# Populate the projected ``ResultSubmissionInput`` schema once the
# class above is fully defined. We re-inline so a refactor that adds a
# new nested model to the submission payload picks up automatically.
RESULT_SUBMISSION_JSON_SCHEMA = inline_defs(ResultSubmissionInput.model_json_schema())
