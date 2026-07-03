"""Campaign routes."""

from collections.abc import Mapping
from typing import Annotated, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from api.deps import (
    CurrentUser,
    IdempotencyKey,
    ensure_owned_campaigns,
    get_authorized_campaign,
    get_current_user,
    validate_uuid,
)
from api.schemas.campaign import (
    BatchStatusRequest,
    BatchStatusResponse,
    CampaignConfigResponse,
    CampaignCreate,
    CampaignCreateResponse,
    CampaignLifecycleRequest,
    CampaignLifecycleResponse,
    CampaignListResponse,
    CampaignQueryRequest,
    CampaignQueryResponse,
    CampaignResponse,
    CompareCampaignsRequest,
    CompareCampaignsResponse,
    TransferCandidatesRequest,
    TransferCandidatesResponse,
    ValidateIntakeRequest,
    ValidateIntakeResponse,
)
from api.schemas.common import API_RESPONSE_SCHEMA_VERSION
from api.schemas.errors import (
    COMMON_HTTP_ERROR_RESPONSES,
    IDEMPOTENCY_ERROR_RESPONSES,
    operation_failure_response,
)
from api.schemas.intake import IntakeData
from bo_engine.constants import MIN_OBSERVATIONS_FOR_MODEL
from bo_mcp_server.client import (
    CampaignIntakeInput,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    VerbosityLevel,
    batch_get_status_operation,
    canonical_create_campaign_payload,
    compare_campaigns_operation,
    create_campaign_operation,
    discover_transfer_candidates_operation,
    export_campaign_operation,
    format_validate_intake_response,
    get_campaign_with_spec,
    get_spec_for_user,
    http_status_for_error,
    list_campaigns_operation,
    list_owner_campaigns_with_specs,
    manage_campaign_lifecycle_operation,
    run_idempotent_operation,
    validate_intake_operation,
)

router = APIRouter(responses=COMMON_HTTP_ERROR_RESPONSES)


@runtime_checkable
class _ModelDumpable(Protocol):
    def model_dump(self, *, mode: str) -> object: ...


def _mapping_to_dict(value: Mapping[Any, Any]) -> dict[str, object]:
    return {str(key): item for key, item in value.items()}


def _dump_model(value: _ModelDumpable) -> dict[str, object]:
    dumped = value.model_dump(mode="json")
    if isinstance(dumped, Mapping):
        return _mapping_to_dict(dumped)
    return {"value": dumped}


def _dump_optional_model(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if isinstance(value, _ModelDumpable):
        return _dump_model(value)
    if isinstance(value, Mapping):
        return _mapping_to_dict(value)
    return {"value": value}


def _dump_model_list(values: tuple[object, ...] | list[object]) -> list[dict[str, object]]:
    return [dumped for value in values if (dumped := _dump_optional_model(value)) is not None]


def _resolved_initial_design_size(spec: object) -> tuple[int | None, str | None]:
    requested = getattr(spec, "initial_design_size", None)
    if requested is not None:
        return int(requested), "requested"

    # BoTorch uses an initial-design fallback until there are enough
    # observations to fit the first GP. Other backends may not have this
    # phase, so leave the value unset unless BO-MCP resolved to BoTorch.
    if getattr(spec, "backend", None) != "botorch":
        return None, None

    n_parameters = len(getattr(spec, "parameters", ()))
    return max(MIN_OBSERVATIONS_FOR_MODEL, n_parameters + 1), "botorch_default"


def _coerce_intake(intake: IntakeData) -> CampaignIntakeInput:
    """Build a strict domain intake from the validated REST payload.

    ``IntakeData`` already validates the full field shape — parameters,
    objectives, constraints, and the advanced cross-backend knobs
    (``turbo_config``, ``saasbo_config``, ``fidelity_parameter``, …) are
    all parsed into the canonical domain types — so they can be forwarded
    by reference.

    ``CampaignIntakeInput`` additionally enforces cross-field invariants
    that ``IntakeData`` does not (unique parameter/objective names,
    ``backend_options`` routing). Any such error must surface as a 422
    (unprocessable entity) — the same status FastAPI uses for stock body
    validation failures — instead of bubbling up as an unhandled
    ``ValidationError`` 500.
    """
    try:
        return CampaignIntakeInput(
            name=intake.name,
            description=intake.description,
            parameters=intake.parameters,
            objectives=intake.objectives,
            constraints=intake.constraints,
            batch_size=intake.batch_size,
            max_iterations=intake.max_iterations,
            max_observations=intake.max_observations,
            convergence_tolerance=intake.convergence_tolerance,
            initial_design_size=intake.initial_design_size,
            random_seed=intake.random_seed,
            acquisition_optimization=intake.acquisition_optimization,
            backend=intake.backend,
            backend_options=intake.backend_options,
            acquisition_method=intake.acquisition_method,
            use_input_warping=intake.use_input_warping,
            use_cost_aware=intake.use_cost_aware,
            turbo_config=intake.turbo_config,
            saasbo_config=intake.saasbo_config,
            fidelity_parameter=intake.fidelity_parameter,
            transfer_learning=intake.transfer_learning,
            outcome_constraints=intake.outcome_constraints,
            acknowledge_degradations=tuple(intake.acknowledge_degradations),
        )
    except ValidationError as exc:
        # Prefix locations with ("body", "intake") so the error shape
        # matches FastAPI's stock request-validation envelope.
        detail = [
            {
                "type": err.get("type"),
                "loc": ("body", "intake", *err.get("loc", ())),
                "msg": err.get("msg"),
                "input": err.get("input"),
            }
            for err in exc.errors()
        ]
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail
        ) from exc


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    responses={
        200: operation_failure_response(
            model=CampaignCreateResponse,
            description=(
                "Operation-level campaign creation rejection. The HTTP request was "
                "processed, but the campaign was not persisted; inspect success=false "
                "and errors."
            ),
            example={
                "schema_version": API_RESPONSE_SCHEMA_VERSION,
                "success": False,
                "campaign_id": None,
                "spec_id": None,
                "warnings": [],
                "errors": ["Campaign intake is incompatible with the active backend."],
                "idempotency_replay": False,
            },
        ),
        **IDEMPOTENCY_ERROR_RESPONSES,
    },
)
async def create_new_campaign(
    request: CampaignCreate,
    current_user: CurrentUser,
    idempotency_key: IdempotencyKey,
    response: Response,
) -> CampaignCreateResponse:
    """Create a new optimization campaign.

    Returns ``201 Created`` with a ``Location`` header pointing at
    :func:`get_campaign` on success. Operation-level rejections —
    the ``success=False`` envelope produced when intake / capability
    validation fails — keep the historical ``200 OK`` shape so
    existing tests for that contract still receive the envelope
    rather than a redirected HTTP error.

    Honours the ``Idempotency-Key`` request header so retries
    against this endpoint replay the cached response instead of
    creating a duplicate campaign — same semantics as the MCP
    ``bo_create_campaign`` tool's ``idempotency_key`` parameter,
    sharing the same cache namespace so a retry on either transport
    sees the other's prior response.
    """
    intake = _coerce_intake(request.intake)
    owner_id = str(current_user.id)

    async def run(session: AsyncSession) -> dict:
        return await create_campaign_operation(
            intake_data=intake,
            owner_id=owner_id,
            session=session,
        )

    # Shared canonical builder — same shape the MCP wrapper feeds to
    # the cache — so a retry against either transport replays the
    # cached response from the original mutation.
    result = await run_idempotent_operation(
        operation_name="bo_create_campaign",
        idempotency_key=idempotency_key,
        request_payload=canonical_create_campaign_payload(
            intake_data=intake,
            owner_id=owner_id,
        ),
        executor=run,
    )
    # Idempotency-layer envelopes (conflict / in-progress / stale-owner)
    # do not carry the operation's success-shape fields, so we cannot
    # construct a ``CampaignCreateResponse`` from them. Detect the
    # missing field and promote them to an HTTPException with the
    # status code derived from the error code (409 for both conflict
    # and in-progress today).
    if "campaign_id" not in result:
        _raise_idempotency_envelope(result)

    if result.get("success") and result.get("campaign_id"):
        response.headers["Location"] = f"/api/v1/campaigns/{result['campaign_id']}"
    else:
        # Operation-level rejection: the campaign was not persisted,
        # so a 201 would mislead clients. Fall back to 200 with the
        # ``success=False`` envelope intact.
        response.status_code = status.HTTP_200_OK

    return CampaignCreateResponse(
        success=result["success"],
        campaign_id=result["campaign_id"],
        spec_id=result["spec_id"],
        warnings=result.get("warnings", []),
        errors=result["errors"],
        # Forward the wrapper's replay marker so REST clients can
        # distinguish a cached replay from a fresh mutation. The
        # marker is added by ``apply_idempotency`` on the cached
        # response and is absent on the original write.
        idempotency_replay=bool(result.get("idempotency_replay", False)),
    )


def _raise_idempotency_envelope(result: dict) -> None:
    """Promote an idempotency-layer error envelope into an HTTPException.

    The idempotency wrapper short-circuits with conflict / in-progress
    / stale-owner envelopes that lack the operation's success-shape
    fields. Surface them as a typed HTTP error (409 by default for
    both ``IDEMPOTENCY_CONFLICT`` and ``IDEMPOTENCY_IN_PROGRESS``) so
    clients see the structured ``error`` payload instead of a 500
    triggered by an unpacking ``KeyError``.
    """
    raise HTTPException(
        status_code=http_status_for_error(result),
        detail=result.get("error", {"message": "Idempotency error"}),
    )


@router.get("")
async def list_campaigns(current_user: CurrentUser) -> CampaignListResponse:
    """List campaigns for the current user.

    The facade helper batches the spec lookup in a single query, so the
    historical N+1 issue stays fixed without the route reaching into
    repositories itself.
    """
    pairs = await list_owner_campaigns_with_specs(current_user.id)
    responses = [
        CampaignResponse(
            id=str(campaign.id),
            spec_id=str(campaign.spec_id),
            name=spec.name,
            description=spec.description,
            status=campaign.status.value,
            iteration=campaign.iteration,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at,
            n_parameters=spec.n_parameters,
            n_objectives=spec.n_objectives,
        )
        for campaign, spec in pairs
    ]
    return CampaignListResponse(campaigns=responses, total=len(responses))


@router.post("/validate", dependencies=[Depends(get_current_user)])
async def validate_campaign_intake(
    request: ValidateIntakeRequest,
) -> ValidateIntakeResponse:
    """Validate a campaign specification without creating a campaign (dry-run).

    ``ValidateIntakeRequest`` (via :class:`IntakeData`) types every field —
    including the advanced cross-backend knobs — so malformed values are
    rejected by FastAPI at the request boundary with a 422.
    :func:`_coerce_intake` then builds the domain intake without a
    dump/validate round-trip, surfacing any remaining cross-field/domain
    invariant error (unique names, ``backend_options`` routing) as a 422
    rather than a 500; ``validate_intake_operation`` accepts the typed
    ``CampaignIntakeInput`` directly.
    """
    intake = _coerce_intake(request.intake)
    full_result = validate_intake_operation(intake)
    formatted = format_validate_intake_response(full_result, VerbosityLevel.STANDARD)

    return ValidateIntakeResponse(
        valid=formatted["valid"],
        errors=formatted.get("errors", []),
        warnings=formatted.get("warnings", []),
        spec_summary=formatted.get("spec_summary"),
    )


@router.post("/query")
async def query_campaigns(
    request: CampaignQueryRequest,
    current_user: CurrentUser,
) -> CampaignQueryResponse:
    """Query campaigns with filtering, pagination, and verbosity control.

    Pagination is cursor-first: pass the ``next_cursor`` returned by
    the previous response back as ``cursor`` to walk the next page
    safely under concurrent inserts. ``offset`` is preserved for
    backwards-compatibility and is mutually exclusive with ``cursor``;
    supplying both surfaces as an ``HTTPException(400)`` whose
    ``detail`` carries the structured ``error`` envelope from the
    operation layer (code, recovery_action, retryable, details).
    """
    # Pydantic emits a DeprecationWarning every time the deprecated
    # ``offset`` field is read. Suppress it locally so well-behaved
    # callers (who leave it at the default 0) do not see noise; the
    # warning still surfaces in the OpenAPI schema and on actual use
    # via the operation-level mutual-exclusion check.
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        offset = request.offset
    result = await list_campaigns_operation(
        owner_id=current_user.id,
        status=request.status,
        limit=request.limit,
        offset=offset,
        verbosity=request.verbosity.value,
        cursor=request.cursor,
    )
    # The success-shaped ``CampaignQueryResponse`` cannot represent
    # the operation's ``{success: false, error: …}`` envelope —
    # Pydantic would silently drop the unknown ``error`` field on
    # construction, so callers would never see the cursor+offset
    # mutual-exclusion violation.
    # Promote the structured error to an ``HTTPException`` whose
    # ``detail`` is the original envelope so clients inspecting
    # ``response.json()["detail"]["code"]`` can route on it the same
    # way they route on operation envelopes.
    if not result.get("success", False):
        raise HTTPException(
            status_code=http_status_for_error(result),
            detail=result.get("error", {"message": "Query failed"}),
        )
    return CampaignQueryResponse(**result)


@router.post("/status/batch", response_model_exclude_unset=True)
async def batch_campaign_status(
    request: BatchStatusRequest,
    current_user: CurrentUser,
) -> BatchStatusResponse:
    """Get status for multiple campaigns.

    Serialized with ``response_model_exclude_unset=True`` so the body
    stays byte-equal to the MCP ``bo_batch_get_status`` projection (see
    :class:`BatchStatusResponse`).
    """
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await batch_get_status_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return BatchStatusResponse(**result)


@router.post("/compare", response_model_exclude_unset=True)
async def compare_campaign_group(
    request: CompareCampaignsRequest,
    current_user: CurrentUser,
) -> CompareCampaignsResponse:
    """Compare multiple campaigns.

    Serialized with ``response_model_exclude_unset=True`` so the body
    stays byte-equal to the MCP ``bo_compare_campaigns`` projection at
    every verbosity (see :class:`CompareCampaignsResponse`).
    """
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await compare_campaigns_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return CompareCampaignsResponse(**result)


@router.post("/{campaign_id}/lifecycle")
async def manage_campaign(
    campaign_id: str,
    request: CampaignLifecycleRequest,
    current_user: CurrentUser,
) -> CampaignLifecycleResponse:
    """Manage campaign lifecycle."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await manage_campaign_lifecycle_operation(
        campaign_id=campaign_id,
        action=request.action,  # pyright: ignore[reportArgumentType]
    )
    return CampaignLifecycleResponse(**result)


@router.post("/{campaign_id}/transfer-candidates", response_model_exclude_unset=True)
async def discover_campaign_transfer_candidates(
    campaign_id: str,
    request: TransferCandidatesRequest,
    current_user: CurrentUser,
) -> TransferCandidatesResponse:
    """Discover transfer-learning candidates for a campaign.

    Serialized with ``response_model_exclude_unset=True`` so the body
    stays byte-equal to the MCP ``bo_discover_transfer_candidates``
    projection at every verbosity (see
    :class:`TransferCandidatesResponse`).
    """
    await get_authorized_campaign(campaign_id, current_user)

    result = await discover_transfer_candidates_operation(
        campaign_id=campaign_id,
        similarity_threshold=request.similarity_threshold,
        max_candidates=request.max_candidates,
        verbosity=request.verbosity.value,
        parameter_aliases=request.parameter_aliases,
    )
    return TransferCandidatesResponse(**result)


@router.get("/{campaign_id}/export")
async def export_campaign(
    campaign_id: str,
    current_user: CurrentUser,
    output_format: Annotated[str, Query(alias="format")] = "csv",
) -> StreamingResponse:
    """Export all campaign results as a downloadable CSV file."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await export_campaign_operation(
        campaign_id=campaign_id,
        output_format=output_format,
    )

    if not result.get("success", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result.get("errors", ["Export failed"])[0]
            if result.get("errors")
            else result.get("message", "Export failed"),
        )

    # Sanitize campaign name for filename (name comes from the operation)
    raw_name = result.get("campaign_name", f"campaign_{campaign_id[:8]}")
    safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in raw_name)
    filename = safe_name.strip().replace(" ", "_")

    csv_content = result.get("content", "")
    return StreamingResponse(
        iter([csv_content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.csv"',
        },
    )


@router.get("/{campaign_id}/config")
async def get_campaign_config(
    campaign_id: str, current_user: CurrentUser
) -> CampaignConfigResponse:
    """Get a stable, sanitized campaign setup snapshot."""
    try:
        campaign, spec = await get_campaign_with_spec(campaign_id, current_user.id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid campaign_id format",
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign",
        ) from None
    except NotFoundError as exc:
        detail = (
            "Campaign spec not found"
            if exc.resource == "Campaign spec"
            else f"Campaign {campaign_id} not found"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        ) from None

    initial_design_size, initial_design_size_source = _resolved_initial_design_size(spec)

    return CampaignConfigResponse(
        campaign_id=str(campaign.id),
        spec_id=str(campaign.spec_id),
        name=spec.name,
        description=spec.description,
        status=campaign.status.value,
        iteration=campaign.iteration,
        backend_requested=spec.requested_backend,
        backend_resolved=spec.backend,
        batch_size=spec.batch_size,
        max_iterations=spec.max_iterations,
        max_observations=spec.max_observations,
        initial_design_size_requested=spec.initial_design_size,
        initial_design_size=initial_design_size,
        initial_design_size_source=initial_design_size_source,
        random_seed=spec.random_seed,
        convergence_tolerance=spec.convergence_tolerance,
        parameters=_dump_model_list(spec.parameters),
        objectives=_dump_model_list(spec.objectives),
        constraints=_dump_model_list(spec.constraints),
        outcome_constraints=_dump_model_list(spec.outcome_constraints),
        acquisition_method=str(spec.acquisition_method),
        acquisition_optimization=_dump_optional_model(spec.acquisition_optimization),
        use_input_warping=spec.use_input_warping,
        use_cost_aware=spec.use_cost_aware,
        turbo_config=_dump_optional_model(spec.turbo_config),
        saasbo_config=_dump_optional_model(spec.saasbo_config),
        fidelity_parameter=_dump_optional_model(spec.fidelity_parameter),
        transfer_learning=_dump_optional_model(spec.transfer_learning),
        backend_options=(
            {key: dict(value) for key, value in spec.backend_options.items()}
            if spec.backend_options
            else None
        ),
        acknowledge_degradations=list(spec.acknowledge_degradations),
    )


@router.get("/spec/{spec_id}")
async def get_campaign_spec(spec_id: str, current_user: CurrentUser) -> dict:
    """Get campaign spec details for a spec the caller owns via a campaign.

    Specs hold parameter, objective, and constraint shape that is often
    IP-sensitive. Resolving them through the owning campaign — rather
    than treating the spec UUID as a global lookup key — closes the
    cross-tenant IDOR that would otherwise expose every spec to anyone
    who knows or guesses a spec id.
    """
    # validate_uuid raises a 400 directly; preserve that behavior.
    validate_uuid(spec_id, "spec_id")
    try:
        spec = await get_spec_for_user(spec_id, current_user.id)
    except NotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Spec {spec_id} not found",
        ) from None

    return {
        "id": spec_id,
        "name": spec.name,
        "description": spec.description,
        "parameters": [p.model_dump() for p in spec.parameters],
        "objectives": [o.model_dump() for o in spec.objectives],
        "constraints": [c.model_dump() for c in spec.constraints] if spec.constraints else [],
        "batch_size": spec.batch_size,
        "created_at": "",  # Spec doesn't have created_at, use empty string
    }


@router.get("/{campaign_id}")
async def get_campaign(campaign_id: str, current_user: CurrentUser) -> CampaignResponse:
    """Get campaign details."""
    try:
        campaign, spec = await get_campaign_with_spec(campaign_id, current_user.id)
    except InvalidIdentifierError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid campaign_id format",
        ) from None
    except NotAuthorizedError:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this campaign",
        ) from None
    except NotFoundError as exc:
        detail = (
            "Campaign spec not found"
            if exc.resource == "Campaign spec"
            else f"Campaign {campaign_id} not found"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
        ) from None

    return CampaignResponse(
        id=str(campaign.id),
        spec_id=str(campaign.spec_id),
        name=spec.name,
        description=spec.description,
        status=campaign.status.value,
        iteration=campaign.iteration,
        created_at=campaign.created_at,
        updated_at=campaign.updated_at,
        n_parameters=spec.n_parameters,
        n_objectives=spec.n_objectives,
    )
