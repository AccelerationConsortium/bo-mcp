"""Campaign routes."""

from bo_mcp_server.client import (
    CampaignIntakeInput,
    InvalidIdentifierError,
    NotAuthorizedError,
    NotFoundError,
    VerbosityLevel,
    batch_get_status_operation,
    compare_campaigns_operation,
    create_campaign_operation,
    discover_transfer_candidates_operation,
    export_campaign_operation,
    format_validate_intake_response,
    get_campaign_with_spec,
    get_spec_for_user,
    list_campaigns_operation,
    list_owner_campaigns_with_specs,
    manage_campaign_lifecycle_operation,
    validate_intake_operation,
)
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from api.deps import (
    CurrentUser,
    ensure_owned_campaigns,
    get_authorized_campaign,
    validate_uuid,
)
from api.schemas.campaign import (
    BatchStatusRequest,
    BatchStatusResponse,
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
from api.schemas.intake import IntakeData

router = APIRouter()


def _coerce_intake(intake: IntakeData) -> CampaignIntakeInput:
    """Build a strict domain intake from the validated REST payload.

    ``IntakeData`` already validates the shared field shape: parameters,
    objectives, and constraints are parsed into the canonical domain
    types, so they can be forwarded by reference. The advanced
    cross-backend knobs (``turbo_config``, ``saasbo_config``,
    ``fidelity_parameter``, …) stay typed as plain dict on REST so the
    schema does not couple to backend-specific Pydantic models;
    ``CampaignIntakeInput`` validates their inner shape here.

    Any validation error on the advanced knobs must surface as a 422
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
            acquisition_optimization=intake.acquisition_optimization,  # ty: ignore[invalid-argument-type]
            backend=intake.backend,
            backend_options=intake.backend_options,
            acquisition_method=intake.acquisition_method,  # ty: ignore[invalid-argument-type]
            use_input_warping=intake.use_input_warping,
            use_cost_aware=intake.use_cost_aware,
            turbo_config=intake.turbo_config,  # ty: ignore[invalid-argument-type]
            saasbo_config=intake.saasbo_config,  # ty: ignore[invalid-argument-type]
            fidelity_parameter=intake.fidelity_parameter,  # ty: ignore[invalid-argument-type]
            transfer_learning=intake.transfer_learning,  # ty: ignore[invalid-argument-type]
            outcome_constraints=intake.outcome_constraints,  # ty: ignore[invalid-argument-type]
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


@router.post("", response_model=CampaignCreateResponse)
async def create_new_campaign(
    request: CampaignCreate,
    current_user: CurrentUser,
) -> CampaignCreateResponse:
    """Create a new optimization campaign."""
    intake = _coerce_intake(request.intake)
    result = await create_campaign_operation(
        intake_data=intake,
        owner_id=str(current_user.id),
    )
    return CampaignCreateResponse(
        success=result["success"],
        campaign_id=result["campaign_id"],
        spec_id=result["spec_id"],
        warnings=result.get("warnings", []),
        errors=result["errors"],
    )


@router.get("", response_model=CampaignListResponse)
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


@router.post("/validate", response_model=ValidateIntakeResponse)
async def validate_campaign_intake(
    request: ValidateIntakeRequest,
    current_user: CurrentUser,
) -> ValidateIntakeResponse:
    """Validate a campaign specification without creating a campaign (dry-run).

    Builds the domain intake from the validated REST payload (without a
    dump/validate round-trip; see :func:`_coerce_intake`) so any
    validation error on the advanced cross-backend knobs surfaces as a
    422 instead of a 500. ``validate_intake_operation`` accepts the typed
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


@router.post("/query", response_model=CampaignQueryResponse)
async def query_campaigns(
    request: CampaignQueryRequest,
    current_user: CurrentUser,
) -> CampaignQueryResponse:
    """Query campaigns with filtering, pagination, and verbosity control."""
    result = await list_campaigns_operation(
        owner_id=current_user.id,
        status=request.status,
        limit=request.limit,
        offset=request.offset,
        verbosity=request.verbosity,
    )
    return CampaignQueryResponse(**result)


@router.post("/status/batch", response_model=BatchStatusResponse)
async def batch_campaign_status(
    request: BatchStatusRequest,
    current_user: CurrentUser,
) -> BatchStatusResponse:
    """Get status for multiple campaigns."""
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await batch_get_status_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return BatchStatusResponse(**result)


@router.post("/compare", response_model=CompareCampaignsResponse)
async def compare_campaign_group(
    request: CompareCampaignsRequest,
    current_user: CurrentUser,
) -> CompareCampaignsResponse:
    """Compare multiple campaigns."""
    await ensure_owned_campaigns(request.campaign_ids, current_user)

    result = await compare_campaigns_operation(
        campaign_ids=request.campaign_ids,
        verbosity=request.verbosity.value,
    )
    return CompareCampaignsResponse(**result)


@router.post(
    "/{campaign_id}/lifecycle",
    response_model=CampaignLifecycleResponse,
)
async def manage_campaign(
    campaign_id: str,
    request: CampaignLifecycleRequest,
    current_user: CurrentUser,
) -> CampaignLifecycleResponse:
    """Manage campaign lifecycle."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await manage_campaign_lifecycle_operation(
        campaign_id=campaign_id,
        action=request.action,  # pyright: ignore[reportArgumentType]  # ty: ignore[invalid-argument-type]
    )
    return CampaignLifecycleResponse(**result)


@router.post(
    "/{campaign_id}/transfer-candidates",
    response_model=TransferCandidatesResponse,
)
async def discover_campaign_transfer_candidates(
    campaign_id: str,
    request: TransferCandidatesRequest,
    current_user: CurrentUser,
) -> TransferCandidatesResponse:
    """Discover transfer-learning candidates for a campaign."""
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
    format: str = Query(default="csv"),
) -> StreamingResponse:
    """Export all campaign results as a downloadable CSV file."""
    await get_authorized_campaign(campaign_id, current_user)

    result = await export_campaign_operation(
        campaign_id=campaign_id,
        format=format,
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


@router.get("/{campaign_id}", response_model=CampaignResponse)
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
