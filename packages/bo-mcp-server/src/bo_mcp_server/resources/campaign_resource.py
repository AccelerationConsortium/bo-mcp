"""Campaign resource for MCP."""

from uuid import UUID

from bo_mcp_server.domain import CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, render_resource_error
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session


def _format_parameters_section(spec: CampaignSpec) -> list[str]:
    """Format the parameters section of a campaign resource."""
    lines: list[str] = []
    for param in spec.parameters:
        if param.bounds:
            bounds_str = f"[{param.bounds.lower}, {param.bounds.upper}]"
            lines.append(f"- **{param.name}** ({param.type.value}): {bounds_str}")
        elif param.categories:
            lines.append(f"- **{param.name}** ({param.type.value}): {param.categories}")
        elif param.values:
            lines.append(f"- **{param.name}** ({param.type.value}): {param.values}")
    return lines


def _format_objectives_section(spec: CampaignSpec) -> list[str]:
    """Format the objectives section of a campaign resource."""
    lines: list[str] = []
    for obj in spec.objectives:
        target = f" (target: {obj.target})" if obj.target else ""
        lines.append(f"- **{obj.name}**: {obj.direction}{target}")
    return lines


def _format_constraints_section(spec: CampaignSpec) -> list[str]:
    """Format the constraints section of a campaign resource."""
    if not spec.constraints:
        return []
    lines = ["", "## Constraints", ""]
    for constraint in spec.constraints:
        lines.append(f"- {constraint.type.value}: {constraint.parameters} = {constraint.value}")
    return lines


@mcp.resource("campaign://{campaign_id}")
async def get_campaign(campaign_id: str) -> str:
    """Get campaign details as a resource.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Campaign details as Markdown text on success, or a JSON-encoded
        structured error envelope (the same shape returned by MCP tools)
        when the campaign cannot be resolved. Agents can detect the error
        path by checking whether the response starts with ``{"success":
        false`` — the envelope carries ``error.code``, ``error.message``,
        and ``error.recovery_action`` fields.
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return render_resource_error(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            return render_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            return render_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign spec not found for campaign {campaign_id}",
                details={"campaign_id": campaign_id, "spec_id": str(campaign.spec_id)},
            )

        lines = [
            f"# Campaign: {spec.name}",
            "",
            f"**ID:** {campaign.id}",
            f"**Status:** {campaign.status.value}",
            f"**Iteration:** {campaign.iteration}",
            f"**Created:** {campaign.created_at.isoformat()}",
            "",
            "## Parameters",
            "",
            *_format_parameters_section(spec),
            "",
            "## Objectives",
            "",
            *_format_objectives_section(spec),
            *_format_constraints_section(spec),
        ]

        return "\n".join(lines)


# Limit pulled from ``list_campaigns_operation.MAX_LIMIT`` so the resource
# advertises the same cap. The default 20 mirrors ``bo_list_campaigns``.
_RESOURCE_DEFAULT_LIMIT = 20
_RESOURCE_MAX_LIMIT = 100


def _parse_resource_limit(raw: str | None) -> int | None:
    if raw is None:
        return _RESOURCE_DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(1, min(value, _RESOURCE_MAX_LIMIT))


def _parse_resource_offset(raw: str | None) -> int | None:
    if raw is None:
        return 0
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(0, value)


def _parse_filters(
    owner: str | None,
    status: str | None,
) -> tuple[UUID | None, CampaignStatus | None] | str:
    """Return parsed ``(owner_uuid, status)`` or a rendered error envelope."""
    owner_uuid: UUID | None = None
    if owner is not None:
        try:
            owner_uuid = UUID(owner)
        except ValueError:
            return render_resource_error(
                ErrorCode.VALIDATION_FAILED,
                message=f"Invalid owner '{owner}'. Must be a UUID.",
                details={"owner": owner},
            )

    status_enum: CampaignStatus | None = None
    if status is not None:
        try:
            status_enum = CampaignStatus(status)
        except ValueError:
            valid = [s.value for s in CampaignStatus]
            return render_resource_error(
                ErrorCode.VALIDATION_FAILED,
                message=f"Invalid status '{status}'. Must be one of: {valid}",
                details={"status": status, "valid_statuses": valid},
            )

    return owner_uuid, status_enum


@mcp.resource("campaigns://list")
async def list_campaigns() -> str:
    """List campaigns (default filters: none, limit 20).

    Workflow: equivalent to calling ``bo_list_campaigns`` with default
    parameters. For owner / status / limit / offset filters use the
    parameterized resource ``campaigns://list/{filters}`` (see below).

    Returns:
        Markdown listing on success; a JSON-encoded structured error
        envelope on failure (same shape as MCP tool errors).
    """
    return await _render_campaigns_listing(
        owner_uuid=None,
        status_enum=None,
        limit=_RESOURCE_DEFAULT_LIMIT,
        offset=0,
    )


_ALLOWED_FILTER_KEYS = ("owner", "status", "limit", "offset", "cursor")


@mcp.resource("campaigns://list/{filters}")
async def list_campaigns_filtered(filters: str) -> str:
    """List campaigns with filters parsed from a query-string segment.

    The ``filters`` segment is parsed as ``key=value`` pairs separated by
    ``&`` (URL-style). Recognised keys mirror the ``bo_list_campaigns``
    tool: ``owner`` (UUID), ``status`` (campaign status string),
    ``limit`` (1-100), ``offset`` (>= 0, deprecated — prefer
    ``cursor``), and ``cursor`` (opaque token from a previous response).
    Unknown keys produce a structured ``VALIDATION_FAILED`` error
    envelope so callers learn about typos instead of silently getting
    the unfiltered list. The rendered response advertises the next
    cursor inline so multi-page workflows can stitch pages together
    without falling back to ``offset``.

    Examples:
        ``campaigns://list/status=running``
        ``campaigns://list/status=running&limit=10``
        ``campaigns://list/owner=00000000-0000-0000-0000-000000000001&limit=5&cursor=<token>``
    """
    parsed_filters: dict[str, str] = {}
    if filters:
        for item in filters.split("&"):
            if not item:
                continue
            if "=" not in item:
                return render_resource_error(
                    ErrorCode.VALIDATION_FAILED,
                    message=(f"Invalid filter segment '{item}': expected 'key=value'."),
                    details={"segment": item},
                )
            key, _, value = item.partition("=")
            parsed_filters[key] = value

    allowed = set(_ALLOWED_FILTER_KEYS)
    unknown = sorted(set(parsed_filters) - allowed)
    if unknown:
        return render_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unknown filter keys: {unknown}. Allowed: {sorted(allowed)}",
            details={"unknown_keys": unknown, "allowed_keys": sorted(allowed)},
        )

    parsed = _parse_filters(parsed_filters.get("owner"), parsed_filters.get("status"))
    if isinstance(parsed, str):
        return parsed
    owner_uuid, status_enum = parsed

    limit = _parse_resource_limit(parsed_filters.get("limit"))
    if limit is None:
        return render_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid limit '{parsed_filters['limit']}'. Must be an integer.",
            details={"limit": parsed_filters["limit"]},
        )

    offset = _parse_resource_offset(parsed_filters.get("offset"))
    if offset is None:
        return render_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid offset '{parsed_filters['offset']}'. Must be an integer.",
            details={"offset": parsed_filters["offset"]},
        )

    return await _render_campaigns_listing(
        owner_uuid=owner_uuid,
        status_enum=status_enum,
        limit=limit,
        offset=offset,
        cursor=parsed_filters.get("cursor"),
    )


async def _render_campaigns_listing(
    owner_uuid: UUID | None,
    status_enum: CampaignStatus | None,
    limit: int,
    offset: int,
    cursor: str | None = None,
) -> str:
    """Run the shared list-campaigns operation and render the result.

    Delegating to :func:`list_campaigns_operation` keeps the resource and
    tool views of the same data perfectly aligned: filters, pagination,
    and total-count semantics never drift.
    """
    response = await list_campaigns_operation(
        owner_id=owner_uuid,
        status=status_enum.value if status_enum is not None else None,
        limit=limit,
        offset=offset,
        verbosity="standard",
        cursor=cursor,
    )

    if not response.get("success"):
        error = response.get("error", {})
        try:
            code = ErrorCode(error.get("code"))
        except ValueError:
            code = ErrorCode.VALIDATION_FAILED
        return render_resource_error(
            code,
            message=error.get("message"),
            details=error.get("details"),
        )

    campaigns = response.get("campaigns", [])
    total = response.get("total_count", 0)
    next_cursor = response.get("next_cursor")

    if not campaigns:
        return f"No campaigns found (total matching filters: {total})."

    header = f"# Campaigns (showing {len(campaigns)} of {total}, offset {offset}, limit {limit})"
    lines = [header, ""]
    for c in campaigns:
        lines.append(
            f"- **{c['name']}** (ID: {c['campaign_id']}): {c['status']}, "
            f"iteration {c.get('iteration', '?')}"
        )
    if next_cursor:
        lines.append("")
        lines.append(f"**Next cursor:** `{next_cursor}`")

    return "\n".join(lines)
