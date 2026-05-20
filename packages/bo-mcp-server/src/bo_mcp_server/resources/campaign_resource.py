"""Campaign resource for MCP.

Resource handlers raise :class:`ResourceOperationError` on failure
(via :func:`raise_resource_error`). On the MCP wire path the read-
boundary wrapper installed by :mod:`bo_mcp_server.resource_boundary`
detects that exception on the cause chain and re-raises
:class:`mcp.shared.exceptions.McpError` carrying the structured
envelope on ``error.data`` and a semantic JSON-RPC code — clients see
the read fail at the protocol level rather than receiving a success-
shaped JSON body that secretly carries an error. Direct callers
(unit tests, in-process consumers) read the envelope off
``exc.envelope`` or from ``str(exc)``.
"""

from uuid import UUID

from bo_mcp_server.domain import CampaignSpec, CampaignStatus
from bo_mcp_server.errors import ErrorCode, raise_resource_error
from bo_mcp_server.operations.list_campaigns import list_campaigns_operation
from bo_mcp_server.recovery import MAX_SUGGESTIONS, find_similar_campaign_ids
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, CampaignSpecRepository, get_session

# Default page size for ``campaigns://recent``. Small on purpose — the
# resource exists so an agent can recover from a hallucinated id in a
# single round-trip without paginating the full catalogue.
_RECENT_DEFAULT_LIMIT = 5
# Cap on ``campaigns://recent?limit=N``. Larger and the resource starts
# to look like ``campaigns://list``; smaller and the agent loses the
# ability to scan when the recent set is dominated by closed campaigns.
_RECENT_MAX_LIMIT = 20


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
    lines.extend(
        f"- {constraint.type.value}: {constraint.parameters} = {constraint.value}"
        for constraint in spec.constraints
    )
    return lines


@mcp.resource("campaign://{campaign_id}")
async def get_campaign(campaign_id: str) -> str:
    """Get campaign details as a resource.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Campaign details as Markdown text on success.

    Raises:
        ResourceOperationError: ``INVALID_CAMPAIGN_ID`` for malformed
            ids, ``CAMPAIGN_NOT_FOUND`` for unknown ids (with up to
            ``MAX_SUGGESTIONS`` fuzzy-match suggestions under
            ``error.details.suggestions``). On the MCP wire path the
            read-boundary wrapper translates these to
            :class:`McpError` with ``INVALID_PARAMS`` and the
            structured envelope on ``error.data``; direct callers
            still see the typed exception with the envelope on
            ``exc.envelope`` / ``str(exc)``.
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        raise_resource_error(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        campaign = await campaign_repo.get(campaign_uuid)
        if campaign is None:
            details: dict[str, object] = {"campaign_id": campaign_id}
            # Surface up to ``MAX_SUGGESTIONS`` close-match ids so an
            # LLM that mistyped a single hex digit can pick the right
            # one on the next turn instead of paginating
            # ``campaigns://list``. The candidate pool is the full
            # active set because MCP resources here are process-scoped,
            # not per-caller — see :mod:`bo_mcp_server.resources` for
            # the tenant model (single-tenant-per-process; multi-
            # tenant access flows through the REST surface).
            known_ids = await campaign_repo.list_active_ids()
            suggestions = find_similar_campaign_ids(campaign_id, known_ids, limit=MAX_SUGGESTIONS)
            if suggestions:
                details["suggestions"] = suggestions
            raise_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details=details,
            )

        spec = await spec_repo.get(campaign.spec_id)
        if spec is None:
            raise_resource_error(
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
) -> tuple[UUID | None, CampaignStatus | None]:
    """Return parsed ``(owner_uuid, status)`` or raise a structured error."""
    owner_uuid: UUID | None = None
    if owner is not None:
        try:
            owner_uuid = UUID(owner)
        except ValueError:
            raise_resource_error(
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
            raise_resource_error(
                ErrorCode.VALIDATION_FAILED,
                message=f"Invalid status '{status}'. Must be one of: {valid}",
                details={"status": status, "valid_statuses": valid},
            )

    return owner_uuid, status_enum


def _parse_recent_limit(raw: str | None) -> int | None:
    """Parse ``?limit=N`` for ``campaigns://recent``.

    Returns ``None`` on a non-integer value so the caller can surface a
    structured validation error.
    """
    if raw is None:
        return _RECENT_DEFAULT_LIMIT
    try:
        value = int(raw)
    except ValueError:
        return None
    return max(1, min(value, _RECENT_MAX_LIMIT))


def _render_recent_campaigns(campaigns: list, names: dict[UUID, str]) -> str:
    """Render the recent-campaigns markdown body.

    Split out of the resource handler so the recent / filtered listings
    keep cognitive load low and remain individually testable.
    """
    if not campaigns:
        return "No campaigns recorded yet."
    lines = [
        f"# Recent Campaigns (showing {len(campaigns)})",
        "",
        "Use this resource to recover from a hallucinated campaign id. The "
        "rows are ordered newest-first so the most recently-touched campaign "
        "is at the top of the list.",
        "",
    ]
    for campaign in campaigns:
        name = names.get(campaign.spec_id, "Unknown")
        lines.append(
            f"- **{name}** (ID: {campaign.id}): {campaign.status.value}, "
            f"iteration {campaign.iteration}"
        )
    return "\n".join(lines)


@mcp.resource("campaigns://recent")
async def recent_campaigns() -> str:
    """Return up to ``_RECENT_DEFAULT_LIMIT`` newest campaigns.

    Cheap discovery surface for agents that hit
    ``CAMPAIGN_NOT_FOUND`` and need to confirm which campaigns
    actually exist. Unlike ``campaigns://list`` this resource takes no
    parameters and is bounded — the response is short enough to fit in
    a single LLM context window without paging.
    """
    return await _render_recent_listing(limit=_RECENT_DEFAULT_LIMIT)


@mcp.resource("campaigns://recent/{filters}")
async def recent_campaigns_with_limit(filters: str) -> str:
    """Return the N most recently-created campaigns.

    Accepts ``limit=N`` as the sole filter; the value is clamped to
    ``[1, _RECENT_MAX_LIMIT]``. Other keys raise a structured
    ``VALIDATION_FAILED`` resource error so an LLM that guesses the
    wrong query-string shape learns the right one on the next turn
    instead of silently getting an unbounded list.

    Example: ``campaigns://recent/limit=10``.
    """
    parsed: dict[str, str] = {}
    if filters:
        for item in filters.split("&"):
            if not item:
                continue
            if "=" not in item:
                raise_resource_error(
                    ErrorCode.VALIDATION_FAILED,
                    message=(f"Invalid filter segment '{item}': expected 'key=value'."),
                    details={"segment": item},
                )
            key, _, value = item.partition("=")
            parsed[key] = value

    unknown = sorted(set(parsed) - {"limit"})
    if unknown:
        raise_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unknown filter keys: {unknown}. Allowed: ['limit']",
            details={"unknown_keys": unknown, "allowed_keys": ["limit"]},
        )

    limit = _parse_recent_limit(parsed.get("limit"))
    if limit is None:
        raise_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid limit '{parsed['limit']}'. Must be an integer.",
            details={"limit": parsed["limit"]},
        )

    return await _render_recent_listing(limit=limit)


async def _render_recent_listing(limit: int) -> str:
    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)
        campaigns = await campaign_repo.list_recent(limit=limit)
        spec_ids = list({c.spec_id for c in campaigns})
        specs = await spec_repo.get_by_ids(spec_ids)
        names = {spec_id: spec.name for spec_id, spec in specs.items()}
    return _render_recent_campaigns(campaigns, names)


@mcp.resource("campaigns://list")
async def list_campaigns() -> str:
    """List campaigns (default filters: none, limit 20).

    Workflow: equivalent to calling ``bo_list_campaigns`` with default
    parameters. For owner / status / limit / offset filters use the
    parameterized resource ``campaigns://list/{filters}`` (see below).

    Returns:
        Markdown listing on success. Raises ``ResourceOperationError``
        on failure (same envelope shape as MCP tool errors).
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
    Unknown keys raise a structured ``VALIDATION_FAILED`` resource
    error so callers learn about typos instead of silently getting the
    unfiltered list. The rendered response advertises the next cursor
    inline so multi-page workflows can stitch pages together without
    falling back to ``offset``.

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
                raise_resource_error(
                    ErrorCode.VALIDATION_FAILED,
                    message=(f"Invalid filter segment '{item}': expected 'key=value'."),
                    details={"segment": item},
                )
            key, _, value = item.partition("=")
            parsed_filters[key] = value

    allowed = set(_ALLOWED_FILTER_KEYS)
    unknown = sorted(set(parsed_filters) - allowed)
    if unknown:
        raise_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Unknown filter keys: {unknown}. Allowed: {sorted(allowed)}",
            details={"unknown_keys": unknown, "allowed_keys": sorted(allowed)},
        )

    owner_uuid, status_enum = _parse_filters(
        parsed_filters.get("owner"), parsed_filters.get("status")
    )

    limit = _parse_resource_limit(parsed_filters.get("limit"))
    if limit is None:
        raise_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid limit '{parsed_filters['limit']}'. Must be an integer.",
            details={"limit": parsed_filters["limit"]},
        )

    offset = _parse_resource_offset(parsed_filters.get("offset"))
    if offset is None:
        raise_resource_error(
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
        raise_resource_error(
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
    lines.extend(
        f"- **{c['name']}** (ID: {c['campaign_id']}): {c['status']}, "
        f"iteration {c.get('iteration', '?')}"
        for c in campaigns
    )
    if next_cursor:
        lines.append("")
        lines.append(f"**Next cursor:** `{next_cursor}`")

    return "\n".join(lines)
