"""Suggestion resource for MCP.

Errors raise :class:`ResourceOperationError`; the MCP read path
(via the wrapper installed by :mod:`bo_mcp_server.resource_boundary`)
surfaces them as :class:`McpError` JSON-RPC errors with the
structured envelope on ``error.data``. Direct in-process callers
catch the typed exception and read the envelope off ``exc.envelope``
or ``str(exc)``.
"""

from uuid import UUID

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.errors import ErrorCode, raise_resource_error
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import CampaignRepository, SuggestionRepository, get_session

# Ceiling for the Markdown listing: the tool twin (``bo_list_suggestions``)
# paginates with cursors, but resources have no parameters, so a long
# campaign's pending pool would otherwise render as an unbounded blob.
# Overflow is summarized in a trailer pointing at the paginated tool.
MAX_PENDING_SUGGESTIONS_RENDERED = 50


@mcp.resource("suggestions://{campaign_id}")
async def get_suggestions(campaign_id: str) -> str:
    """Get pending suggestions for a campaign.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Markdown listing on success — at most
        :data:`MAX_PENDING_SUGGESTIONS_RENDERED` suggestions (oldest
        first) plus a trailer naming how many more are pending.

    Raises:
        ResourceOperationError: ``INVALID_CAMPAIGN_ID`` for malformed
            ids and ``CAMPAIGN_NOT_FOUND`` when the well-formed id is
            unknown. An existing campaign with no pending suggestions
            still returns the plain Markdown placeholder.
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        raise_resource_error(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )

    async with get_session() as session:
        # Verify the campaign exists before listing suggestions so that
        # ``no rows`` from the suggestions table cannot be confused with
        # ``campaign does not exist``.
        campaign_repo = CampaignRepository(session)
        if await campaign_repo.get(campaign_uuid) is None:
            raise_resource_error(
                ErrorCode.CAMPAIGN_NOT_FOUND,
                message=f"Campaign {campaign_id} not found",
                details={"campaign_id": campaign_id},
            )

        suggestion_repo = SuggestionRepository(session)
        # Count first, then hydrate only the rendered prefix: the pool
        # can be arbitrarily large, so the LIMIT must live in the query,
        # not in a Python slice over fully-hydrated rows. The repository
        # orders by (created_at, id) ascending, keeping the capped view
        # stable across reads.
        pending_counts = await suggestion_repo.count_pending_by_campaigns([campaign_uuid])
        n_pending = pending_counts.get(campaign_uuid, 0)
        if n_pending == 0:
            return f"No pending suggestions for campaign {campaign_id}"

        rendered = await suggestion_repo.list_by_campaign(
            campaign_uuid,
            status=SuggestionStatus.PENDING,
            limit=MAX_PENDING_SUGGESTIONS_RENDERED,
        )
        # max() guards the count-then-fetch race (a suggestion created
        # between the two queries must not produce a negative trailer).
        n_overflow = max(n_pending - len(rendered), 0)

        lines = [f"# Pending Suggestions for Campaign {campaign_id}", ""]

        for i, sugg in enumerate(rendered, 1):
            lines.append(f"## Suggestion {i} (ID: {sugg.id})")
            lines.append("")
            lines.append("**Parameters:**")
            for name, value in sugg.parameter_values.items():
                if isinstance(value, float):
                    lines.append(f"- {name}: {value:.4g}")
                else:
                    lines.append(f"- {name}: {value}")
            lines.append("")
            lines.append(f"**Iteration:** {sugg.provenance.iteration}")
            lines.append(f"**Method:** {sugg.provenance.generation_method}")
            if sugg.provenance.acquisition_value is not None:
                lines.append(f"**Acquisition Value:** {sugg.provenance.acquisition_value:.4g}")
            lines.append("")

        if n_overflow:
            lines.append(
                f"{n_overflow} more pending suggestion(s) not shown — "
                "use bo_list_suggestions with cursor pagination to read them."
            )
            lines.append("")

        return "\n".join(lines)


@mcp.resource("suggestion://{suggestion_id}")
async def get_suggestion(suggestion_id: str) -> str:
    """Get details of a specific suggestion.

    Args:
        suggestion_id: UUID of the suggestion

    Returns:
        Markdown details on success.

    Raises:
        ResourceOperationError: ``VALIDATION_FAILED`` for malformed
            ids; ``SUGGESTION_NOT_FOUND`` when the id is unknown.
    """
    try:
        suggestion_uuid = UUID(suggestion_id)
    except ValueError:
        raise_resource_error(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid suggestion_id format: {suggestion_id}",
            details={"suggestion_id": suggestion_id},
        )

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)

        suggestion = await suggestion_repo.get(suggestion_uuid)

        if suggestion is None:
            raise_resource_error(
                ErrorCode.SUGGESTION_NOT_FOUND,
                message=f"Suggestion {suggestion_id} not found",
                details={"suggestion_id": suggestion_id},
            )

        lines = [
            f"# Suggestion {suggestion.id}",
            "",
            f"**Campaign:** {suggestion.campaign_id}",
            f"**Status:** {suggestion.status.value}",
            f"**Created:** {suggestion.created_at.isoformat()}",
            "",
            "## Parameters",
            "",
        ]

        for name, value in suggestion.parameter_values.items():
            if isinstance(value, float):
                lines.append(f"- **{name}:** {value:.4g}")
            else:
                lines.append(f"- **{name}:** {value}")

        lines.extend(
            [
                "",
                "## Provenance",
                "",
                f"- **Iteration:** {suggestion.provenance.iteration}",
                f"- **Batch Index:** {suggestion.provenance.batch_index}",
                f"- **Method:** {suggestion.provenance.generation_method}",
            ]
        )

        if suggestion.provenance.acquisition_value is not None:
            lines.append(f"- **Acquisition Value:** {suggestion.provenance.acquisition_value:.4g}")
        if suggestion.provenance.model_uncertainty is not None:
            lines.append(f"- **Model Uncertainty:** {suggestion.provenance.model_uncertainty:.4g}")

        return "\n".join(lines)
