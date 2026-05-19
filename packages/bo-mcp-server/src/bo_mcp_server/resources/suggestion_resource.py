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


@mcp.resource("suggestions://{campaign_id}")
async def get_suggestions(campaign_id: str) -> str:
    """Get pending suggestions for a campaign.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Markdown listing on success.

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
        suggestions = await suggestion_repo.list_by_campaign(
            campaign_uuid, status=SuggestionStatus.PENDING
        )

        if not suggestions:
            return f"No pending suggestions for campaign {campaign_id}"

        lines = [f"# Pending Suggestions for Campaign {campaign_id}", ""]

        for i, sugg in enumerate(suggestions, 1):
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
