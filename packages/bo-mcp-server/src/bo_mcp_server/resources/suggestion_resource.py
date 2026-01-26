"""Suggestion resource for MCP."""

from uuid import UUID

from bo_mcp_server.domain import SuggestionStatus
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import SuggestionRepository, get_session


@mcp.resource("suggestions://{campaign_id}")
async def get_suggestions(campaign_id: str) -> str:
    """Get pending suggestions for a campaign.

    Args:
        campaign_id: UUID of the campaign

    Returns:
        Formatted list of pending suggestions
    """
    try:
        campaign_uuid = UUID(campaign_id)
    except ValueError:
        return f"Error: Invalid campaign_id format: {campaign_id}"

    async with get_session() as session:
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
        Formatted suggestion details
    """
    try:
        suggestion_uuid = UUID(suggestion_id)
    except ValueError:
        return f"Error: Invalid suggestion_id format: {suggestion_id}"

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)

        suggestion = await suggestion_repo.get(suggestion_uuid)

        if suggestion is None:
            return f"Suggestion {suggestion_id} not found"

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
