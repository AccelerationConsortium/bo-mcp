"""Get suggestion explanation tool for MCP."""

import logging
from typing import Any
from uuid import UUID

from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.server import mcp
from bo_mcp_server.storage import SuggestionRepository, get_session

logger = logging.getLogger(__name__)


@mcp.tool()
async def get_suggestion_explanation(suggestion_id: str) -> dict[str, Any]:
    """Get detailed explanation for why a suggestion was generated.

    Args:
        suggestion_id: UUID of the suggestion

    Returns:
        Dictionary with:
            - success: Boolean
            - explanation: Human-readable explanation
            - provenance: Full provenance data
            - errors: List of errors if failed
    """
    logger.debug("Getting explanation for suggestion %s", suggestion_id)

    try:
        suggestion_uuid = UUID(suggestion_id)
    except ValueError:
        logger.warning("Invalid suggestion_id format: %s", suggestion_id)
        response = make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            message="Invalid suggestion_id format",
            details={"suggestion_id": suggestion_id},
        )
        response.update({"explanation": None, "provenance": None})
        return response

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)
        suggestion = await suggestion_repo.get(suggestion_uuid)

        if suggestion is None:
            response = make_error_response(
                ErrorCode.SUGGESTION_NOT_FOUND,
                message=f"Suggestion {suggestion_id} not found",
                details={"suggestion_id": suggestion_id},
            )
            response.update({"explanation": None, "provenance": None})
            return response

        provenance = suggestion.provenance

        # Build detailed explanation
        explanation_parts = []

        if provenance.explanation:
            explanation_parts.append(provenance.explanation)

        explanation_parts.append("\n**Generation Details:**")
        explanation_parts.append(f"- Method: {provenance.generation_method}")
        explanation_parts.append(f"- Model: {provenance.model_type or 'Unknown'}")
        explanation_parts.append(f"- Acquisition: {provenance.acquisition_function or 'Unknown'}")

        if provenance.acquisition_value is not None:
            explanation_parts.append(f"- Acquisition Value: {provenance.acquisition_value:.6f}")

        if provenance.model_uncertainty is not None:
            explanation_parts.append(f"- Model Uncertainty: {provenance.model_uncertainty:.6f}")

        explanation_parts.append(f"- Confidence: {provenance.confidence_level or 'Unknown'}")

        if provenance.random_seed is not None:
            explanation_parts.append(f"- Random Seed: {provenance.random_seed}")

        explanation_parts.append(
            f"- Iteration: {provenance.iteration}, Batch Index: {provenance.batch_index}"
        )

        return {
            "success": True,
            "explanation": "\n".join(explanation_parts),
            "provenance": provenance.model_dump(),
            "errors": [],
        }
