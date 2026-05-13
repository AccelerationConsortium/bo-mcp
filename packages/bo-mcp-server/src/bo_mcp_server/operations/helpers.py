"""Shared helpers for operations.

Deduplicates converters used by multiple operations (generate_suggestions,
submit_results) so bug fixes apply in one place.
"""

import logging
from typing import Any
from uuid import UUID

from bo_engine.types import ObservationData

from bo_mcp_server.domain import Result
from bo_mcp_server.errors import ErrorCode, ValidationError, make_error_response
from bo_mcp_server.response_formatter import VerbosityLevel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception-based helpers (preferred — use in new code)
# ---------------------------------------------------------------------------


def parse_verbosity_strict(verbosity: str) -> VerbosityLevel:
    """Parse a verbosity string, raising on failure.

    Raises:
        ValidationError: If *verbosity* is not a valid level.
    """
    try:
        return VerbosityLevel(verbosity)
    except ValueError:
        raise ValidationError(
            ErrorCode.VALIDATION_FAILED,
            message=f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed",
        ) from None


def parse_campaign_id_strict(campaign_id: str) -> UUID:
    """Parse a campaign_id string into a UUID, raising on failure.

    Raises:
        ValidationError: If *campaign_id* is not a valid UUID.
    """
    try:
        return UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        raise ValidationError(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        ) from None


# ---------------------------------------------------------------------------
# Dict-return helpers (kept for backward compatibility with existing callers)
# ---------------------------------------------------------------------------


def parse_verbosity(verbosity: str) -> VerbosityLevel | dict[str, Any]:
    """Parse a verbosity string into a VerbosityLevel.

    Returns VerbosityLevel on success, or an error response dict on failure.

    .. deprecated:: Use :func:`parse_verbosity_strict` in new code.
    """
    try:
        return VerbosityLevel(verbosity)
    except ValueError:
        return make_error_response(
            ErrorCode.VALIDATION_FAILED,
            message=(
                f"Invalid verbosity '{verbosity}'. Must be one of: minimal, standard, detailed"
            ),
        )


def parse_campaign_id(campaign_id: str) -> UUID | dict[str, Any]:
    """Parse a campaign_id string into a UUID.

    Returns UUID on success, or an error response dict on failure.

    .. deprecated:: Use :func:`parse_campaign_id_strict` in new code.
    """
    try:
        return UUID(campaign_id)
    except ValueError:
        logger.warning("Invalid campaign_id format: %s", campaign_id)
        return make_error_response(
            ErrorCode.INVALID_CAMPAIGN_ID,
            details={"campaign_id": campaign_id},
        )


# ---------------------------------------------------------------------------
# Shared conversion utilities
# ---------------------------------------------------------------------------


def results_to_observations(results: list[Result]) -> list[ObservationData]:
    """Convert domain Result objects to ObservationData for bo-engine.

    Forwards ``Result.measurement_uncertainty`` so the bo-engine can route
    the GP onto a ``FixedNoiseGaussianLikelihood`` when every observation
    has uncertainty for every objective. Missing entries fall through as
    ``None``, which the engine treats as "trainable noise for this batch".
    """
    return [
        ObservationData(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            cost=r.metadata.get("cost") if r.metadata else None,
            measurement_uncertainty=r.measurement_uncertainty,
        )
        for r in results
    ]
