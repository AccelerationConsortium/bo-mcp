"""Shared helpers for operations.

Deduplicates converters used by multiple operations (generate_suggestions,
submit_results) so bug fixes apply in one place.
"""

import logging
from typing import Any
from uuid import UUID

from bo_engine.turbo import TurboState
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
    """Convert domain Result objects to ObservationData for bo-engine."""
    return [
        ObservationData(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            cost=r.metadata.get("cost") if r.metadata else None,
        )
        for r in results
    ]


def turbo_state_to_dict(state: TurboState) -> dict[str, Any]:
    """Serialize TurboState to dictionary for JSON storage."""
    return {
        "dim": state.dim,
        "batch_size": state.batch_size,
        "length": state.length,
        "length_min": state.length_min,
        "length_max": state.length_max,
        "failure_counter": state.failure_counter,
        "failure_tolerance": state.failure_tolerance,
        "success_counter": state.success_counter,
        "success_tolerance": state.success_tolerance,
        "best_value": state.best_value,
        "restart_triggered": state.restart_triggered,
    }


_TURBO_REQUIRED_KEYS = frozenset(
    {
        "dim",
        "batch_size",
        "length",
        "length_min",
        "length_max",
        "failure_counter",
        "failure_tolerance",
        "success_counter",
        "success_tolerance",
        "best_value",
        "restart_triggered",
    }
)


def dict_to_turbo_state(data: dict[str, Any], *, n_parameters: int | None = None) -> TurboState:
    """Deserialize dictionary to TurboState.

    Args:
        data: Serialized TuRBO state dictionary.
        n_parameters: If provided, validates that ``data["dim"]`` matches
            the current campaign's parameter count to catch stale state.

    Raises:
        ValueError: On missing keys or dimension mismatch.
    """
    missing = _TURBO_REQUIRED_KEYS - data.keys()
    if missing:
        raise ValueError(f"Corrupted TuRBO state — missing keys: {sorted(missing)}")

    if n_parameters is not None and data["dim"] != n_parameters:
        raise ValueError(
            f"TuRBO state dimension mismatch: state has dim={data['dim']} "
            f"but campaign has {n_parameters} parameters"
        )

    return TurboState(
        dim=data["dim"],
        batch_size=data["batch_size"],
        length=data["length"],
        length_min=data["length_min"],
        length_max=data["length_max"],
        failure_counter=data["failure_counter"],
        failure_tolerance=data["failure_tolerance"],
        success_counter=data["success_counter"],
        success_tolerance=data["success_tolerance"],
        best_value=data["best_value"],
        restart_triggered=data["restart_triggered"],
    )
