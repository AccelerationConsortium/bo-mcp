"""Shared helpers for operations.

Deduplicates converters used by multiple operations (generate_suggestions,
submit_results) so bug fixes apply in one place.
"""

import logging
from typing import Any
from uuid import UUID

from bo_engine.types import ObservationData, TargetMode
from bo_mcp_server.domain import Result
from bo_mcp_server.domain.campaign_spec import Objective
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


def objective_analysis_is_minimize(objective: Objective) -> bool:
    """Direction of the analysis metric from :func:`objective_analysis_series`.

    MATCH objectives are analyzed on distance-to-target, which minimizes;
    every other goal keeps its declared direction. Directional analytics
    (best value, improvement history, convergence) must use this instead
    of the raw ``is_minimize`` boolean, which cannot express MATCH and
    would silently analyze "hit pH 7.4" as maximization of the raw value.
    """
    if objective.effective_mode == TargetMode.MATCH and objective.target is not None:
        return True
    return objective.is_minimize


def objective_analysis_series(
    objective: Objective,
    values: list[float],
) -> tuple[list[float], bool]:
    """Metric trajectory + direction for best-value/improvement analytics.

    Returns ``(metric_values, minimize)``: MATCH objectives yield the
    absolute distance to ``objective.target`` (best = smallest), matching
    the BayBE backend's own diagnostics convention; every other goal
    passes the raw values through with the declared direction, so
    existing non-MATCH analytics are byte-identical.
    """
    if objective.effective_mode == TargetMode.MATCH and objective.target is not None:
        return [abs(v - objective.target) for v in values], True
    return list(values), objective.is_minimize


def objective_identity(objective: Objective) -> str:
    """Order-independent identity of one objective's optimization goal.

    Built from the *resolved* mode (never the raw optional ``direction``
    string, which renders as ``None`` for ``target_mode`` spellings), and
    including the target value for MATCH — two MATCH objectives with
    different targets are different goals, while legacy-``direction`` and
    ``target_mode`` spellings of the same goal are identical.
    """
    mode = objective.effective_mode
    if mode == TargetMode.MATCH:
        return f"{objective.name}:{mode.value}:{objective.target}"
    return f"{objective.name}:{mode.value}"


def results_to_observations(results: list[Result]) -> list[ObservationData]:
    """Convert domain Result objects to ObservationData for bo-engine.

    Forwards ``Result.measurement_uncertainty`` so the bo-engine can route
    the GP onto a ``FixedNoiseGaussianLikelihood`` when every observation
    has uncertainty for every objective. Missing entries fall through as
    ``None``, which the engine treats as "trainable noise for this batch".

    ``Result.id`` is threaded through as ``ObservationData.result_id`` so
    backends that serialise per-observation identity (notably BayBE)
    can use a stable cross-system discriminator instead of a
    parameter/objective fingerprint that collapses replicates onto a
    single identity slot. Without the discriminator, two identical
    replicate rows are indistinguishable in the serialized identity index
    and reconciliation has to fall back to a positional ``Counter``.
    """
    return [
        ObservationData(
            parameter_values=r.parameter_values,
            objective_values=r.objective_values,
            cost=r.metadata.get("cost") if r.metadata else None,
            measurement_uncertainty=r.measurement_uncertainty,
            result_id=str(r.id),
        )
        for r in results
    ]
