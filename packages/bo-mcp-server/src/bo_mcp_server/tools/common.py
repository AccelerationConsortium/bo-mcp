"""Boundary helpers shared by multiple MCP tool wrappers.

``IntakePayload``, the generic container-shape check, and the
identity-resolution error envelope were each copy-pasted across
``create_campaign.py``, ``validate_intake.py``, ``submit_results.py``,
and ``upload_results_file.py``. Centralizing them here means a fix to
the identity-error shape or the schema-splicing trick applies
everywhere at once instead of drifting between call sites.
"""

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from bo_mcp_server.client import AuthenticationConfigurationError
from bo_mcp_server.errors import ErrorCode, make_error_response
from bo_mcp_server.field_errors import shape_envelope
from bo_mcp_server.response_formatter import attach_response_metadata
from bo_mcp_server.schema_extension import intake_schema_with_backend_extensions

# Shared across every tool that accepts a ``verbosity`` parameter so the
# generated MCP schema declares the same ``enum`` constraint everywhere
# instead of each tool module re-typing (and risking drift from) the
# three-level contract documented in ``response_formatter.VerbosityLevel``.
VerbosityLiteral = Literal["minimal", "standard", "detailed"]

# The tool boundary is typed ``Any`` (not ``dict[str, Any]``) so that
# FastMCP's pre-call Pydantic validator cannot intercept the failure
# when callers send the wrong outer shape (e.g. a string instead of
# an object). Without this, FastMCP would raise an opaque ``ToolError``
# before our wrapper runs and agents would lose both the structured
# envelope and the dotted-path ``field_errors`` map. The shape check
# lives in :func:`check_intake_shape`. Schema discoverability is
# preserved by splicing the full ``CampaignIntakeInput`` JSON schema
# into ``json_schema_extra`` so ``tools/list`` still advertises the
# rich nested structure, enriched with each backend's typed
# ``parameter_options`` shape (e.g. BayBE's ``role=substance`` and
# ``role=custom`` recipes) so agents discover the molecular / custom-
# representation surface from ``tools/list`` directly.
_INTAKE_SCHEMA = intake_schema_with_backend_extensions()
IntakePayload = Annotated[
    Any,
    Field(
        description=(
            "Campaign intake specification. Validated against "
            "CampaignIntakeInput; validation failures are returned as a "
            "structured error envelope with ``field_errors`` keyed by "
            "dotted path."
        ),
        json_schema_extra={
            "type": "object",
            "properties": _INTAKE_SCHEMA.get("properties", {}),
            "required": _INTAKE_SCHEMA.get("required", []),
            "$defs": _INTAKE_SCHEMA.get("$defs", {}),
            "additionalProperties": False,
        },
    ),
]


def check_intake_shape(intake_data: object, *, extra: dict[str, Any]) -> dict[str, Any] | None:
    """Return a structured envelope iff ``intake_data`` is not object-shaped.

    The Pydantic models down the call chain produce per-field
    ``field_errors`` for malformed sub-fields, but their entry points
    expect a mapping. Anything else (string, number, list) would raise
    an opaque ``ToolError`` if we forwarded it. Catching the
    container-shape failure here keeps the agent-facing contract
    identical to inner-field failures. ``extra`` supplies the calling
    tool's success-shape keys (e.g. ``campaign_id: None``) so the
    envelope matches that tool's response shape.
    """
    if isinstance(intake_data, (Mapping, BaseModel)):
        return None
    return shape_envelope(
        "intake_data",
        f"Input should be an object, got {type(intake_data).__name__}",
        extra=extra,
    )


def mcp_identity_error(exc: AuthenticationConfigurationError) -> dict[str, Any]:
    """Return a structured error when MCP cannot resolve a current user."""
    return attach_response_metadata(
        make_error_response(
            ErrorCode.INTERNAL_ERROR,
            message=str(exc),
            details={"transport": "mcp", "missing_identity": True},
        )
    )
