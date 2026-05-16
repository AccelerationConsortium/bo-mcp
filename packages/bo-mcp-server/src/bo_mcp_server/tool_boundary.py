"""FastMCP tool-boundary error conversion.

FastMCP's ``ToolManager.call_tool`` validates arguments against the
function signature with Pydantic before the tool function runs. When
validation fails it raises ``ToolError`` whose ``__cause__`` is the
original Pydantic ``ValidationError``. Without intervention, agents
see only the stringified ``ToolError`` -- they lose both the
structured ``{success, error, errors, field_errors, ...}`` envelope
the operation layer emits and the dotted-path ``field_errors`` map
that lets them target the offending field.

Widening every boundary argument to ``Any`` and re-validating inside
each tool body would work but pollutes the function signatures and
forces every scalar (``owner_id``, ``campaign_id``, ``submitted_by``,
boolean flags) to be revalidated by hand. Instead, this module wraps
``ToolManager.call_tool`` once per server and converts the
ValidationError-caused ToolError into the same envelope the tool
body would have produced if validation had been deferred. Tools opt
in by registering their canonical "extra" fields (``campaign_id:
None``, ``result_ids: []``, etc.) so the envelope keeps the keys
downstream consumers expect.

Notes:
    * Non-validation ``ToolError`` (tool body raised, ``Unknown tool``)
      are re-raised unchanged so the rest of the MCP stack still
      surfaces them as transport-level errors.
    * Tools not registered in :data:`_TOOL_ENVELOPE_DEFAULTS` pass
      through verbatim -- the wrapper is opt-in, not blanket.

Reference: MCP tool error semantics
https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-call-errors
distinguishes transport errors (``isError: true``) from structured
result envelopes. For payload-shape failures we prefer the structured
envelope so agents handle them with the same code path as inner-field
failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from bo_mcp_server.field_errors import validation_envelope

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


# Canonical fields injected into the boundary-failure envelope per tool
# so downstream consumers see the same shape as a tool-body-driven
# failure. Tools register here by name; missing tools fall through to
# the wrapped call unchanged.
_TOOL_ENVELOPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "bo_create_campaign": {
        "campaign_id": None,
        "spec_id": None,
        "warnings": [],
    },
    "bo_submit_results": {
        "result_ids": [],
        "warnings": [],
        "duplicates_detected": [],
    },
    # ``bo_validate_intake`` is part of the same validation /
    # localization workflow as ``bo_create_campaign``: agents use it
    # as a dry-run before committing to a campaign. Its operation-
    # layer failure path already produces both ``valid=False`` and
    # the canonical envelope keys, so the boundary path mirrors that
    # shape -- the response stays interchangeable regardless of
    # whether the validator caught the breakage or FastMCP did.
    "bo_validate_intake": {
        "valid": False,
        "warnings": [],
        "spec": None,
    },
}


def install_validation_envelope_wrapper(mcp_instance: Any) -> None:
    """Wrap ``mcp_instance._tool_manager.call_tool`` to convert ToolError → envelope.

    Idempotent: a wrapper already installed by an earlier call sets a
    private marker attribute on the bound method so a second call is
    a no-op. This matters because :func:`create_mcp_server` may be
    invoked more than once in a single process (test fixtures, REST
    + MCP co-hosting).
    """
    tool_manager = mcp_instance._tool_manager  # noqa: SLF001 -- the FastMCP seam
    original = tool_manager.call_tool
    if getattr(original, "_bo_mcp_envelope_wrapped", False):
        return

    async def wrapped(
        name: str,
        arguments: dict[str, Any],
        context: Any = None,
        convert_result: bool = False,
    ) -> Any:
        defaults = _TOOL_ENVELOPE_DEFAULTS.get(name)
        try:
            return await original(
                name,
                arguments,
                context=context,
                convert_result=convert_result,
            )
        except ToolError as exc:
            cause = exc.__cause__
            if defaults is None or not isinstance(cause, ValidationError):
                raise
            return validation_envelope(cause, extra=defaults)

    # ``setattr`` keeps ty happy: the attribute is dynamic and we
    # never type-narrow against it, just probe for it on re-entry.
    setattr(wrapped, "_bo_mcp_envelope_wrapped", True)  # noqa: B010
    tool_manager.call_tool = wrapped


_install: Callable[[Any], None] = install_validation_envelope_wrapper
"""Backwards-compatible alias kept short for use inside server.py."""


__all__ = ["install_validation_envelope_wrapper"]


# ---------------------------------------------------------------------------
# Type-checking helper: re-export the coroutine alias FastMCP returns so
# call sites can ``await`` the wrapped manager without ``Any``-spreading.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    ToolCallCoroutine = Coroutine[Any, Any, Any]
