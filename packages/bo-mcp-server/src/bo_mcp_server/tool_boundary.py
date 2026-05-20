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
body would have produced if validation had been deferred.

Envelope wrapping is now on by default for every
registered tool. The previous opt-in model required new tool authors
to remember to register in ``_TOOL_ENVELOPE_DEFAULTS`` or their
boundary failures would leak as opaque ``ToolError`` text. The
allowlist now only carries *overrides* — extra keys to merge into the
envelope so the response shape matches a specific tool's success path
(e.g. ``campaign_id: None`` so downstream consumers can index into
the field unconditionally). Tools not listed in the override map
still receive the canonical envelope; only the extra-keys polish is
skipped. A startup invariant in :func:`assert_all_tools_routed_through_wrapper`
catches accidental regressions if a future code path bypasses the
wrapper.

Notes:
    * Non-validation ``ToolError`` (tool body raised, ``Unknown tool``)
      are re-raised unchanged so the rest of the MCP stack still
      surfaces them as transport-level errors.

Reference: MCP tool error semantics
https://modelcontextprotocol.io/specification/2025-06-18/server/tools#tool-call-errors
distinguishes transport errors (``isError: true``) from structured
result envelopes. For payload-shape failures we prefer the structured
envelope so agents handle them with the same code path as inner-field
failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from pydantic import ValidationError

from bo_mcp_server.errors import make_corrupted_json_response
from bo_mcp_server.field_errors import validation_envelope
from bo_mcp_server.storage.models import CorruptedJsonColumnError

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


# Per-tool overrides merged into the boundary-failure envelope so the
# response shape matches the tool's success path. Tools not listed
# here still receive the canonical envelope; only the extra polish
# (campaign_id placeholders, warnings/duplicates arrays, etc.) is
# skipped. The empty default lives in ``_DEFAULT_EXTRA``.
_TOOL_ENVELOPE_OVERRIDES: dict[str, dict[str, Any]] = {
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

# Backwards-compat alias for callers that read the old name. New code
# should reference ``_TOOL_ENVELOPE_OVERRIDES``; the alias keeps a
# previously-public-feeling seam stable for any out-of-tree consumer.
_TOOL_ENVELOPE_DEFAULTS = _TOOL_ENVELOPE_OVERRIDES

_DEFAULT_EXTRA: dict[str, Any] = {}

# Sentinel attribute pinned on the wrapped ``call_tool`` bound method
# so a second installation is a no-op. Externalised so the assertion
# helper can detect "wrapper not installed" reliably.
_WRAPPED_MARKER = "_bo_mcp_envelope_wrapped"


def install_validation_envelope_wrapper(mcp_instance: FastMCP) -> None:
    """Wrap ``mcp_instance._tool_manager.call_tool`` to convert ToolError → envelope.

    Idempotent: a wrapper already installed by an earlier call sets a
    private marker attribute on the bound method so a second call is
    a no-op. This matters because :func:`create_mcp_server` may be
    invoked more than once in a single process (test fixtures, REST
    + MCP co-hosting).

    The wrapper applies to every registered tool: tools listed in
    :data:`_TOOL_ENVELOPE_OVERRIDES` get their extra polish merged in;
    others still get the canonical envelope.
    """
    tool_manager = mcp_instance._tool_manager
    original = tool_manager.call_tool
    if getattr(original, _WRAPPED_MARKER, False):
        return

    async def wrapped(
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
        convert_result: bool = False,
    ) -> object:
        extras = _TOOL_ENVELOPE_OVERRIDES.get(name, _DEFAULT_EXTRA)
        try:
            return await original(
                name,
                arguments,
                context=context,
                convert_result=convert_result,
            )
        except CorruptedJsonColumnError as exc:
            # Convert the typed storage-layer exception into the
            # canonical ``DATA_INTEGRITY_ERROR`` envelope (a distinct,
            # non-retryable code — corrupted rows stay broken until an
            # operator repairs them, so we steer clients away from a
            # retry loop). Without this mapper the exception would
            # propagate as a raw ``RuntimeError`` through FastMCP and
            # leak as opaque text.
            envelope = make_corrupted_json_response(exc)
            return {**extras, **envelope}
        except ToolError as exc:
            cause = exc.__cause__
            if isinstance(cause, CorruptedJsonColumnError):
                envelope = make_corrupted_json_response(cause)
                return {**extras, **envelope}
            if not isinstance(cause, ValidationError):
                raise
            return validation_envelope(cause, extra=extras)

    # ``setattr`` keeps ty happy: the attribute is dynamic and we
    # never type-narrow against it, just probe for it on re-entry.
    setattr(wrapped, _WRAPPED_MARKER, True)
    tool_manager.call_tool = wrapped  # ty: ignore[invalid-assignment]


def assert_all_tools_routed_through_wrapper(mcp_instance: FastMCP) -> None:
    """Startup invariant: every registered tool sees the envelope wrapper.

    The previous opt-in model could silently regress when a new tool
    was added without an entry in the registry — its argument-
    validation failures would leak as raw ``ToolError`` text. With the
    wrapper now installed unconditionally, the remaining failure mode
    is a code path that swaps in an unwrapped ``call_tool`` (e.g. a
    second FastMCP instance constructed without going through
    :func:`create_mcp_server`). This assertion catches that at server
    start so the regression surfaces in the boot log rather than on
    the first failing tool call.
    """
    tool_manager = mcp_instance._tool_manager
    call_tool = getattr(tool_manager, "call_tool", None)
    if call_tool is None or not getattr(call_tool, _WRAPPED_MARKER, False):
        msg = (
            "MCP tool boundary wrapper is not installed; "
            "call install_validation_envelope_wrapper() before serving."
        )
        raise RuntimeError(msg)
    if not getattr(tool_manager, "_tools", None):
        # No tools registered yet — nothing to enforce.
        return


_install: Callable[[Any], None] = install_validation_envelope_wrapper
"""Backwards-compatible alias kept short for use inside server.py."""


__all__ = [
    "assert_all_tools_routed_through_wrapper",
    "install_validation_envelope_wrapper",
]


# ---------------------------------------------------------------------------
# Type-checking helper: re-export the coroutine alias FastMCP returns so
# call sites can ``await`` the wrapped manager without ``Any``-spreading.
# ---------------------------------------------------------------------------
if TYPE_CHECKING:
    ToolCallCoroutine = Coroutine[Any, Any, Any]
