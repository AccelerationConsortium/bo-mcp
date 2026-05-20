"""FastMCP read-resource boundary: surface :class:`ResourceOperationError` cleanly.

Background: the previous pass made resource
handlers raise :class:`ResourceOperationError` with the structured
``{success: false, error: …}`` envelope embedded in ``str(exc)``. In
isolation that works — direct callers see a clean exception. But the
real wire path runs through three nested ``raise ... from None`` wraps
that mangle the message:

1. ``ResourceTemplate.create_resource`` catches the handler's
   exception and re-raises ``ValueError(f"Error creating resource from
   template: {e}")``.
2. ``ResourceManager.get_resource`` catches **that** ``ValueError`` and
   wraps it **again** with the same prefix.
3. The lowlevel JSON-RPC dispatcher reports
   ``ErrorData(message=str(err), code=0)``.

A client reading the wire response therefore sees ``"Error creating
resource from template: Error creating resource from template: {…clean
envelope…}"`` instead of the envelope alone. The structured payload
still arrives — it just hides behind two stringly-typed wrap prefixes
and the JSON-RPC error code is a generic ``0``.

This module patches the FastMCP instance so the round-trip looks the
way the audit intended:

* ``mcp.read_resource`` is wrapped to walk the ``__context__`` /
  ``__cause__`` chain after a failed read. If the original cause is a
  :class:`ResourceOperationError`, the wrapper re-raises
  :class:`McpError` with the clean JSON envelope as ``message`` and a
  semantically-appropriate JSON-RPC error code (``INVALID_PARAMS`` for
  caller-supplied validation issues, ``INTERNAL_ERROR`` otherwise).
* ``mcp._resource_manager.get_resource`` is also patched because the
  unguarded ``ValueError`` it raises for templates escapes
  ``read_resource``'s own ``try`` block (which only covers
  ``resource.read()``); the wrapper performs the same chain walk
  there.

Both patches are idempotent (sentinel attribute on the bound method)
so they survive multiple ``create_mcp_server`` invocations (test
fixtures, REST + MCP co-hosting). The lowlevel server then maps
``McpError`` to the JSON-RPC error response directly (see
``mcp.server.lowlevel.server.Server`` request dispatch); operators
get a clean wire trace, agents get the structured envelope.

Reference: MCP resource-read error semantics
https://modelcontextprotocol.io/specification/2025-06-18/server/resources
treats failed reads as JSON-RPC errors. ``McpError`` is FastMCP's
typed way of producing one.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from mcp import types
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.resources.base import Resource
from mcp.server.fastmcp.server import ReadResourceContents
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl

from bo_mcp_server.errors import ErrorCode, ResourceOperationError

logger = logging.getLogger(__name__)

# Sentinel attributes pinned on the wrapped bound methods so repeated
# installs are no-ops.
_READ_WRAPPED = "_bo_mcp_resource_read_wrapped"
_GET_WRAPPED = "_bo_mcp_resource_get_wrapped"


def _find_resource_operation_error(exc: BaseException) -> ResourceOperationError | None:
    """Walk the implicit / explicit cause chain looking for our typed error.

    The FastMCP wraps use ``raise ValueError(...)`` without ``from`` so
    the original exception lives on ``__context__``. Operations layer
    code may use ``raise X from Y`` which uses ``__cause__``. We check
    both, with a depth bound so a malformed cycle cannot trap us.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(16):
        if current is None or id(current) in seen:
            return None
        seen.add(id(current))
        if isinstance(current, ResourceOperationError):
            return current
        current = current.__cause__ or current.__context__
    return None


def _json_rpc_code_for(code: ErrorCode) -> int:
    """Map our structured ``ErrorCode`` onto a JSON-RPC numeric code.

    JSON-RPC reserves the -32000..-32099 range for server errors; the
    spec's ``INVALID_PARAMS`` (-32602) is the right semantic match for
    caller-supplied validation problems (bad UUID, unknown filter
    key). Everything else surfaces as ``INTERNAL_ERROR`` so clients
    that route on the code can tell ``my-input-was-wrong`` apart from
    ``server-failed-to-serve``.
    """
    caller_input_codes = {
        ErrorCode.INVALID_CAMPAIGN_ID,
        ErrorCode.VALIDATION_FAILED,
        ErrorCode.CAMPAIGN_NOT_FOUND,
        ErrorCode.SUGGESTION_NOT_FOUND,
    }
    if code in caller_input_codes:
        return types.INVALID_PARAMS
    return types.INTERNAL_ERROR


def _to_mcp_error(roe: ResourceOperationError) -> McpError:
    """Build the structured ``McpError`` carrying the clean envelope."""
    return McpError(
        types.ErrorData(
            code=_json_rpc_code_for(roe.code),
            message=str(roe),
            data=roe.envelope,
        )
    )


def install_resource_envelope_wrapper(mcp_instance: FastMCP) -> None:
    """Patch ``read_resource`` + ``get_resource`` to surface envelopes cleanly.

    Idempotent. Safe to call multiple times.
    """
    _wrap_read_resource(mcp_instance)
    _wrap_resource_manager(mcp_instance)


def _wrap_read_resource(mcp_instance: FastMCP) -> None:
    original = mcp_instance.read_resource
    if getattr(original, _READ_WRAPPED, False):
        return

    async def wrapped(uri: AnyUrl | str) -> Iterable[ReadResourceContents]:
        try:
            return await original(uri)
        except Exception as exc:
            # Resource handlers can raise either ``ResourceOperationError``
            # (which we re-wrap) or any other exception (which we
            # re-raise unchanged). FastMCP's resource manager wraps the
            # original exception in opaque chains, so we have to walk
            # the whole chain via ``_find_resource_operation_error``;
            # the catch-all is the only correct shape here.
            roe = _find_resource_operation_error(exc)
            if roe is None:
                raise
            raise _to_mcp_error(roe) from None

    setattr(wrapped, _READ_WRAPPED, True)
    mcp_instance.read_resource = wrapped  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]


def _wrap_resource_manager(mcp_instance: FastMCP) -> None:
    manager = mcp_instance._resource_manager
    original = manager.get_resource
    if getattr(original, _GET_WRAPPED, False):
        return

    async def wrapped_get(uri: AnyUrl | str, context: Context | None = None) -> Resource | None:
        try:
            return await original(uri, context=context)
        except Exception as exc:
            # Resource handlers can raise either ``ResourceOperationError``
            # (which we re-wrap) or any other exception (which we
            # re-raise unchanged). FastMCP's resource manager wraps the
            # original exception in opaque chains, so we have to walk
            # the whole chain via ``_find_resource_operation_error``;
            # the catch-all is the only correct shape here.
            roe = _find_resource_operation_error(exc)
            if roe is None:
                raise
            raise _to_mcp_error(roe) from None

    setattr(wrapped_get, _GET_WRAPPED, True)
    manager.get_resource = wrapped_get  # type: ignore[method-assign]  # ty: ignore[invalid-assignment]


__all__ = ["install_resource_envelope_wrapper"]
