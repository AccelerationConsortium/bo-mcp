"""Transport-protocol propagation for response metadata.

``_metadata.protocol`` used to hardcode ``get_response_metadata``'s
parameter default (``"mcp"``) on every shared-operation response. That
was invisible while REST envelopes dropped ``_metadata``; once they
started forwarding it (issue #82 follow-up), REST bodies advertised
themselves as MCP responses — false transport provenance for
diagnostics and telemetry consumers.

Transports bind the active protocol here at their dispatch boundary —
the REST request middleware binds ``"rest"``; MCP dispatch relies on
the formatter's ``"mcp"`` default — mirroring
:mod:`bo_mcp_server.trace_context`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

protocol_var: ContextVar[str | None] = ContextVar("bo_mcp_protocol", default=None)


def get_bound_protocol() -> str | None:
    """Return the transport protocol bound in the current context, if any."""
    return protocol_var.get()


@contextmanager
def bind_protocol(protocol: str) -> Iterator[None]:
    """Bind the active transport protocol for the duration of the block."""
    token = protocol_var.set(protocol)
    try:
        yield
    finally:
        protocol_var.reset(token)
