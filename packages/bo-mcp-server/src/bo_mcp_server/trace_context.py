"""Workflow-level trace id propagation.

Agents that string multiple MCP / REST calls together (e.g. ``create
campaign`` → ``generate suggestions`` → ``submit results``) currently
have no first-class way to correlate the resulting audit and log
trails. This module exposes a :class:`~contextvars.ContextVar` they can
set per workflow; the value is echoed in the response ``_metadata``
envelope and recorded on every audit event so a single trace id ties
the full workflow together.

Reference: the same context-var-plus-echo pattern is used by HTTP
``traceparent`` propagation in the W3C trace-context specification —
https://www.w3.org/TR/trace-context/ — and by the OpenTelemetry
``Tracer.start_as_current_span`` ergonomics.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

trace_id_var: ContextVar[str | None] = ContextVar("bo_mcp_trace_id", default=None)


def get_trace_id() -> str | None:
    """Return the trace id active in the current execution context, if any."""
    return trace_id_var.get()


@contextmanager
def bind_trace_id(trace_id: str | None) -> Iterator[None]:
    """Bind ``trace_id`` for the duration of the ``with`` block.

    Setting ``None`` is treated as a no-op so callers can pass an
    optional argument straight through without branching at the call
    site.
    """
    if trace_id is None:
        yield
        return
    token = trace_id_var.set(trace_id)
    try:
        yield
    finally:
        trace_id_var.reset(token)
