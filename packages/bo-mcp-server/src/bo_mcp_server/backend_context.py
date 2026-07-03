"""Campaign-backend propagation for response metadata.

``_metadata.backend`` used to stamp the *server default* backend on every
response, so a BayBE campaign's artifacts claimed BoTorch (issue #57). This
module carries the backend of the campaign the current request is about via
a :class:`~contextvars.ContextVar`, mirroring :mod:`bo_mcp_server.trace_context`.

The variable is set where the campaign's spec becomes known (spec-repository
load, backend resolution at create/validate time). Requests run in their own
asyncio task, so a plain ``set`` without token reset stays request-scoped.
Campaign-agnostic calls (health check, capability listing) never set it and
keep stamping the default backend.
"""

from __future__ import annotations

from contextvars import ContextVar

campaign_backend_var: ContextVar[str | None] = ContextVar("bo_mcp_campaign_backend", default=None)


def get_campaign_backend() -> str | None:
    """Return the backend of the campaign in scope, if any."""
    return campaign_backend_var.get()


def set_campaign_backend(backend: str | None) -> None:
    """Record the campaign's backend for the current request context.

    ``None`` / empty is a no-op so callers can pass optional values through
    without branching.
    """
    if backend:
        campaign_backend_var.set(backend)
