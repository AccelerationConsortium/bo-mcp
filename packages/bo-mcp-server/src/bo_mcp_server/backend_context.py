"""Campaign-backend propagation for response metadata.

``_metadata.backend`` used to stamp the *server default* backend on every
response, so a BayBE campaign's artifacts claimed BoTorch (issue #57). This
module carries the backend of the campaign the current request is about via
a :class:`~contextvars.ContextVar`, mirroring :mod:`bo_mcp_server.trace_context`.

The variable is set where the campaign's spec becomes known (spec-repository
load, backend resolution at create/validate time) and reset at operation
boundaries via :func:`campaign_backend_scope`, which the transport layers
(MCP tool dispatch, REST middleware) enter per request. Requests already run
in their own asyncio task, so bindings do not cross requests; the scope is
defense-in-depth for same-task sequences (in-process ``client`` facade use,
scripts running several operations in one task). Campaign-agnostic calls
(health check, capability listing) never set the variable and stamp the
default backend.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

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


@contextmanager
def campaign_backend_scope() -> Iterator[None]:
    """Isolate the campaign-backend binding for one operation/request.

    Entered at transport boundaries (and usable around any in-process
    operation sequence) so a binding left by a previous operation in the
    same context can never leak into the next response's
    ``_metadata.backend``.
    """
    token = campaign_backend_var.set(None)
    try:
        yield
    finally:
        campaign_backend_var.reset(token)


def with_campaign_backend_scope[**P, R](
    fn: Callable[P, Coroutine[Any, Any, R]],
) -> Callable[P, Coroutine[Any, Any, R]]:
    """Async decorator that runs an operation inside :func:`campaign_backend_scope`.

    For cross-campaign operations (batch status, compare): their envelope
    must stamp the server default, never whichever campaign a previous
    operation in the same context happened to bind. Transports enter the
    scope per dispatch, but in-process callers (client facade, scripts,
    tests) may not — without this, a leaked binding mislabels a
    multi-campaign response as campaign-scoped (issue #82).

    Stack it outermost (above ``with_response_metadata``) so the scope
    still covers the decorator's own metadata-attach step.
    """

    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        with campaign_backend_scope():
            return await fn(*args, **kwargs)

    return wrapper
