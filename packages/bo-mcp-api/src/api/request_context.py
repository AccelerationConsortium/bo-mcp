"""Per-request logging context for the REST API.

The ``X-Request-ID`` middleware in :mod:`api.main` stores the active
request id in :data:`request_id_var`.  Every log record emitted while
a request is being served exposes the id under the ``request_id``
attribute so formatters can address ``%(request_id)s`` unconditionally;
outside of a request the field defaults to ``"-"``.

The same middleware binds the optional ``X-Trace-Id`` workflow id via
:func:`bo_mcp_server.trace_context.bind_trace_id`.  The factory stamps
it on records as ``trace_id`` so JSON logs correlate with the trace
echoed in response metadata and audit events.  Unlike ``request_id``
the attribute is set only while a trace is actually bound — when no
workflow is active the JSON handler's ``CorrelationIdFilter`` backfills
a stable ``null``, and call sites remain free to supply their own
``extra={"trace_id": ...}``.

Implementation: we install a :class:`logging.LogRecordFactory` rather
than a :class:`logging.Filter` on the root logger.  Filters attached
to a logger do **not** decorate records that originated on a child
logger and propagated upward — only the child logger's own filters
run before propagation, so a root-only filter never sees library log
records.  The factory hook runs whenever *any* ``LogRecord`` is
constructed, so the request id reaches records from arbitrary
loggers (FastAPI, uvicorn, SQLAlchemy, our own ``api.*`` modules).

Reference: https://docs.python.org/3/library/logging.html#logging.setLogRecordFactory
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextvars import ContextVar

from bo_mcp_server.trace_context import get_trace_id

# ``logging.setLogRecordFactory`` accepts any callable that returns a
# ``LogRecord``; the stdlib does not expose a typed alias so we model
# the contract directly here.
LogRecordFactory = Callable[..., logging.LogRecord]

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_FACTORY_INSTALL_LOCK = threading.Lock()
_INSTALLED_FACTORY: LogRecordFactory | None = None


def _make_request_id_factory(base_factory: LogRecordFactory) -> LogRecordFactory:
    """Wrap ``base_factory`` so every record carries the request/trace context.

    ``trace_id`` is stamped only while a workflow trace is bound:
    ``Logger.makeRecord`` refuses ``extra=`` keys that already exist on
    the record, so an unconditional assignment would break call sites
    that pass ``extra={"trace_id": ...}`` outside a bound workflow.
    """

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        trace_id = get_trace_id()
        if trace_id is not None:
            record.trace_id = trace_id
        return record

    return factory


def install_request_id_log_factory() -> None:
    """Install the request-id ``LogRecordFactory`` (idempotent).

    The factory is global state, so we wrap whatever factory was active
    at install time (typically ``logging.LogRecord``) and remember the
    wrapped instance so a repeat call is a no-op.  Repeat installation
    is otherwise harmful: each wrap layers an extra ``record.request_id``
    assignment, and the first wrap traps the *original* factory under the
    new one — re-wrapping would cause unbounded recursion.
    """
    global _INSTALLED_FACTORY
    with _FACTORY_INSTALL_LOCK:
        current = logging.getLogRecordFactory()
        if current is _INSTALLED_FACTORY:
            return
        wrapped = _make_request_id_factory(current)
        logging.setLogRecordFactory(wrapped)
        _INSTALLED_FACTORY = wrapped


class RequestIdLogFilter(logging.Filter):
    """Deprecated: kept as a back-compat helper that delegates to the factory.

    Earlier versions installed a :class:`logging.Filter` on the root
    logger; that approach did not reach records emitted by child
    loggers.  The filter is retained so callers that injected it
    explicitly on a handler still get the ``request_id`` field — but
    new code should rely on :func:`install_request_id_log_factory`,
    which the FastAPI middleware does on app construction.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Attach the active request/trace context to the record if not already set."""
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        if not hasattr(record, "trace_id"):
            trace_id = get_trace_id()
            if trace_id is not None:
                record.trace_id = trace_id
        return True


def install_request_id_log_filter(logger: logging.Logger | None = None) -> None:
    """Install the request-id propagation hook.

    Activates the global :func:`install_request_id_log_factory` so that
    every log record — including those emitted by child loggers and
    third-party libraries — picks up the active request id.  When a
    specific ``logger`` is supplied, the legacy
    :class:`RequestIdLogFilter` is also attached as a belt-and-braces
    fallback for handlers that bypass the factory.
    """
    install_request_id_log_factory()
    if logger is None:
        return
    if any(isinstance(existing, RequestIdLogFilter) for existing in logger.filters):
        return
    logger.addFilter(RequestIdLogFilter())
