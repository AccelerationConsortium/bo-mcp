"""Per-request logging context for the REST API.

The ``X-Request-ID`` middleware in :mod:`api.main` stores the active
request id in :data:`request_id_var`.  Every log record emitted while
a request is being served exposes the id under the ``request_id``
attribute so formatters can address ``%(request_id)s`` unconditionally;
outside of a request the field defaults to ``"-"``.

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

# ``logging.setLogRecordFactory`` accepts any callable that returns a
# ``LogRecord``; the stdlib does not expose a typed alias so we model
# the contract directly here.
LogRecordFactory = Callable[..., logging.LogRecord]

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_FACTORY_INSTALL_LOCK = threading.Lock()
_INSTALLED_FACTORY: LogRecordFactory | None = None


def _make_request_id_factory(base_factory: LogRecordFactory) -> LogRecordFactory:
    """Wrap ``base_factory`` so every record carries the current request id."""

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = base_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
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
    global _INSTALLED_FACTORY  # noqa: PLW0603
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
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
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
