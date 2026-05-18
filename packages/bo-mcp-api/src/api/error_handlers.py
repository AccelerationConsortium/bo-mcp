"""Global exception handlers for the REST API.

Routes have a long tail of code paths that can raise an unexpected
``Exception``: pandas / openpyxl while parsing user-supplied files,
SQLAlchemy from a transient connection issue, third-party libraries
during model fitting. Without a global handler each of those leaks
the raw exception string into the response body — and that string
typically names the library, file path, or dependency version, which
helps an attacker fingerprint the deployment.

The handlers in this module normalise every unhandled exception into
the same structured envelope MCP tools already use
(:func:`bo_mcp_server.errors.make_error_response`) and emit a
correlated server-side log carrying the request id so operators can
pivot from the sanitized client response back to the full traceback.

References
----------
* RFC 9110 §15.6.1 (500 Internal Server Error): the appropriate
  status for unhandled server-side failures.
* OWASP API Security Top 10 ``API8: Security Misconfiguration`` —
  the "no stack traces in production responses" rationale.
* FastAPI exception-handlers documentation
  https://fastapi.tiangolo.com/tutorial/handling-errors/.
"""

from __future__ import annotations

import logging

from bo_mcp_server.client import ErrorCode, make_error_response
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.request_context import request_id_var

logger = logging.getLogger(__name__)


def _current_request_id(request: Request) -> str:
    """Return the request id bound to the active request.

    Prefers the value the middleware bound on the ``ContextVar`` so a
    handler running mid-request sees the same id the response header
    will carry. Falls back to the inbound header (in case the
    middleware re-raised before binding) and finally to the sentinel
    so the field is always present in the envelope.
    """
    bound = request_id_var.get()
    if bound and bound != "-":
        return bound
    header = request.headers.get("X-Request-ID")
    if header:
        return header
    return "-"


def _attach_trace_headers(response: JSONResponse, request_id: str) -> JSONResponse:
    """Echo the request id on the error response so clients can correlate."""
    response.headers["X-Request-ID"] = request_id
    return response


async def handle_unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
    """Normalise an unhandled exception into the structured envelope.

    Every detail beyond the request id stays server-side; the response
    body never carries the exception message, type, or traceback. The
    full traceback is logged at ERROR with ``exc_info`` so operators
    can pivot from the request id surfaced to the client back to the
    underlying failure.
    """
    request_id = _current_request_id(request)
    logger.error(
        "Unhandled exception on %s %s (request_id=%s)",
        request.method,
        request.url.path,
        request_id,
        exc_info=exc,
    )
    envelope = make_error_response(
        ErrorCode.INTERNAL_ERROR,
        details={"request_id": request_id},
    )
    return _attach_trace_headers(
        JSONResponse(status_code=500, content=envelope),
        request_id,
    )


async def handle_http_exception(request: Request, exc: Exception) -> JSONResponse:
    """Preserve handler-raised ``HTTPException`` shape while echoing the request id.

    Route handlers raise :class:`fastapi.HTTPException` with curated
    ``detail`` payloads; the global handler must not rewrite those
    bodies. We just forward them through a :class:`JSONResponse` so
    the request-id header is attached the same way as for the
    catch-all handler.

    The ``exc`` parameter is annotated as :class:`Exception` to match
    FastAPI's :func:`add_exception_handler` signature; the runtime
    only dispatches :class:`~starlette.exceptions.HTTPException`
    subclasses here, so the cast inside is safe.
    """
    assert isinstance(exc, StarletteHTTPException)
    request_id = _current_request_id(request)
    response = JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )
    return _attach_trace_headers(response, request_id)


async def handle_request_validation_error(request: Request, exc: Exception) -> JSONResponse:
    """Forward FastAPI request-validation errors with the request id attached.

    FastAPI already emits a 422 with a structured ``detail`` list for
    Pydantic validation failures. We preserve that shape (clients
    parse it) and just bolt the request-id header on so an operator
    can correlate the rejected request to its log line.

    The ``exc.errors()`` payload can contain non-JSON-serializable
    objects in ``ctx`` — Pydantic 2 captures the underlying
    :class:`ValueError` from custom :class:`model_validator` /
    :class:`field_validator` calls there. Passing the raw list to
    :class:`JSONResponse` would crash the standard ``json.dumps``
    and let the catch-all handler convert the failure into a
    misleading ``500 INTERNAL_ERROR`` envelope. Mirroring FastAPI's
    own :func:`fastapi.exception_handlers.request_validation_exception_handler`,
    we route the payload through :func:`fastapi.encoders.jsonable_encoder`
    so every nested value is coerced to a JSON-safe representation.

    The ``exc`` parameter is annotated as :class:`Exception` to match
    FastAPI's :func:`add_exception_handler` signature; the runtime
    only dispatches :class:`RequestValidationError` here, so the cast
    inside is safe.
    """
    assert isinstance(exc, RequestValidationError)
    request_id = _current_request_id(request)
    response = JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )
    return _attach_trace_headers(response, request_id)


def install_exception_handlers(app: FastAPI) -> None:
    """Register the global handlers on a FastAPI app.

    Ordering matters: Starlette resolves the most specific class
    first, so ``HTTPException`` and ``RequestValidationError`` are
    registered before the bare ``Exception`` catch-all to keep their
    curated bodies intact.
    """
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_unhandled_exception)
