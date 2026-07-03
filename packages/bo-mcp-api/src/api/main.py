"""FastAPI application setup."""

import logging
import time
import uuid as _uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from bo_mcp_server.logging_config import configure_logging

# Logging must be configured before the imports below run their side
# effects: the ``bo_mcp_server.client`` chain reaches backend discovery,
# which logs an INFO breadcrumb at import time — that record would
# otherwise bypass BO_MCP_LOG_LEVEL / LOG_FORMAT and the
# PII/correlation filters.
configure_logging()

from fastapi import APIRouter, FastAPI, Request, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.openapi.utils import get_openapi  # noqa: E402
from fastapi.responses import RedirectResponse  # noqa: E402

from api.body_size_middleware import BodySizeLimitMiddleware  # noqa: E402
from api.error_handlers import (  # noqa: E402
    handle_corrupted_json_column,
    handle_unhandled_exception,
    install_exception_handlers,
)
from api.limits import MAX_JSON_REQUEST_BODY_BYTES  # noqa: E402
from api.metrics import install_metrics  # noqa: E402
from api.request_context import install_request_id_log_filter, request_id_var  # noqa: E402
from api.routes import campaigns, capabilities, diagnostics, results, suggestions  # noqa: E402
from api.settings import WILDCARD_ORIGIN, ApiSettings, get_api_settings  # noqa: E402
from bo_mcp_server.client import (  # noqa: E402
    CorruptedJsonColumnError,
    augment_parameter_options,
    bind_trace_id,
    campaign_backend_scope,
    ensure_dev_user,
    idempotency_gc_lifespan,
    init_database,
    ping_database_detailed,
)

# Suffixes the body-size middleware exempts because they apply their
# own per-route streaming reader. Listed as suffixes so both the
# versioned ``/api/v1/results/{id}/upload`` and the legacy
# ``/api/results/{id}/upload`` alias are matched.
_UPLOAD_PATH_SUFFIXES: tuple[str, ...] = ("/upload",)

logger = logging.getLogger(__name__)
_api_start_time = time.time()

# Re-run the bootstrap (idempotent: handlers are replaced, not
# appended). The call above the imports covers import-time records; this
# one evicts any handler a third-party import installed on the root
# logger afterwards. The request-id hook is a global LogRecordFactory,
# independent of handlers, so its ordering relative to the bootstrap
# does not matter; both must run before the first request is served.
configure_logging()
install_request_id_log_filter()


def _assert_dev_auth_safe(settings: ApiSettings) -> None:
    """Refuse to start when development auth is enabled in production.

    The dev-auth bootstrap creates a shared user whose API key is
    checked into source control; allowing it in production would amount
    to publishing a master credential. We fail loudly at startup rather
    than silently leaving the bypass in place.
    """
    if settings.dev_auth and settings.api_env == "production":
        msg = (
            "DEV_AUTH=1 is not allowed when API_ENV=production. "
            "Provision real API keys before deploying to production."
        )
        raise RuntimeError(msg)


def _assert_cors_safe(settings: ApiSettings) -> None:
    """Refuse to start with the wildcard / credentials CORS footgun.

    Per the Fetch spec, ``Access-Control-Allow-Origin: *`` cannot be
    combined with ``Access-Control-Allow-Credentials: true``. Starlette
    works around this by echoing the request origin when both are
    requested, which silently re-introduces the original CSRF surface.
    """
    if WILDCARD_ORIGIN in settings.cors_allowed_origins and settings.cors_allow_credentials:
        msg = (
            "CORS_ALLOWED_ORIGINS='*' is unsafe with CORS_ALLOW_CREDENTIALS=true. "
            "Pin an explicit origin list or disable credentialed responses."
        )
        raise RuntimeError(msg)


def _include_versioned_router(app: FastAPI, router: APIRouter, *, name: str, prefix: str) -> None:
    """Mount a router at both the versioned and the legacy alias path.

    The legacy alias keeps existing frontends working while clients
    migrate to ``/api/v1/...``; it is excluded from the OpenAPI schema
    so the public spec advertises only the supported prefix.
    """
    app.include_router(router, prefix=f"{prefix}/{name}", tags=[name])
    app.include_router(router, prefix=f"/api/{name}", include_in_schema=False)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler.

    The ``_app`` parameter is required by FastAPI's lifespan protocol but
    this implementation does not need a reference to the application
    instance — startup state lives in module-level singletons.
    """
    settings = get_api_settings()
    _assert_dev_auth_safe(settings)
    await init_database()
    if settings.dev_auth:
        await ensure_dev_user()
        logger.warning(
            "DEV_AUTH is enabled; the shared development user is bootstrapped. "
            "This must not be set in production environments."
        )
    async with idempotency_gc_lifespan():
        yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_api_settings()
    _assert_dev_auth_safe(settings)
    _assert_cors_safe(settings)

    app = FastAPI(
        title="BO MCP API",
        description="REST API proxy for Bayesian Optimization MCP Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_allowed_origins),
            allow_credentials=settings.cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Body-size cap. Pure ASGI middleware so the streaming receive
    # wrapper sees the actual bytes regardless of the advertised
    # ``Content-Length`` or transfer encoding. The upload route is
    # exempted because it applies its own per-route streaming cap;
    # every other route — JSON or not — is bounded here.
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_body_size=MAX_JSON_REQUEST_BODY_BYTES,
        upload_paths=_UPLOAD_PATH_SUFFIXES,
    )

    # Prometheus instrumentation: registers ``/metrics`` and a per-request
    # latency/counter middleware. Mounted before the request-id middleware
    # so duration measurements cover the full handler invocation.
    install_metrics(app)

    install_exception_handlers(app)

    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Inject ``X-Request-ID`` and propagate ``X-Trace-Id`` for the request.

        - ``X-Request-ID`` identifies the individual HTTP request; we
          generate one when missing and bind it via a
          :class:`~contextvars.ContextVar` so log records carry it.
        - ``X-Trace-Id`` is the optional workflow id that ties multi-step
          agent calls together. When supplied, it is bound for the
          duration of the request via
          :func:`bo_mcp_server.trace_context.bind_trace_id` so audit
          events and response metadata echo it.

        Unhandled exceptions raised below this middleware are caught
        here and routed through :func:`handle_unhandled_exception`.
        Starlette's :class:`BaseHTTPMiddleware` does not propagate
        exceptions to ``app.add_exception_handler(Exception, ...)``
        cleanly (`encode/starlette#1591`_), so catching at the
        outermost user middleware is the only reliable way to keep
        the sanitization contract for routes that raise after their
        request body is parsed.

        .. _encode/starlette#1591: https://github.com/encode/starlette/issues/1591
        """
        request_id = request.headers.get("X-Request-ID", str(_uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-Id") or None
        token = request_id_var.set(request_id)
        try:
            # campaign_backend_scope isolates the ``_metadata.backend``
            # binding per request, mirroring the trace-id binding.
            with bind_trace_id(trace_id), campaign_backend_scope():
                try:
                    response = await call_next(request)
                except CorruptedJsonColumnError as exc:
                    # Storage-layer JSON corruption gets the dedicated
                    # ``DATA_INTEGRITY_ERROR`` envelope (non-retryable —
                    # the row stays broken until an operator repairs it)
                    # rather than the generic catch-all; without this
                    # dispatch the middleware would mask a known
                    # data-corruption signal as ``INTERNAL_ERROR``.
                    response = await handle_corrupted_json_column(request, exc)
                except Exception as exc:  # noqa: BLE001 - intentional catch-all
                    response = await handle_unhandled_exception(request, exc)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        if trace_id is not None:
            response.headers["X-Trace-Id"] = trace_id
        return response

    # Mount routers on the versioned prefix and keep legacy aliases
    # for clients still on /api/* (excluded from the OpenAPI schema).
    api_prefix = "/api/v1"
    _include_versioned_router(app, campaigns.router, name="campaigns", prefix=api_prefix)
    _include_versioned_router(app, suggestions.router, name="suggestions", prefix=api_prefix)
    _include_versioned_router(app, results.router, name="results", prefix=api_prefix)
    _include_versioned_router(app, diagnostics.router, name="diagnostics", prefix=api_prefix)
    _include_versioned_router(app, capabilities.router, name="capabilities", prefix=api_prefix)

    @app.get("/health")
    async def health_check() -> dict[str, str | bool | int | None]:
        """Health check endpoint for API readiness.

        Carries ``database_error`` when the probe fails so operators
        can distinguish a connectivity outage (``OperationalError``)
        from a credentials/permission issue (``ProgrammingError``) or
        a timeout without grepping the server logs. The class name
        only — exception arguments may include query fragments or
        credentials and are deliberately not surfaced.
        """
        probe = await ping_database_detailed()
        if not probe.healthy:
            logger.warning(
                "API health check: database not reachable (%s)",
                probe.error_class or "unknown",
            )

        uptime = int(time.time() - _api_start_time)

        return {
            "healthy": probe.healthy,
            "service": "api",
            "version": app.version,
            "database": "connected" if probe.healthy else "error",
            "database_error": probe.error_class if not probe.healthy else None,
            "uptime_seconds": uptime,
        }

    @app.get("/")
    async def root() -> RedirectResponse:
        """Redirect root to API docs."""
        return RedirectResponse(url="/docs")

    _install_parameter_options_openapi(app)
    return app


def _install_parameter_options_openapi(app: FastAPI) -> None:
    """Splice typed per-backend ``parameter_options`` into the OpenAPI doc.

    The intake request body's ``parameters[].parameter_options`` is an
    opaque per-backend ``dict`` in the domain model — the neutral schema
    cannot describe a backend's options without importing that backend.
    We post-process the generated OpenAPI so REST/OpenAPI clients discover
    the same typed shape the MCP tool schemas advertise (e.g. BayBE's
    ``role=substance`` recipe). Sourced from the backend-aware
    :func:`bo_mcp_server.schema_extension.augment_parameter_options` so
    ``bo-mcp-api`` never imports a backend package directly. The standard
    FastAPI ``app.openapi`` override pattern caches on ``app.openapi_schema``.
    """

    def custom_openapi() -> dict[str, object]:
        if app.openapi_schema is not None:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        augment_parameter_options(openapi_schema)
        app.openapi_schema = openapi_schema
        return app.openapi_schema

    app.openapi = custom_openapi  # ty: ignore[invalid-assignment]


def main() -> None:
    """CLI entry point for bo-mcp-api."""
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)  # noqa: S104


# Create default app instance
app = create_app()
