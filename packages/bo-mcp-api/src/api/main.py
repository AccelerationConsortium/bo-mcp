"""FastAPI application setup."""

import logging
import time
import uuid as _uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from bo_mcp_server.client import ensure_dev_user, init_database, ping_database_detailed
from bo_mcp_server.trace_context import bind_trace_id
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.metrics import install_metrics
from api.request_context import install_request_id_log_filter, request_id_var
from api.routes import campaigns, capabilities, diagnostics, results, suggestions
from api.settings import WILDCARD_ORIGIN, ApiSettings, get_api_settings

logger = logging.getLogger(__name__)
_api_start_time = time.time()
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler."""
    settings = get_api_settings()
    _assert_dev_auth_safe(settings)
    await init_database()
    if settings.dev_auth:
        await ensure_dev_user()
        logger.warning(
            "DEV_AUTH is enabled; the shared development user is bootstrapped. "
            "This must not be set in production environments."
        )
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

    # Prometheus instrumentation: registers ``/metrics`` and a per-request
    # latency/counter middleware. Mounted before the request-id middleware
    # so duration measurements cover the full handler invocation.
    install_metrics(app)

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
        """
        request_id = request.headers.get("X-Request-ID", str(_uuid.uuid4()))
        trace_id = request.headers.get("X-Trace-Id") or None
        token = request_id_var.set(request_id)
        try:
            with bind_trace_id(trace_id):
                response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = request_id
        if trace_id is not None:
            response.headers["X-Trace-Id"] = trace_id
        return response

    # Include routers — versioned prefix for forward compatibility.
    # Legacy /api/* paths are kept as aliases so existing frontends don't break.
    api_prefix = "/api/v1"
    app.include_router(campaigns.router, prefix=f"{api_prefix}/campaigns", tags=["campaigns"])
    app.include_router(suggestions.router, prefix=f"{api_prefix}/suggestions", tags=["suggestions"])
    app.include_router(results.router, prefix=f"{api_prefix}/results", tags=["results"])
    app.include_router(diagnostics.router, prefix=f"{api_prefix}/diagnostics", tags=["diagnostics"])
    app.include_router(
        capabilities.router, prefix=f"{api_prefix}/capabilities", tags=["capabilities"]
    )

    # Backward-compat aliases at /api/* (no version) for existing clients
    app.include_router(campaigns.router, prefix="/api/campaigns", include_in_schema=False)
    app.include_router(suggestions.router, prefix="/api/suggestions", include_in_schema=False)
    app.include_router(results.router, prefix="/api/results", include_in_schema=False)
    app.include_router(diagnostics.router, prefix="/api/diagnostics", include_in_schema=False)
    app.include_router(capabilities.router, prefix="/api/capabilities", include_in_schema=False)

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

    return app


def main() -> None:
    """CLI entry point for bo-mcp-api."""
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)  # noqa: S104


# Create default app instance
app = create_app()
