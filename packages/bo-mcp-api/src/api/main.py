"""FastAPI application setup."""

import logging
import time
import uuid as _uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from bo_mcp_server.storage import get_session, init_database
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from api.dev_auth import ensure_dev_user
from api.routes import campaigns, capabilities, diagnostics, results, suggestions

logger = logging.getLogger(__name__)
_api_start_time = time.time()

# Revert reference: `ensure_dev_user` can stay in `api.dev_auth`. To restore
# real API-key auth, revert `get_current_user()` in `api.deps`.


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Application lifespan handler."""
    # Initialize database on startup
    await init_database()
    # Create dev user for testing
    await ensure_dev_user()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="BO MCP API",
        description="REST API proxy for Bayesian Optimization MCP Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS middleware for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Inject a unique X-Request-ID header into every response."""
        request_id = request.headers.get("X-Request-ID", str(_uuid.uuid4()))
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
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
    async def health_check() -> dict[str, str | bool | int]:
        """Health check endpoint for API readiness."""
        db_status = "error"
        try:
            async with get_session() as session:
                await session.execute(text("SELECT 1"))
                db_status = "connected"
        except Exception as e:  # noqa: BLE001 - health checks must never crash
            logger.warning("API health check failed: %s", e)

        healthy = db_status == "connected"
        uptime = int(time.time() - _api_start_time)

        return {
            "healthy": healthy,
            "service": "api",
            "version": app.version,
            "database": db_status,
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
