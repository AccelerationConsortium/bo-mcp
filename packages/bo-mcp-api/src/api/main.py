"""FastAPI application setup."""

import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from bo_mcp_server.storage import get_session, init_database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from sqlalchemy import text

from api.dev_auth import ensure_dev_user
from api.routes import campaigns, diagnostics, results, suggestions

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

    # Include routers
    app.include_router(campaigns.router, prefix="/api/campaigns", tags=["campaigns"])
    app.include_router(suggestions.router, prefix="/api/suggestions", tags=["suggestions"])
    app.include_router(results.router, prefix="/api/results", tags=["results"])
    app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])

    @app.get("/health")
    async def health_check() -> dict[str, str | bool | int]:
        """Health check endpoint for API readiness."""
        db_status = "error"
        try:
            async with get_session() as session:
                await session.execute(text("SELECT 1"))
                db_status = "connected"
        except Exception as e:
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


# Create default app instance
app = create_app()
