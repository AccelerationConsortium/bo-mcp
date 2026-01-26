"""FastAPI application setup."""

import hashlib
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from bo_mcp_server.domain import User
from bo_mcp_server.storage import UserRepository, get_session, init_database
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.routes import campaigns, diagnostics, results, suggestions

logger = logging.getLogger(__name__)

# Development API key
DEV_API_KEY = "dev-api-key-12345"


async def ensure_dev_user() -> None:
    """Ensure a development user exists for testing."""
    api_key_hash = hashlib.sha256(DEV_API_KEY.encode()).hexdigest()
    async with get_session() as session:
        repo = UserRepository(session)
        existing = await repo.get_by_email("test@example.com")
        if existing:
            logger.info(f"Dev user already exists: {existing.id}")
            return
        user = User(
            name="Test User",
            email="test@example.com",
            api_key_hash=api_key_hash,
        )
        saved = await repo.save(user)
        logger.info(f"Created dev user: {saved.id}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
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
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.get("/")
    async def root() -> RedirectResponse:
        """Redirect root to API docs."""
        return RedirectResponse(url="/docs")

    return app


# Create default app instance
app = create_app()
