"""Startup-time safety gates for the REST app.

These tests pin the security defaults that the multi-reviewer audit
asked for in Phase A of section 8:

* Combining ``DEV_AUTH=1`` with ``API_ENV=production`` must refuse to
  start the app — the shared dev user's API key is checked into source
  control and would amount to a published master credential.
* ``CORS_ALLOWED_ORIGINS='*'`` combined with credentialed responses
  must refuse to start — Starlette would otherwise echo the request
  origin and silently re-enable the original CSRF surface.
* The default configuration (no explicit allowlist) must not emit any
  CORS headers, so a deployment that forgets to configure origins
  cannot accidentally expose the API to arbitrary browser origins.
* An explicit allowlist is honored and a foreign origin is rejected.

Reference: Fetch spec — ``Access-Control-Allow-Origin`` and
credentialed responses, https://fetch.spec.whatwg.org/#http-cors-protocol.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import create_app
from api.settings import get_api_settings


@pytest.mark.usefixtures("setup_database")
class TestDevAuthProductionGate:
    """``DEV_AUTH=1`` is incompatible with a production deployment."""

    def test_create_app_refuses_dev_auth_in_production(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("API_ENV", "production")
        monkeypatch.setenv("DEV_AUTH", "1")
        # Default CORS keeps the second guard out of the way.
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

        with pytest.raises(RuntimeError, match="DEV_AUTH=1 is not allowed"):
            create_app()

    def test_dev_auth_allowed_outside_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("API_ENV", "development")
        monkeypatch.setenv("DEV_AUTH", "1")
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

        app = create_app()
        assert app is not None

    @pytest.mark.asyncio
    async def test_dev_auth_lifespan_bootstraps_authenticatable_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``DEV_AUTH=1`` lifespan must bootstrap a user whose key the
        standard auth path accepts.

        Without this end-to-end check, the lifespan branch could regress
        silently — ``httpx.ASGITransport`` does not run lifespan
        automatically, so we drive ``app.router.lifespan_context``
        manually to mirror what uvicorn does at startup.
        """
        from bo_mcp_server.client import DEV_API_KEY

        monkeypatch.setenv("DEV_AUTH", "1")
        monkeypatch.setenv("API_ENV", "development")
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

        app = create_app()
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            response = await client.get("/api/campaigns", headers={"X-API-Key": DEV_API_KEY})

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0  # fresh DB; dev user has no campaigns yet

    @pytest.mark.asyncio
    async def test_dev_auth_off_does_not_bootstrap_dev_user(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without ``DEV_AUTH=1`` the lifespan must NOT seed the dev user."""
        from bo_mcp_server.client import DEV_API_KEY

        monkeypatch.delenv("DEV_AUTH", raising=False)
        monkeypatch.setenv("API_ENV", "development")
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)

        app = create_app()
        async with (
            app.router.lifespan_context(app),
            AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client,
        ):
            response = await client.get("/api/campaigns", headers={"X-API-Key": DEV_API_KEY})

        assert response.status_code == 401


class TestCorsStartupGate:
    """Wildcard origins must not be combined with credentialed responses."""

    def test_wildcard_with_credentials_refuses_to_start(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
        monkeypatch.setenv("API_ENV", "development")

        with pytest.raises(RuntimeError, match="CORS_ALLOWED_ORIGINS='\\*' is unsafe"):
            create_app()

    def test_wildcard_without_credentials_is_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "false")
        monkeypatch.setenv("API_ENV", "development")

        app = create_app()
        assert app is not None


@pytest.mark.usefixtures("setup_database")
class TestCorsRuntime:
    """End-to-end CORS behaviour matches the configured allowlist."""

    @pytest.mark.asyncio
    async def test_default_config_emits_no_cors_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an explicit allowlist, the middleware is not mounted.

        A misconfigured deployment that omits ``CORS_ALLOWED_ORIGINS``
        must not accidentally serve cross-origin requests; the absence
        of the ``Access-Control-Allow-Origin`` header is what enforces
        that closed default.
        """
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health", headers={"Origin": "http://untrusted.example"})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in {key.lower() for key in response.headers}

    @pytest.mark.asyncio
    async def test_allowed_origin_receives_cors_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://app.example,http://other.example")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
        monkeypatch.setenv("API_ENV", "development")

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health", headers={"Origin": "http://app.example"})

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://app.example"
        # The echo of the request origin is acceptable here because the
        # origin is in the allowlist; the spec only forbids the literal
        # wildcard ``*`` with credentialed responses.
        assert response.headers.get("access-control-allow-origin") != "*"

    @pytest.mark.asyncio
    async def test_foreign_origin_is_not_echoed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://app.example")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
        monkeypatch.setenv("API_ENV", "development")

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.get("/health", headers={"Origin": "http://attacker.example"})

        assert response.status_code == 200
        assert "access-control-allow-origin" not in {key.lower() for key in response.headers}

    @pytest.mark.asyncio
    async def test_preflight_for_allowed_origin_with_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Browsers preflight credentialed ``X-API-Key`` requests with ``OPTIONS``.

        The preflight response must mirror the configured origin and
        explicitly permit the ``X-API-Key`` request header; otherwise
        the browser blocks the actual request even though the server
        would have accepted it.

        Reference: Fetch spec — CORS preflight fetch,
        https://fetch.spec.whatwg.org/#cors-preflight-fetch.
        """
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://app.example")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
        monkeypatch.setenv("API_ENV", "development")

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.options(
                "/api/campaigns",
                headers={
                    "Origin": "http://app.example",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "x-api-key",
                },
            )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "http://app.example"
        assert response.headers.get("access-control-allow-credentials") == "true"
        allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
        assert "x-api-key" in allowed_headers
        allowed_methods = response.headers.get("access-control-allow-methods", "").upper()
        assert "GET" in allowed_methods

    @pytest.mark.asyncio
    async def test_preflight_for_foreign_origin_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A preflight from an unallowed origin must not be approved.

        Starlette's CORSMiddleware returns the preflight response with
        no ``Access-Control-Allow-Origin`` header for foreign origins,
        which is enough for the browser to refuse the actual request.
        """
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://app.example")
        monkeypatch.setenv("CORS_ALLOW_CREDENTIALS", "true")
        monkeypatch.setenv("API_ENV", "development")

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.options(
                "/api/campaigns",
                headers={
                    "Origin": "http://attacker.example",
                    "Access-Control-Request-Method": "GET",
                    "Access-Control-Request-Headers": "x-api-key",
                },
            )

        # Whatever the status, the foreign origin must not be echoed.
        assert response.headers.get("access-control-allow-origin") != "http://attacker.example"
        assert "access-control-allow-origin" not in {key.lower() for key in response.headers}


class TestApiSettingsParsing:
    """``ApiSettings`` parses comma-separated origins lists."""

    def test_comma_separated_origins_parse_into_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://a.example, http://b.example ,")
        settings = get_api_settings()
        assert settings.cors_allowed_origins == (
            "http://a.example",
            "http://b.example",
        )

    def test_empty_origins_default_to_empty_tuple(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
        settings = get_api_settings()
        assert settings.cors_allowed_origins == ()

    def test_dev_auth_defaults_to_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEV_AUTH", raising=False)
        settings = get_api_settings()
        assert settings.dev_auth is False
