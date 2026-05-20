"""Operational configuration for ``bo-mcp-api``.

Each accessor instantiates :class:`ApiSettings` on every call so test
overrides applied via ``monkeypatch.setenv`` are observed immediately —
the same pattern :mod:`bo_mcp_server.settings` uses for the server side.

Reference: pydantic-settings ``BaseSettings`` documentation,
https://docs.pydantic.dev/latest/concepts/pydantic_settings/.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

WILDCARD_ORIGIN = "*"


class ApiSettings(BaseSettings):
    """REST-transport configuration sourced from environment variables.

    All knobs are optional and default to safe values: production-style
    deployments must explicitly turn on dev-auth or open CORS.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    api_env: Literal["development", "staging", "production"] = Field(
        default="development",
        alias="API_ENV",
        description="Deployment environment; production refuses to start with DEV_AUTH=1.",
    )
    dev_auth: bool = Field(
        default=False,
        alias="DEV_AUTH",
        description="When true, lifespan bootstraps the shared development user so the "
        "DEV_API_KEY resolves via the standard API-key lookup. Refused when API_ENV=production.",
    )
    # Pydantic-settings auto-parses complex types (list/tuple) as JSON. We
    # accept the more ergonomic comma-separated form by keeping the raw
    # env value as a string and exposing the parsed tuple via a computed
    # property. Reference:
    # https://docs.pydantic.dev/latest/concepts/pydantic_settings/#parsing-environment-variable-values.
    cors_allowed_origins_raw: str = Field(
        default="",
        alias="CORS_ALLOWED_ORIGINS",
        description="Comma-separated origin allowlist. Empty disables cross-origin requests.",
    )
    cors_allow_credentials: bool = Field(
        default=True,
        alias="CORS_ALLOW_CREDENTIALS",
        description="Whether to emit credentialed CORS responses. Refused with wildcard origins.",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_allowed_origins(self) -> tuple[str, ...]:
        """Return the parsed origin allowlist as an immutable tuple."""
        raw = self.cors_allowed_origins_raw or ""
        return tuple(part.strip() for part in raw.split(",") if part.strip())


def get_api_settings() -> ApiSettings:
    """Return a fresh :class:`ApiSettings` instance.

    Re-reads ``os.environ`` on every call so tests using
    ``monkeypatch.setenv`` see their overrides without explicit cache
    plumbing.
    """
    return ApiSettings()
