"""Drift-control tests for the ``/manpage`` operating-manual surface.

Four guards keep the served manual and the application in sync:

1. Route presence/shape — ``/manpage`` (negotiated HTML/Markdown) and
   ``/manpage.md`` (raw Markdown) exist, are public, and carry
   ``ETag`` / ``Cache-Control``.
2. Referenced-path validity — every ``METHOD /api/v1/...`` endpoint the
   manual cites (inline code, pseudocode comments, and the recorded
   curl session alike) must exist in the live OpenAPI schema with that
   method, so renaming an endpoint without updating the prose fails CI.
3. Internal anchors + link policy — every ``](#...)`` target resolves
   to a generated heading slug, and every other link target is a
   service-served path (never a repository file path or external host).
4. Discoverability — the OpenAPI ``externalDocs`` and the app
   description point readers at ``/manpage``.

Reference: the drift-control pattern (documentation validity asserted
against the live application schema) follows FastAPI's own
``app.openapi()`` testing guidance; see also RFC 9110 §8.8.3 (ETag) and
§12.5.1 (Accept) for the HTTP semantics exercised here.
"""

import re
from typing import ClassVar

import pytest

from api.main import create_app
from api.routes.manpage import (
    MANPAGE_CACHE_MAX_AGE_SECONDS,
    extract_headings,
    slugify_heading,
)
from bo_mcp_server.client import load_manpage_markdown

MANPAGE = load_manpage_markdown()

# How the manual cites endpoints (a documented convention, greppable):
# an HTTP method followed by an absolute service path. Matches inline
# code, fenced pseudocode comments, and curl-session annotations alike.
_ENDPOINT_REFERENCE = re.compile(
    r"\b(GET|POST|PUT|PATCH|DELETE)\s+(/(?:api/v1|health)[^\s`\"')\]?#]*)"
)
# Paths that appear in the recorded curl session as "$BASE<path>".
_CURL_REFERENCE = re.compile(r"\$BASE(/[^\s`\"'?]+)")
# Markdown link targets: ``](target)``.
_LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")

# The only non-anchor link targets an end-user document served by the
# API may use: things its readers can reach on the same origin.
_ALLOWED_LINK_PREFIXES = (
    "/docs",
    "/redoc",
    "/openapi.json",
    "/manpage",
    "/health",
)


def _normalize_path(path: str) -> str:
    """Collapse path parameters so doc citations compare to OpenAPI paths.

    ``{campaign_id}`` (OpenAPI), ``{suggestion_id}`` (manual), and
    ``$CID`` (curl session shell variables) all normalize to ``{}``.
    """
    path = path.split("?", 1)[0]
    segments = ["{}" if segment.startswith(("{", "$")) else segment for segment in path.split("/")]
    return "/".join(segments)


def _openapi_operations() -> set[tuple[str, str]]:
    schema = create_app().openapi()
    return {
        (method.upper(), _normalize_path(path))
        for path, operations in schema["paths"].items()
        for method in operations
    }


class TestManpageRoutes:
    @pytest.mark.asyncio
    async def test_manpage_serves_html_with_title_and_toc(self, api_client) -> None:
        response = await api_client.get("/manpage")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "BO-MCP Operating Manual" in response.text
        assert 'class="toc"' in response.text
        assert response.headers["ETag"]
        assert f"max-age={MANPAGE_CACHE_MAX_AGE_SECONDS}" in response.headers["Cache-Control"]

    @pytest.mark.asyncio
    async def test_manpage_md_serves_exact_packaged_markdown(self, api_client) -> None:
        response = await api_client.get("/manpage.md")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.content == MANPAGE.encode("utf-8")

    @pytest.mark.asyncio
    async def test_manpage_content_negotiation_prefers_markdown(self, api_client) -> None:
        response = await api_client.get("/manpage", headers={"Accept": "text/markdown"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.content == MANPAGE.encode("utf-8")

    @pytest.mark.asyncio
    async def test_manpage_html_wins_browser_style_accept(self, api_client) -> None:
        """A browser Accept header (html listed, markdown absent) gets HTML."""
        response = await api_client.get(
            "/manpage",
            headers={"Accept": "text/html,application/xhtml+xml,*/*;q=0.8"},
        )

        assert response.headers["content-type"].startswith("text/html")

    @pytest.mark.asyncio
    async def test_manpage_etag_revalidation_returns_304(self, api_client) -> None:
        first = await api_client.get("/manpage.md")
        etag = first.headers["ETag"]

        revalidated = await api_client.get("/manpage.md", headers={"If-None-Match": etag})

        assert revalidated.status_code == 304
        assert revalidated.headers["ETag"] == etag

    @pytest.mark.asyncio
    async def test_etags_are_per_representation(self, api_client) -> None:
        """RFC 9110 §8.8.3: HTML and Markdown carry distinct validators."""
        html = await api_client.get("/manpage")
        markdown = await api_client.get("/manpage", headers={"Accept": "text/markdown"})

        assert html.headers["ETag"] != markdown.headers["ETag"]

    @pytest.mark.asyncio
    async def test_html_validator_does_not_revalidate_markdown(self, api_client) -> None:
        """A cache holding HTML must not get a 304 for a Markdown request."""
        html = await api_client.get("/manpage")

        response = await api_client.get(
            "/manpage",
            headers={"Accept": "text/markdown", "If-None-Match": html.headers["ETag"]},
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")

    @pytest.mark.asyncio
    async def test_each_representation_revalidates_with_its_own_etag(self, api_client) -> None:
        for accept in ("text/html", "text/markdown"):
            first = await api_client.get("/manpage", headers={"Accept": accept})
            revalidated = await api_client.get(
                "/manpage",
                headers={"Accept": accept, "If-None-Match": first.headers["ETag"]},
            )
            assert revalidated.status_code == 304, accept

    @pytest.mark.asyncio
    async def test_negotiated_route_sets_vary_accept(self, api_client) -> None:
        """Shared caches must key /manpage entries per Accept header."""
        response = await api_client.get("/manpage")
        assert response.headers["Vary"] == "Accept"

        revalidated = await api_client.get(
            "/manpage", headers={"If-None-Match": response.headers["ETag"]}
        )
        assert revalidated.status_code == 304
        assert revalidated.headers["Vary"] == "Accept"

    @pytest.mark.asyncio
    async def test_manpage_is_public(self, api_client) -> None:
        """No X-API-Key on any request in this module — 200 proves public."""
        for path in ("/manpage", "/manpage.md"):
            response = await api_client.get(path)
            assert response.status_code == 200, f"{path} must not require auth"


class TestManpageContentValidity:
    def test_every_cited_endpoint_exists_in_openapi_schema(self) -> None:
        """Renaming/removing an endpoint without updating the manual fails."""
        operations = _openapi_operations()
        cited = {
            (method, _normalize_path(path)) for method, path in _ENDPOINT_REFERENCE.findall(MANPAGE)
        }
        assert cited, "manual must cite endpoints via the METHOD /path convention"

        missing = sorted(ref for ref in cited if ref not in operations)
        assert not missing, f"manual cites endpoints missing from OpenAPI: {missing}"

    def test_curl_session_paths_exist_in_openapi_schema(self) -> None:
        known_paths = {path for _, path in _openapi_operations()}
        curl_paths = {_normalize_path(path) for path in _CURL_REFERENCE.findall(MANPAGE)}
        assert curl_paths, "manual must contain the recorded curl session"

        missing = sorted(path for path in curl_paths if path not in known_paths)
        assert not missing, f"curl session uses paths missing from OpenAPI: {missing}"

    def test_internal_anchors_resolve_to_heading_slugs(self) -> None:
        slugs = {slugify_heading(title) for _, title in extract_headings(MANPAGE)}
        anchors = {target[1:] for target in _LINK_TARGET.findall(MANPAGE) if target.startswith("#")}
        assert anchors, "manual must use §-style internal cross-references"

        dangling = sorted(anchor for anchor in anchors if anchor not in slugs)
        assert not dangling, f"manual has dangling anchors: {dangling}"

    def test_link_policy_only_anchors_and_service_paths(self) -> None:
        """No repository file paths, no gitignored docs, no external hosts."""
        violations = [
            target
            for target in _LINK_TARGET.findall(MANPAGE)
            if not target.startswith("#") and not target.startswith(_ALLOWED_LINK_PREFIXES)
        ]
        assert not violations, f"manual links outside the service surface: {violations}"


class TestRestOptionParity:
    """Every mutation option the manual promises exists on the REST surface.

    The manual documents ``dry_run`` / ``force`` / ``atomic`` /
    ``continue_on_error`` as transport-neutral. This guard asserts each
    option is (a) actually named in the manual and (b) present in the
    corresponding OpenAPI request schema (body property or query
    parameter), so dropping an option from a schema — or promising one
    the API does not accept — fails here instead of drifting.
    """

    _BODY_OPTIONS: ClassVar[dict[tuple[str, str], set[str]]] = {
        ("post", "/api/v1/campaigns"): {"dry_run"},
        ("post", "/api/v1/campaigns/{campaign_id}/lifecycle"): {"dry_run"},
        ("post", "/api/v1/suggestions/{suggestion_id}/status"): {"dry_run"},
        ("post", "/api/v1/results/{campaign_id}"): {
            "dry_run",
            "force",
            "atomic",
            "continue_on_error",
        },
    }
    _QUERY_OPTIONS: ClassVar[dict[tuple[str, str], set[str]]] = {
        ("post", "/api/v1/suggestions/{campaign_id}/generate"): {"dry_run"},
        ("post", "/api/v1/results/{campaign_id}/upload"): {"dry_run"},
    }

    @staticmethod
    def _resolve_schema(schema: dict, root: dict) -> dict:
        while "$ref" in schema:
            ref_name = schema["$ref"].rsplit("/", 1)[-1]
            schema = root["components"]["schemas"][ref_name]
        return schema

    def test_manual_names_every_guarded_option(self) -> None:
        options = set().union(*self._BODY_OPTIONS.values(), *self._QUERY_OPTIONS.values())
        missing = sorted(option for option in options if f"`{option}" not in MANPAGE)
        assert not missing, f"manual no longer documents guarded options: {missing}"

    def test_body_options_exist_in_request_schemas(self) -> None:
        schema = create_app().openapi()
        for (method, path), options in self._BODY_OPTIONS.items():
            operation = schema["paths"][path][method]
            body = operation["requestBody"]["content"]["application/json"]["schema"]
            properties = self._resolve_schema(body, schema).get("properties", {})
            missing = sorted(options - set(properties))
            assert not missing, f"{method.upper()} {path} request schema lacks {missing}"

    def test_query_options_exist_in_parameters(self) -> None:
        schema = create_app().openapi()
        for (method, path), options in self._QUERY_OPTIONS.items():
            operation = schema["paths"][path][method]
            query_params = {
                parameter["name"]
                for parameter in operation.get("parameters", [])
                if parameter.get("in") == "query"
            }
            missing = sorted(options - query_params)
            assert not missing, f"{method.upper()} {path} lacks query options {missing}"


class TestManpageDiscoverability:
    def test_openapi_external_docs_point_at_manpage(self) -> None:
        schema = create_app().openapi()
        assert schema["externalDocs"]["url"] == "/manpage"
        assert schema["externalDocs"]["description"]

    def test_app_description_links_the_manual(self) -> None:
        assert "/manpage" in create_app().description
