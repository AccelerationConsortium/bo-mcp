"""The BO-MCP operating manual (`/manpage`) HTTP surface.

Serves the canonical narrative manual shipped as package data in
``bo_mcp_server.docs`` (loaded via the client facade):

- ``GET /manpage`` — rendered HTML by default; the raw Markdown when the
  ``Accept`` header prefers ``text/markdown`` over ``text/html``.
- ``GET /manpage.md`` — always the exact packaged Markdown bytes; the
  agent-friendly representation, zero rendering involved.

Both routes are public (no ``X-API-Key``), matching ``/docs`` / ``/redoc``
/ ``/openapi.json`` / ``/health``: the manual contains no tenant data.
Responses carry a content-hash ``ETag`` and a modest ``Cache-Control`` —
the content only changes on deploy.
"""

from __future__ import annotations

import hashlib
import re
from functools import cache
from html import escape

from fastapi import APIRouter, Request, Response, status
from fastapi.responses import HTMLResponse, PlainTextResponse
from markdown_it import MarkdownIt
from mdit_py_plugins.anchors import anchors_plugin

from bo_mcp_server.client import load_manpage_markdown

# The manual is immutable between deploys, so clients may cache it for a
# few minutes without risking staleness during a rolling deploy window;
# the ETag makes revalidation free afterwards.
MANPAGE_CACHE_MAX_AGE_SECONDS = 300

# Heading levels included in anchor generation and the rendered table of
# contents. H1 is the document title; H2/H3 are the numbered sections the
# manual's §-style cross-references target.
_ANCHOR_MIN_HEADING_LEVEL = 1
_ANCHOR_MAX_HEADING_LEVEL = 3

_MARKDOWN_MEDIA_TYPE = "text/markdown"
_HTML_MEDIA_TYPE = "text/html"

router = APIRouter(tags=["documentation"])


def slugify_heading(title: str) -> str:
    """Deterministic heading → anchor slug used by renderer and CI alike.

    GitHub-compatible slugging (lowercase, drop punctuation, spaces →
    hyphens): ``"5.1 Next-action vs diagnostics"`` →
    ``"51-next-action-vs-diagnostics"``. Matching GitHub's scheme keeps
    the anchors identical whether the manual is read at ``/manpage`` or
    as rendered Markdown elsewhere, and the drift-control test can
    recompute the exact anchor set from the source.
    """
    cleaned = re.sub(r"[^\w\- ]", "", title.strip().lower())
    return re.sub(r" +", "-", cleaned)


def _build_parser() -> MarkdownIt:
    """CommonMark + tables + deterministic heading anchors."""
    parser = MarkdownIt("commonmark").enable("table")
    parser.use(
        anchors_plugin,
        min_level=_ANCHOR_MIN_HEADING_LEVEL,
        max_level=_ANCHOR_MAX_HEADING_LEVEL,
        slug_func=slugify_heading,
    )
    return parser


def extract_headings(markdown_text: str) -> list[tuple[int, str]]:
    """Return ``(level, title)`` for each heading in document order."""
    parser = _build_parser()
    tokens = parser.parse(markdown_text)
    headings: list[tuple[int, str]] = []
    for index, token in enumerate(tokens):
        if token.type == "heading_open":
            level = int(token.tag.removeprefix("h"))
            title = tokens[index + 1].content
            headings.append((level, title))
    return headings


def _render_toc(markdown_text: str) -> str:
    """Nested table-of-contents list over the H2/H3 heading tree."""
    items: list[str] = []
    for level, title in extract_headings(markdown_text):
        if level < 2 or level > _ANCHOR_MAX_HEADING_LEVEL:
            continue
        css_class = f"toc-l{level}"
        items.append(
            f'<li class="{css_class}"><a href="#{slugify_heading(title)}">{escape(title)}</a></li>'
        )
    return '<nav class="toc"><strong>Contents</strong><ul>' + "".join(items) + "</ul></nav>"


# Inline-only styling: no JS, no external assets, readable measure,
# monospace code blocks, and a leading table of contents.
_HTML_STYLE = """
body { margin: 0 auto; max-width: 46rem; padding: 2rem 1.25rem 6rem;
       font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
       line-height: 1.55; color: #1c1e21; background: #ffffff; }
h1, h2, h3 { line-height: 1.25; }
h2 { margin-top: 2.5rem; border-bottom: 1px solid #d0d4da; padding-bottom: .3rem; }
h3 { margin-top: 1.8rem; }
pre, code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: .9em; }
pre { background: #f5f6f8; border: 1px solid #e2e5e9; border-radius: 6px;
      padding: .75rem 1rem; overflow-x: auto; }
code { background: #f5f6f8; border-radius: 4px; padding: .1em .3em; }
pre > code { background: none; border: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; display: block;
        overflow-x: auto; }
th, td { border: 1px solid #d0d4da; padding: .4rem .6rem; text-align: left;
         vertical-align: top; }
th { background: #f5f6f8; }
a { color: #0b57d0; }
.toc { background: #f5f6f8; border: 1px solid #e2e5e9; border-radius: 6px;
       padding: .75rem 1.25rem; margin: 1.5rem 0; }
.toc ul { list-style: none; margin: .5rem 0 0; padding: 0; }
.toc li.toc-l3 { padding-left: 1.25rem; }
.toc a { text-decoration: none; }
"""


@cache
def render_manpage_html() -> str:
    """Render the packaged manual to a self-contained HTML page.

    Cached for the process lifetime like the Markdown source itself.
    """
    markdown_text = load_manpage_markdown()
    body = _build_parser().render(markdown_text)
    toc = _render_toc(markdown_text)
    # Inject the TOC right after the document H1 so the title stays first.
    body_with_toc = re.sub(r"(</h1>)", r"\1" + toc, body, count=1)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>BO-MCP Operating Manual</title>\n"
        f"<style>{_HTML_STYLE}</style>\n"
        f"</head>\n<body>\n{body_with_toc}\n</body>\n</html>\n"
    )


@cache
def _representation_etag(media_type: str) -> str:
    """Strong ETag for one negotiated representation.

    RFC 9110 §8.8.3: an entity tag identifies a *representation*, not
    the underlying resource — the HTML and Markdown renderings of
    ``/manpage`` must therefore carry distinct validators, or a shared
    cache holding one representation could answer a request for the
    other with a 304 and reuse the wrong body.
    """
    body = render_manpage_html() if media_type == _HTML_MEDIA_TYPE else load_manpage_markdown()
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    return f'"{digest[:32]}"'


def _cache_headers(media_type: str, *, vary_accept: bool) -> dict[str, str]:
    """Caching headers for one representation.

    ``Vary: Accept`` is set on the negotiated ``/manpage`` route so
    shared caches key entries per Accept header; ``/manpage.md`` has a
    single representation and needs no Vary.
    """
    headers = {
        "ETag": _representation_etag(media_type),
        "Cache-Control": f"public, max-age={MANPAGE_CACHE_MAX_AGE_SECONDS}",
    }
    if vary_accept:
        headers["Vary"] = "Accept"
    return headers


def _matches_etag(request: Request, media_type: str) -> bool:
    """True when ``If-None-Match`` revalidates this representation."""
    header = request.headers.get("if-none-match")
    if header is None:
        return False
    candidates = {candidate.strip() for candidate in header.split(",")}
    return "*" in candidates or _representation_etag(media_type) in candidates


def _entry_quality(params: list[str]) -> float:
    """The ``q=`` value of one Accept entry (1.0 when absent, 0 if bad)."""
    quality = 1.0
    for param in params:
        if param.startswith("q="):
            try:
                quality = float(param[2:])
            except ValueError:
                quality = 0.0
    return quality


def _accept_quality(accept_header: str, media_type: str) -> float:
    """Effective quality the ``Accept`` header assigns to ``media_type``.

    Minimal RFC 9110 semantics: exact match beats ``text/*`` beats
    ``*/*``; unmatched types score 0.
    """
    specificity_by_candidate = {
        media_type: 2,
        media_type.split("/", 1)[0] + "/*": 1,
        "*/*": 0,
    }
    best_specificity = -1
    quality = 0.0
    for entry in accept_header.split(","):
        parts = [part.strip() for part in entry.split(";")]
        specificity = specificity_by_candidate.get(parts[0].lower())
        if specificity is not None and specificity > best_specificity:
            best_specificity = specificity
            quality = _entry_quality(parts[1:])
    return quality


def _prefers_markdown(request: Request) -> bool:
    """Content negotiation for ``/manpage``: Markdown only when preferred.

    HTML wins ties (including the absent-header and ``*/*`` cases) so
    browsers keep getting the rendered page.
    """
    accept_header = request.headers.get("accept", "")
    if not accept_header:
        return False
    markdown_quality = _accept_quality(accept_header, _MARKDOWN_MEDIA_TYPE)
    html_quality = _accept_quality(accept_header, _HTML_MEDIA_TYPE)
    return markdown_quality > html_quality


def _serve_representation(request: Request, media_type: str, *, vary_accept: bool) -> Response:
    """Serve one negotiated representation with its own validator.

    Negotiation has already happened; conditional revalidation runs
    against this representation's ETag only, so a validator obtained
    for the other representation always yields a full 200 body.
    """
    headers = _cache_headers(media_type, vary_accept=vary_accept)
    if _matches_etag(request, media_type):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=headers)
    if media_type == _HTML_MEDIA_TYPE:
        return HTMLResponse(render_manpage_html(), headers=headers)
    return PlainTextResponse(
        load_manpage_markdown(),
        media_type=f"{_MARKDOWN_MEDIA_TYPE}; charset=utf-8",
        headers=headers,
    )


@router.get(
    "/manpage",
    summary="BO-MCP operating manual",
    description=(
        "The canonical narrative manual for operating BO-MCP campaigns: "
        "call order, state ownership, continuation, and error recovery. "
        "Returns rendered HTML by default and raw Markdown when the "
        "Accept header prefers text/markdown. Public — no API key."
    ),
    response_class=HTMLResponse,
)
async def get_manpage(request: Request) -> Response:
    """Serve the operating manual with HTML/Markdown content negotiation."""
    media_type = _MARKDOWN_MEDIA_TYPE if _prefers_markdown(request) else _HTML_MEDIA_TYPE
    return _serve_representation(request, media_type, vary_accept=True)


@router.get(
    "/manpage.md",
    summary="BO-MCP operating manual (raw Markdown)",
    description=(
        "The exact packaged Markdown source of the operating manual — "
        "the agent-friendly representation. Public — no API key."
    ),
    response_class=PlainTextResponse,
)
async def get_manpage_markdown(request: Request) -> Response:
    """Serve the manual's raw Markdown bytes."""
    return _serve_representation(request, _MARKDOWN_MEDIA_TYPE, vary_accept=False)
