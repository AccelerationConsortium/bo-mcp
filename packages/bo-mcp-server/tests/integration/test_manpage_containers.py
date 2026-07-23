"""Containerized consumption tests for the operating-manual surface.

Runs against the docker-compose stack (``db`` + ``api`` + ``mcp``),
asserting from *outside* the containers: the HTTP/MCP behavior of the
manual surface through a real network boundary (negotiation, caching
headers, the FastMCP custom route), the renderer dependency
(``markdown-it-py`` + ``mdit-py-plugins``) being importable in the API
image, and the §2.3 transcript actually executing end to end.

Scope note: the compose stack bind-mounts the repository and installs
the packages editable (``docker/dev-entrypoint.sh``), so these tests
canNOT catch a wheel that omits the manual as package data — the
wheel-packaging guard is the unzip-diff step in the ``testing-docker``
CI job (``.github/workflows/code_quality.yaml``).

Follows the ``test_api_endpoints.py`` live-stack pattern.
Run with: pytest -m docker
"""

import os
import re
import subprocess
from pathlib import Path

import httpx
import pytest

pytestmark = [pytest.mark.docker, pytest.mark.integration]

# Overridable so the stack can run on alternate ports when the defaults
# are taken by another local service (e.g. compose started with custom
# BO_MCP_API_PORT / BO_MCP_SSE_PORT) — same convention as
# BO_MCP_API_BASE_URL in test_api_endpoints.py.
API_BASE_URL = os.environ.get("BO_MCP_API_BASE_URL", "http://localhost:8000")
MCP_BASE_URL = os.environ.get("BO_MCP_SSE_BASE_URL", "http://localhost:8001")

# The repository source of the manual — compared against the served
# bytes so the check fails when the packaged copy inside the container
# diverges from (or omits) the checked-in file.
_MANPAGE_SOURCE = (
    Path(__file__).resolve().parents[2] / "src" / "bo_mcp_server" / "docs" / "MANPAGE.md"
)


def _repo_manpage_text() -> str:
    return _MANPAGE_SOURCE.read_text(encoding="utf-8")


class TestManpageOnApiContainer:
    """``/manpage`` on the REST API container; no ``X-API-Key`` anywhere."""

    def test_manpage_html_served_with_title_and_toc(self):
        response = httpx.get(f"{API_BASE_URL}/manpage")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "BO-MCP Operating Manual" in response.text
        assert 'class="toc"' in response.text
        print("✓ /manpage HTML served by api container")

    def test_manpage_md_matches_repo_source(self):
        """Served bytes == checked-in MANPAGE.md (through the bind mount)."""
        response = httpx.get(f"{API_BASE_URL}/manpage.md")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.text, "served manual must not be empty"
        assert response.text == _repo_manpage_text()
        print("✓ /manpage.md matches the repository MANPAGE.md source")

    def test_content_negotiation_through_container_boundary(self):
        response = httpx.get(f"{API_BASE_URL}/manpage", headers={"Accept": "text/markdown"})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.headers["Vary"] == "Accept"
        assert response.text == _repo_manpage_text()
        print("✓ Accept: text/markdown negotiation works through the container")

    def test_etag_revalidation_through_container_boundary(self):
        first = httpx.get(f"{API_BASE_URL}/manpage.md")
        etag = first.headers["ETag"]
        assert etag

        revalidated = httpx.get(f"{API_BASE_URL}/manpage.md", headers={"If-None-Match": etag})

        assert revalidated.status_code == 304
        print("✓ ETag revalidation returns 304 through the container")


class TestManpageOnMcpContainer:
    """The FastMCP custom route serves the same text on the MCP container.

    No reachability skip: the ``docker`` marker means the full compose
    stack (including ``mcp``) is a precondition, so an unreachable MCP
    container is a failure, not an environment quirk — otherwise a
    broken MCP ``/manpage`` surface would pass CI silently.
    """

    def test_mcp_http_surface_serves_the_manual(self):
        response = httpx.get(f"{MCP_BASE_URL}/manpage")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/markdown")
        assert response.text == _repo_manpage_text()
        print("✓ /manpage served by mcp container custom route")


class TestManpageTranscriptExecution:
    """The §2.3 "complete runnable session" actually runs, end to end.

    The fenced bash block is extracted from the checked-in manual and
    executed against the live stack — the document itself is the test
    fixture, so payload or response-shape drift that would break the
    advertised executable example fails CI. The only permitted mutation
    is pointing ``BASE`` at the compose slot's port; everything else
    runs verbatim (with ``set -euo pipefail`` so any failing step, bad
    JSON path, or unset variable aborts the run).
    """

    _API_KEY = "dev-api-key-12345"

    def _extract_transcript(self) -> str:
        manpage = _repo_manpage_text()
        section_match = re.search(r"### 2\.3 .*?\n## ", manpage, flags=re.DOTALL)
        assert section_match, "manual must contain §2.3"
        fence_match = re.search(r"```bash\n(.*?)```", section_match.group(0), flags=re.DOTALL)
        assert fence_match, "§2.3 must contain the runnable bash transcript"
        script = fence_match.group(1)
        script = re.sub(r"^BASE=.*$", f"BASE={API_BASE_URL}", script, count=1, flags=re.MULTILINE)
        return "set -euo pipefail\n" + script

    def test_transcript_runs_end_to_end_with_replay_proofs(self, tmp_path):
        script_path = tmp_path / "manpage_session.sh"
        script_path.write_text(self._extract_transcript(), encoding="utf-8")

        # S603: the "untrusted input" is the repository's own manual —
        # executing it is precisely what this test verifies.
        proc = subprocess.run(  # noqa: S603
            ["/bin/bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=110,
            check=False,
        )
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

        # Both idempotent retries (create 3b, submit 7b) replayed.
        assert proc.stdout.count('"idempotency_replay":true') == 2, proc.stdout

        # No line anchors: curl prints JSON without a trailing newline, so
        # the transcript's echo markers continue the same output line.
        cid_match = re.search(r"campaign: ([0-9a-f-]{36})", proc.stdout)
        sid_match = re.search(r"suggestion: ([0-9a-f-]{36})", proc.stdout)
        assert cid_match, proc.stdout
        assert sid_match, proc.stdout
        campaign_id, suggestion_id = cid_match.group(1), sid_match.group(1)
        # The replays echoed the original ids, not fresh ones.
        assert proc.stdout.count(campaign_id) >= 3

        headers = {"X-API-Key": self._API_KEY}
        campaign = httpx.get(f"{API_BASE_URL}/api/v1/campaigns/{campaign_id}", headers=headers)
        assert campaign.status_code == 200, campaign.text
        assert campaign.json()["status"] == "paused"

        suggestions = httpx.get(f"{API_BASE_URL}/api/v1/suggestions/{campaign_id}", headers=headers)
        assert suggestions.status_code == 200, suggestions.text
        by_id = {row["id"]: row for row in suggestions.json()}
        assert by_id[suggestion_id]["status"] == "completed"
        print("✓ §2.3 transcript executed end to end with replay proofs")
