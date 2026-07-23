"""Drift-control tests for the ``docs://manpage`` MCP surface.

Guards that keep the MCP manual surface and the tool registry in sync:

- the resource is registered and returns exactly the canonical text the
  shared accessor loads (the same bytes REST serves at ``/manpage.md``);
- the server ``instructions`` tell agents to read it;
- every REST endpoint named in the manual's §6 endpoint map either
  names a *registered* MCP tool alongside it or explicitly declares it
  has no MCP twin — so adding/renaming a tool without updating the map
  fails here rather than drifting silently.

Reference: parity-by-inspection between transports follows the existing
``test_parity_routes.py`` / ``test_api_parity.py`` pattern in this
repository.
"""

import re
from typing import Any

import pytest

from bo_mcp_server.docs import load_manpage_markdown
from bo_mcp_server.resources.manpage_resource import MANPAGE_RESOURCE_URI
from bo_mcp_server.server import create_mcp_server

# A §6 endpoint-map table row: cites an /api/v1 path (or /health) in the
# first cell. Rows must name a bo_* tool, a resource URI, or explicitly
# declare "no MCP tool".
_ENDPOINT_ROW = re.compile(r"^\|\s*`(?:GET|POST|PUT|PATCH|DELETE) /(?:api/v1|health)")


def _endpoint_map_section() -> str:
    manpage = load_manpage_markdown()
    match = re.search(r"^## 6\..*", manpage, flags=re.MULTILINE)
    assert match, "manual must contain the §6 endpoint map"
    return manpage[match.start() :]


class TestManpageResource:
    @pytest.mark.asyncio
    async def test_resource_returns_canonical_markdown(self) -> None:
        mcp = create_mcp_server()

        contents = list(await mcp.read_resource(MANPAGE_RESOURCE_URI))

        assert len(contents) == 1
        assert contents[0].mime_type == "text/markdown"
        assert contents[0].content == load_manpage_markdown()

    @pytest.mark.asyncio
    async def test_resource_is_listed(self) -> None:
        mcp = create_mcp_server()

        uris = {str(resource.uri) for resource in await mcp.list_resources()}

        assert MANPAGE_RESOURCE_URI in uris

    def test_server_instructions_name_the_resource(self) -> None:
        mcp = create_mcp_server()

        assert mcp.instructions is not None
        assert MANPAGE_RESOURCE_URI in mcp.instructions


class TestEndpointMapParity:
    @pytest.mark.asyncio
    async def test_every_mapped_endpoint_names_a_registered_tool(self) -> None:
        """§6 rows must carry a real MCP tool name or an explicit opt-out."""
        mcp = create_mcp_server()
        registered = {tool.name for tool in await mcp.list_tools()}

        rows = [line for line in _endpoint_map_section().splitlines() if _ENDPOINT_ROW.match(line)]
        assert rows, "§6 endpoint map must contain endpoint rows"

        for row in rows:
            named_tools = set(re.findall(r"\bbo_\w+", row))
            if not named_tools:
                assert "no MCP tool" in row, (
                    f"endpoint-map row names no MCP tool and no explicit opt-out: {row}"
                )
                continue
            unknown = named_tools - registered
            assert not unknown, f"endpoint map names unregistered tools {unknown}: {row}"

    @pytest.mark.asyncio
    async def test_workflow_tools_all_appear_in_endpoint_map(self) -> None:
        """The map stays complete: campaign-workflow tools must be present.

        ``bo_health_check`` and ``bo_check_progress`` are covered by the
        service rows / MCP-only note; every other registered tool is a
        campaign operation whose REST twin belongs in the §6 tables.
        """
        mcp = create_mcp_server()
        registered = {tool.name for tool in await mcp.list_tools()}
        section = _endpoint_map_section()

        missing = sorted(name for name in registered if name not in section)
        assert not missing, f"registered tools missing from the §6 endpoint map: {missing}"


class TestNextActionCatalog:
    """The manual's next-action table matches the decision code's vocabulary.

    Both recommendation implementations are exercised over a scenario
    grid (the friend-review remedy for hand-copied action prose): every
    action the code can emit must have a row in the manual's §2.2
    table, and every documented action must be reachable — a stale
    manual row or an undocumented new action both fail here.
    """

    @staticmethod
    def _documented_actions() -> set[str]:
        manpage = load_manpage_markdown()
        match = re.search(r"### 2\.2 .*?### 2\.3", manpage, flags=re.DOTALL)
        assert match, "manual must contain the §2.2 loop-skeleton section"
        rows = set(re.findall(r"^\| `([a-z_]+)` \|", match.group(0), flags=re.MULTILINE))
        # The table's header row is literally "| `action` | ...".
        return rows - {"action"}

    @staticmethod
    def _emitted_actions() -> set[str]:
        from bo_mcp_server.domain import CampaignStatus
        from bo_mcp_server.operations.batch_status import _minimal_next_action
        from bo_mcp_server.operations.diagnostics.actions import (
            compute_next_action_recommendation,
        )

        actions: set[str] = set()

        # Batch-status surface: status x pending x results x budget grid.
        for status in CampaignStatus:
            for n_pending in (0, 2):
                for n_results in (0, 5):
                    for iteration, max_iterations in ((0, None), (10, 10)):
                        recommendation = _minimal_next_action(
                            status, n_results, n_pending, iteration, max_iterations
                        )
                        actions.add(recommendation["action"])

        # Diagnostics surface: campaign state x budget x suggestions x
        # diagnostics-signal grid.
        signal_variants: list[dict[str, Any]] = [
            {},
            {"convergence": {"converged": True, "reason": "stable"}},
            {"outliers": {"count": 2}, "n_results": 10},
            {"health_status": "critical"},
            {"health_status": "warning"},
        ]
        for campaign_status in ("created", "running", "paused", "completed", "failed"):
            for iteration, max_iterations in ((0, None), (10, 10)):
                for n_actionable in (0, 3):
                    for signals in signal_variants:
                        diagnostics: dict[str, Any] = {
                            "n_results": 5,
                            "health_status": "healthy",
                            **signals,
                        }
                        compute_next_action_recommendation(
                            diagnostics,
                            n_actionable,
                            campaign_status,
                            iteration,
                            max_iterations,
                        )
                        actions.add(diagnostics["next_action_recommendation"]["action"])

        return actions

    def test_every_emitted_action_is_documented(self) -> None:
        undocumented = sorted(self._emitted_actions() - self._documented_actions())
        assert not undocumented, f"decision code emits undocumented actions: {undocumented}"

    def test_every_documented_action_is_emitted(self) -> None:
        stale = sorted(self._documented_actions() - self._emitted_actions())
        assert not stale, f"manual documents unreachable actions: {stale}"


class TestSuggestionTransitionTable:
    """The manual's §1.4 edge table matches the transition code exactly.

    Manual edges compare against
    :data:`bo_mcp_server.operations.update_suggestion_status.VALID_SOURCE_STATUSES`;
    automatic edges compare against
    :data:`bo_mcp_server.operations.submit_results_pipeline.AUTO_COMPLETE_SOURCE_STATUSES`.
    A widened, narrowed, or re-routed transition fails here until the
    manual's table is updated in the same change.
    """

    @staticmethod
    def _documented_edges() -> set[tuple[str, str, str]]:
        manpage = load_manpage_markdown()
        match = re.search(r"### 1\.4 .*?- `pending`", manpage, flags=re.DOTALL)
        assert match, "manual must contain the §1.4 transition table"
        rows = re.findall(
            r"^\| `([a-z]+)` \| `([a-z]+)` \| (manual|automatic)",
            match.group(0),
            flags=re.MULTILINE,
        )
        assert rows, "§1.4 must document transitions as a table"
        return {(source, target, trigger) for source, target, trigger in rows}

    def test_manual_edges_match_valid_source_statuses(self) -> None:
        from bo_mcp_server.operations.update_suggestion_status import (
            VALID_SOURCE_STATUSES,
        )

        code_edges = {
            (source.value, target.value)
            for target, sources in VALID_SOURCE_STATUSES.items()
            for source in sources
        }
        documented = {
            (source, target)
            for source, target, trigger in self._documented_edges()
            if trigger == "manual"
        }
        assert documented == code_edges, (
            f"documented-only: {sorted(documented - code_edges)}; "
            f"code-only: {sorted(code_edges - documented)}"
        )

    def test_automatic_edges_match_pipeline_sources(self) -> None:
        from bo_mcp_server.operations.submit_results_pipeline import (
            AUTO_COMPLETE_SOURCE_STATUSES,
        )

        code_edges = {(source.value, "completed") for source in AUTO_COMPLETE_SOURCE_STATUSES}
        documented = {
            (source, target)
            for source, target, trigger in self._documented_edges()
            if trigger == "automatic"
        }
        assert documented == code_edges, (
            f"documented-only: {sorted(documented - code_edges)}; "
            f"code-only: {sorted(code_edges - documented)}"
        )
