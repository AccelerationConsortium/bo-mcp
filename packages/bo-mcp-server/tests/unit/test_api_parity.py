"""Tests that guard against drift between MCP tools and HTTP endpoints.

These tests ensure:
1. Every business-capability MCP tool has a corresponding HTTP route.
2. The logged/documented tool count matches the actual registered count.

MCP resources (campaign://, campaigns://list, events://) are transport-specific
and explicitly out of parity scope.
"""

from bo_mcp_server.server import create_mcp_server, mcp

# --- Tool-to-HTTP parity mapping ---
#
# Each MCP business tool must have at least one HTTP endpoint that exposes the
# same capability. The mapping is maintained manually: if you add a new MCP
# tool, add it here or to EXEMPT_TOOLS with a comment explaining why.

TOOL_TO_HTTP_ROUTE: dict[str, str] = {
    # Campaign management
    "bo_create_campaign": "POST /api/campaigns",
    "bo_list_campaigns": "POST /api/campaigns/query",
    "bo_validate_intake": "POST /api/campaigns/validate",
    "bo_list_capabilities": "GET /api/capabilities",
    "bo_batch_get_status": "POST /api/campaigns/status/batch",
    "bo_compare_campaigns": "POST /api/campaigns/compare",
    "bo_export_campaign": "GET /api/campaigns/{campaign_id}/export",
    # Campaign lifecycle (three tools -> one endpoint with action param)
    "bo_pause_campaign": "POST /api/campaigns/{campaign_id}/lifecycle",
    "bo_resume_campaign": "POST /api/campaigns/{campaign_id}/lifecycle",
    "bo_terminate_campaign": "POST /api/campaigns/{campaign_id}/lifecycle",
    # Suggestions
    "bo_generate_suggestions": "POST /api/suggestions/{campaign_id}/generate",
    "bo_list_suggestions": "POST /api/suggestions/{campaign_id}/query",
    "bo_get_suggestion_explanation": "GET /api/suggestions/{suggestion_id}/explanation",
    "bo_update_suggestion_status": "POST /api/suggestions/{suggestion_id}/status",
    # Results
    "bo_submit_results": "POST /api/results/{campaign_id}",
    "bo_upload_results_file": "POST /api/results/{campaign_id}/upload",
    "bo_list_results": "POST /api/results/{campaign_id}/query",
    # Diagnostics
    "bo_get_diagnostics": "GET /api/diagnostics/{campaign_id}",
    # Transfer learning
    "bo_discover_transfer_candidates": "POST /api/campaigns/{campaign_id}/transfer-candidates",
}

EXEMPT_TOOLS: dict[str, str] = {
    "bo_health_check": "Infrastructure tool, not a business capability. HTTP has its own /health.",
}


class TestApiParity:
    def test_every_tool_has_http_route_or_exemption(self):
        """Every registered MCP tool must appear in the parity mapping or exemptions."""
        create_mcp_server()
        registered_tools = set(mcp._tool_manager._tools.keys())

        mapped = set(TOOL_TO_HTTP_ROUTE.keys())
        exempt = set(EXEMPT_TOOLS.keys())
        covered = mapped | exempt

        unmapped = registered_tools - covered
        assert unmapped == set(), (
            f"MCP tools without HTTP parity route or exemption: {sorted(unmapped)}. "
            "Add them to TOOL_TO_HTTP_ROUTE or EXEMPT_TOOLS in this test file."
        )

    def test_no_stale_mappings(self):
        """Parity mapping must not reference tools that no longer exist."""
        create_mcp_server()
        registered_tools = set(mcp._tool_manager._tools.keys())

        all_referenced = set(TOOL_TO_HTTP_ROUTE.keys()) | set(EXEMPT_TOOLS.keys())
        stale = all_referenced - registered_tools
        assert stale == set(), (
            f"Parity mapping references non-existent tools: {sorted(stale)}. "
            "Remove them from TOOL_TO_HTTP_ROUTE or EXEMPT_TOOLS."
        )

    def test_tool_count_is_consistent(self):
        """The actual registered tool count must match expectations.

        This catches silent tool registration failures (e.g. import errors
        that are swallowed) and documentation drift.
        """
        create_mcp_server()
        registered_count = len(mcp._tool_manager._tools)

        expected_count = len(TOOL_TO_HTTP_ROUTE) + len(EXEMPT_TOOLS)
        assert registered_count == expected_count, (
            f"Expected {expected_count} tools (mapped + exempt), "
            f"but {registered_count} are registered. "
            "Update TOOL_TO_HTTP_ROUTE or EXEMPT_TOOLS."
        )

    def test_exempt_tools_have_justification(self):
        """Every exempt tool must have a non-empty justification string."""
        for tool_name, justification in EXEMPT_TOOLS.items():
            assert justification.strip(), (
                f"Exempt tool '{tool_name}' has no justification. "
                "Add a comment explaining why it doesn't need an HTTP route."
            )
