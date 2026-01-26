"""Search tools meta-tool for MCP.

This tool implements lazy tool loading/discovery pattern from Anthropic's
code execution with MCP best practices. Instead of sending all tool definitions
upfront (~2000 tokens), agents can discover relevant tools on-demand.

Reference: Anthropic - Code Execution with MCP
https://www.anthropic.com/engineering/code-execution-with-mcp
"""

import logging
from typing import Any

from bo_mcp_server.server import mcp

logger = logging.getLogger(__name__)

# Tool metadata registry - short descriptions for search matching
# This is kept separate from the full docstrings to enable efficient search
_TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "create_campaign": {
        "name": "create_campaign",
        "category": "campaign_management",
        "short_description": "Create a new optimization campaign from intake data",
        "keywords": ["create", "new", "campaign", "setup", "initialize", "start"],
        "parameters": ["intake_data", "owner_id"],
    },
    "validate_intake": {
        "name": "validate_intake",
        "category": "campaign_management",
        "short_description": "Validate campaign intake data before creation",
        "keywords": ["validate", "check", "intake", "verify", "configuration"],
        "parameters": ["intake_data"],
    },
    "list_campaigns": {
        "name": "list_campaigns",
        "category": "campaign_management",
        "short_description": "List optimization campaigns with optional filtering",
        "keywords": ["list", "campaigns", "all", "filter", "search", "find"],
        "parameters": ["owner_id", "status", "limit", "verbosity"],
    },
    "generate_suggestions": {
        "name": "generate_suggestions",
        "category": "optimization",
        "short_description": "Generate parameter suggestions using Bayesian optimization",
        "keywords": [
            "suggest",
            "generate",
            "parameters",
            "next",
            "batch",
            "optimize",
            "recommendation",
        ],
        "parameters": ["campaign_id", "batch_size", "verbosity"],
    },
    "submit_results": {
        "name": "submit_results",
        "category": "optimization",
        "short_description": "Submit experimental results for a campaign",
        "keywords": ["submit", "results", "observations", "data", "experiments", "feedback"],
        "parameters": ["campaign_id", "results", "submitted_by", "source", "force", "atomic"],
    },
    "get_diagnostics": {
        "name": "get_diagnostics",
        "category": "monitoring",
        "short_description": "Get diagnostic information and health status for a campaign",
        "keywords": [
            "diagnostics",
            "health",
            "status",
            "progress",
            "metrics",
            "convergence",
            "monitor",
        ],
        "parameters": ["campaign_id", "use_cache", "verbosity"],
    },
    "batch_get_status": {
        "name": "batch_get_status",
        "category": "monitoring",
        "short_description": "Get status of multiple campaigns in one call",
        "keywords": ["batch", "status", "multiple", "campaigns", "dashboard", "overview"],
        "parameters": ["campaign_ids", "verbosity"],
    },
    "manage_campaign_lifecycle": {
        "name": "manage_campaign_lifecycle",
        "category": "lifecycle",
        "short_description": "Manage campaign lifecycle (pause, resume, terminate)",
        "keywords": ["pause", "resume", "terminate", "stop", "lifecycle", "state"],
        "parameters": ["campaign_id", "action"],
    },
    "pause_campaign": {
        "name": "pause_campaign",
        "category": "lifecycle",
        "short_description": "Pause an active campaign",
        "keywords": ["pause", "stop", "hold"],
        "parameters": ["campaign_id"],
    },
    "resume_campaign": {
        "name": "resume_campaign",
        "category": "lifecycle",
        "short_description": "Resume a paused campaign",
        "keywords": ["resume", "continue", "restart"],
        "parameters": ["campaign_id"],
    },
    "terminate_campaign": {
        "name": "terminate_campaign",
        "category": "lifecycle",
        "short_description": "Terminate a campaign permanently",
        "keywords": ["terminate", "end", "finish", "complete", "stop"],
        "parameters": ["campaign_id"],
    },
    "health_check": {
        "name": "health_check",
        "category": "system",
        "short_description": "Check MCP server and database health",
        "keywords": ["health", "check", "status", "ping", "alive", "system"],
        "parameters": [],
    },
    "get_suggestion_explanation": {
        "name": "get_suggestion_explanation",
        "category": "analysis",
        "short_description": "Get detailed explanation of why a suggestion was made",
        "keywords": ["explain", "suggestion", "why", "reasoning", "analysis"],
        "parameters": ["suggestion_id", "verbosity"],
    },
    "compare_campaigns": {
        "name": "compare_campaigns",
        "category": "analysis",
        "short_description": "Compare multiple campaigns for benchmarking",
        "keywords": ["compare", "benchmark", "campaigns", "analysis", "performance"],
        "parameters": ["campaign_ids", "verbosity"],
    },
    "discover_transfer_candidates": {
        "name": "discover_transfer_candidates",
        "category": "advanced",
        "short_description": "Find campaigns suitable for transfer learning",
        "keywords": ["transfer", "learning", "similar", "candidates", "warm-start"],
        "parameters": ["campaign_id", "verbosity"],
    },
    "upload_results_file": {
        "name": "upload_results_file",
        "category": "data",
        "short_description": "Upload results from CSV or JSON file",
        "keywords": ["upload", "file", "csv", "json", "import", "batch"],
        "parameters": ["campaign_id", "file_content", "file_format", "submitted_by"],
    },
}

# Category descriptions for grouping
_CATEGORIES: dict[str, str] = {
    "campaign_management": "Tools for creating and managing optimization campaigns",
    "optimization": "Core Bayesian optimization tools for suggestions and results",
    "monitoring": "Tools for monitoring campaign progress and health",
    "lifecycle": "Tools for managing campaign state transitions",
    "system": "System health and status tools",
    "analysis": "Tools for analyzing and comparing campaigns",
    "advanced": "Advanced features like transfer learning",
    "data": "Tools for bulk data operations",
}


def _match_query(tool_info: dict[str, Any], query: str) -> float:
    """Calculate relevance score for a tool based on query.

    Returns a score between 0 and 1, higher means more relevant.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())

    score = 0.0

    # Exact name match
    if query_lower == tool_info["name"]:
        return 1.0

    # Name contains query
    if query_lower in tool_info["name"]:
        score += 0.5

    # Query contains tool name
    if tool_info["name"] in query_lower:
        score += 0.4

    # Keyword matches
    keywords = set(tool_info.get("keywords", []))
    keyword_matches = len(query_words & keywords)
    if keyword_matches > 0:
        score += 0.3 * min(1.0, keyword_matches / len(query_words))

    # Description contains query words
    description = tool_info.get("short_description", "").lower()
    for word in query_words:
        if len(word) > 2 and word in description:
            score += 0.1

    # Category match
    if query_lower in tool_info.get("category", ""):
        score += 0.2

    return min(1.0, score)


@mcp.tool()
async def search_tools(
    query: str,
    category: str | None = None,
    limit: int = 5,
) -> dict[str, Any]:
    """Search available tools by keyword or category.

    Use this tool to discover relevant tools without loading all tool definitions.
    This reduces context consumption by ~60-80% compared to loading all tools upfront.

    Args:
        query: Search query (tool name, keyword, or description).
            Examples: "create campaign", "suggestions", "health", "compare"
        category: Optional category filter. Categories:
            - "campaign_management": Creating and managing campaigns
            - "optimization": Suggestions and results
            - "monitoring": Diagnostics and health
            - "lifecycle": Pause, resume, terminate
            - "system": Health checks
            - "analysis": Comparison and explanations
            - "advanced": Transfer learning
            - "data": File uploads
        limit: Maximum number of results (default 5, max 10).

    Returns:
        Dictionary with:
            - success: Boolean indicating if search succeeded
            - query: The search query used
            - results: List of matching tools with name, description, and relevance
            - categories: Available categories (for exploration)
            - errors: List of error messages (if any)

    Example:
        # Find tools for creating campaigns
        search_tools(query="create campaign")

        # Find all monitoring tools
        search_tools(query="", category="monitoring")

        # Find tools related to suggestions
        search_tools(query="suggest next parameters")
    """
    logger.info("Searching tools: query='%s', category=%s, limit=%d", query, category, limit)

    # Validate limit
    if limit < 1:
        limit = 1
    elif limit > 10:
        limit = 10

    # Filter by category if provided
    tools_to_search = _TOOL_REGISTRY.values()
    if category is not None:
        category_lower = category.lower()
        if category_lower not in _CATEGORIES:
            valid_categories = list(_CATEGORIES.keys())
            return {
                "success": False,
                "query": query,
                "results": [],
                "categories": _CATEGORIES,
                "errors": [f"Invalid category '{category}'. Valid: {valid_categories}"],
            }
        tools_to_search = [t for t in tools_to_search if t["category"] == category_lower]

    # Score and rank tools
    scored_tools: list[tuple[float, dict[str, Any]]] = []

    for tool_info in tools_to_search:
        if query:
            score = _match_query(tool_info, query)
            if score > 0:
                scored_tools.append((score, tool_info))
        else:
            # If no query, include all tools in category with equal score
            scored_tools.append((0.5, tool_info))

    # Sort by score descending
    scored_tools.sort(key=lambda x: x[0], reverse=True)

    # Limit results
    scored_tools = scored_tools[:limit]

    # Format results
    results = []
    for score, tool_info in scored_tools:
        results.append(
            {
                "name": tool_info["name"],
                "description": tool_info["short_description"],
                "category": tool_info["category"],
                "parameters": tool_info["parameters"],
                "relevance": round(score, 2),
            }
        )

    logger.info("Found %d matching tools for query '%s'", len(results), query)

    return {
        "success": True,
        "query": query,
        "results": results,
        "categories": _CATEGORIES,
        "errors": [],
    }
