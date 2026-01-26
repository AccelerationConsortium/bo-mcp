#!/usr/bin/env python3
"""Shared utilities for MCP protocol demo scripts.

This module provides helpers for connecting to the BO-MCP server via the
MCP protocol and displaying transparent method selection information.
"""

import hashlib
import json
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

# Project root for spawning the MCP server
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Demo user configuration for MCP protocol scripts
DEMO_API_KEY = "demo-mcp-protocol-key"
DEMO_USER_EMAIL = "demo@mcp-protocol.example.com"
DEMO_USER_NAME = "MCP Protocol Demo User"

# Default to SQLite for demos (can be overridden by environment variables)
DEFAULT_DATABASE_URL = f"sqlite+aiosqlite:///{PROJECT_ROOT}/data/mcp_demo.db"


@asynccontextmanager
async def connect_to_mcp_server() -> AsyncIterator[ClientSession]:
    """Connect to the BO-MCP server via stdio transport.

    This spawns the MCP server as a subprocess and communicates via
    stdin/stdout using the MCP protocol.

    The server inherits DATABASE_URL and USE_ALEMBIC environment variables,
    defaulting to SQLite for ease of demo use.

    Yields:
        ClientSession: An active MCP client session for calling tools.
    """
    # Prepare environment for subprocess - inherit current env and add defaults
    env = os.environ.copy()
    if "DATABASE_URL" not in env:
        env["DATABASE_URL"] = DEFAULT_DATABASE_URL
    if "USE_ALEMBIC" not in env:
        env["USE_ALEMBIC"] = "false"

    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "scripts/run_mcp_server.py"],
        cwd=str(PROJECT_ROOT),
        env=env,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def call_tool(
    session: ClientSession,
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Call an MCP tool and parse the JSON response.

    Args:
        session: Active MCP client session.
        name: Name of the tool to call.
        arguments: Arguments to pass to the tool.

    Returns:
        Parsed JSON response from the tool.
    """
    result = await session.call_tool(name, arguments)
    # MCP returns content as a list of content objects (TextContent, ImageContent, etc.)
    if result.content and len(result.content) > 0:
        content = result.content[0]
        # Check if it's a TextContent and extract text
        if isinstance(content, TextContent):
            return json.loads(content.text)
    return {}


def print_header(title: str) -> None:
    """Print a formatted header."""
    print("=" * 60)
    print(title)
    print("=" * 60)


def print_subheader(title: str) -> None:
    """Print a formatted subheader."""
    print("\n" + "-" * 40)
    print(title)
    print("-" * 40)


def print_method_selection(ms: dict[str, Any]) -> None:
    """Display automatic method selection choices with explanations.

    This shows the user exactly what models and acquisition functions
    were automatically selected based on their problem specification.

    Args:
        ms: Method selection dictionary from generate_suggestions.
    """
    print_header("AUTOMATIC MODEL SELECTION")
    print()
    print(f"  Model Type:       {ms.get('model_type', 'N/A')}")
    print(f"  Acquisition:      {ms.get('acquisition_function', 'N/A')}")
    print(f"  Strategy:         {ms.get('optimization_strategy', 'N/A')}")

    transforms = ms.get("input_transforms", [])
    if transforms:
        print(f"  Transforms:       {', '.join(transforms)}")

    print(f"  Confidence:       {ms.get('confidence', 'N/A')}")

    # Show explanation
    explanation = ms.get("explanation", "")
    if explanation:
        print()
        print(f"  Explanation: {explanation}")

    # Show alternatives
    alternatives = ms.get("alternatives", [])
    if alternatives:
        print()
        print("  Alternatives (you could specify these):")
        for alt in alternatives:
            acq = alt.get("acquisition", "")
            reason = alt.get("reason", "")
            print(f"    - {acq}: {reason}")

    # Show warnings
    warnings = ms.get("warnings", [])
    if warnings:
        print()
        print("  Warnings:")
        for w in warnings:
            print(f"    ! {w}")

    print()
    print("=" * 60)


def print_config_summary(config: dict[str, Any]) -> None:
    """Print a summary of the campaign configuration.

    Args:
        config: Campaign configuration dictionary.
    """
    print()
    print("Configuration provided:")
    n_params = len(config.get("parameters", []))
    n_objectives = len(config.get("objectives", []))
    n_constraints = len(config.get("constraints", []))

    print(f"  - Parameters: {n_params}")
    for p in config.get("parameters", []):
        ptype = p.get("type", "unknown")
        pname = p.get("name", "?")
        if ptype == "continuous":
            bounds = p.get("bounds", [])
            print(f"      {pname}: {ptype} {bounds}")
        elif ptype == "categorical":
            cats = p.get("categories", [])
            print(f"      {pname}: {ptype} {cats}")
        elif ptype == "discrete":
            bounds = p.get("bounds", [])
            print(f"      {pname}: {ptype} {bounds}")
        else:
            print(f"      {pname}: {ptype}")

    print(f"  - Objectives: {n_objectives}")
    for o in config.get("objectives", []):
        oname = o.get("name", "?")
        direction = o.get("direction", "minimize")
        print(f"      {oname}: {direction}")

    if n_constraints > 0:
        print(f"  - Constraints: {n_constraints}")
        for c in config.get("constraints", []):
            ctype = c.get("type", "unknown")
            params = c.get("parameters", [])
            value = c.get("value", 0)
            print(f"      {ctype}: {params} = {value}")


def print_iteration_header(iteration: int, phase: str) -> None:
    """Print iteration header.

    Args:
        iteration: Current iteration number.
        phase: Phase description (e.g., "Initial Design", "BO-guided").
    """
    print(f"\n{'─' * 60}")
    print(f"ITERATION {iteration} ({phase})")
    print("─" * 60)


def print_suggestion(
    params: dict[str, Any],
    objectives: dict[str, float] | None = None,
    index: int | None = None,
) -> None:
    """Print a single suggestion with optional objective values.

    Args:
        params: Parameter values dictionary.
        objectives: Optional objective values after evaluation.
        index: Optional suggestion index.
    """
    prefix = f"  [{index}] " if index is not None else "  "
    param_str = ", ".join(
        f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()
    )

    if objectives:
        obj_str = ", ".join(f"{k}={v:.4f}" for k, v in objectives.items())
        print(f"{prefix}{param_str} -> {obj_str}")
    else:
        print(f"{prefix}{param_str}")


def print_diagnostics(diagnostics: dict[str, Any], iteration: int) -> None:
    """Print diagnostic information for the current iteration.

    Args:
        diagnostics: Diagnostics from get_diagnostics tool.
        iteration: Current iteration number.
    """
    print()
    print(f"  Diagnostics (iteration {iteration}):")
    print(f"    Total results: {diagnostics.get('n_results', 0)}")

    best = diagnostics.get("best_value")
    if best is not None:
        print(f"    Best value: {best:.4f}")

    hv = diagnostics.get("hypervolume")
    if hv is not None:
        print(f"    Hypervolume: {hv:.4f}")

    n_pareto = diagnostics.get("n_pareto_points")
    if n_pareto is not None:
        print(f"    Pareto points: {n_pareto}")


def print_final_results(
    best_params: dict[str, Any] | None,
    best_value: float | None,
    total_iterations: int,
    diagnostics: dict[str, Any] | None = None,
) -> None:
    """Print final optimization results.

    Args:
        best_params: Best parameter values found.
        best_value: Best objective value found.
        total_iterations: Total number of iterations run.
        diagnostics: Optional final diagnostics.
    """
    print_header("OPTIMIZATION COMPLETE")
    print()
    print(f"  Total iterations: {total_iterations}")

    if best_value is not None:
        print(f"  Best value: {best_value:.4f}")

    if best_params:
        print("  Best parameters:")
        for k, v in best_params.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    if diagnostics:
        n_results = diagnostics.get("n_results", 0)
        print(f"  Total evaluations: {n_results}")

        hv = diagnostics.get("hypervolume")
        if hv is not None:
            print(f"  Final hypervolume: {hv:.4f}")

        n_pareto = diagnostics.get("n_pareto_points")
        if n_pareto is not None:
            print(f"  Pareto front size: {n_pareto}")


def suppress_stderr():
    """Context manager to suppress stderr output (from MCP server logs)."""
    import os
    from contextlib import contextmanager

    @contextmanager
    def suppressor():
        stderr_fd = sys.stderr.fileno()
        saved_stderr = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, stderr_fd)
            yield
        finally:
            os.dup2(saved_stderr, stderr_fd)
            os.close(devnull)
            os.close(saved_stderr)

    return suppressor()


# =============================================================================
# Demo User and Campaign Helpers
# =============================================================================
#
# These functions handle user creation and campaign management for MCP protocol
# demos. They ensure that the required owner_id and submitted_by parameters
# are properly set.
# =============================================================================


async def get_or_create_demo_user() -> str:
    """Get or create a demo user and return the user ID.

    This function creates a demo user directly via the storage layer
    (not through MCP) since user management is typically done outside
    the optimization workflow.

    Note: This sets up environment variables for the database before importing
    the storage module to ensure consistent database usage with the MCP server.

    Returns:
        str: The UUID of the demo user as a string.
    """
    # Ensure environment is set before importing storage (which reads DATABASE_URL at import)
    if "DATABASE_URL" not in os.environ:
        os.environ["DATABASE_URL"] = DEFAULT_DATABASE_URL
    if "USE_ALEMBIC" not in os.environ:
        os.environ["USE_ALEMBIC"] = "false"

    from bo_mcp_server.domain import User
    from bo_mcp_server.storage import UserRepository, get_session, init_database

    # Initialize database (idempotent)
    await init_database()

    api_key_hash = hashlib.sha256(DEMO_API_KEY.encode()).hexdigest()

    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_email(DEMO_USER_EMAIL)
        if not user:
            user = User(
                name=DEMO_USER_NAME,
                email=DEMO_USER_EMAIL,
                api_key_hash=api_key_hash,
            )
            user = await repo.save(user)
        return str(user.id)


async def create_demo_campaign(
    session: ClientSession,
    config: dict[str, Any],
    owner_id: str,
) -> dict[str, Any]:
    """Create a campaign with proper parameters.

    This is a helper that wraps the create_campaign tool call with the
    correct parameter names.

    Args:
        session: Active MCP client session.
        config: Campaign configuration (parameters, objectives, etc.).
        owner_id: UUID string of the campaign owner.

    Returns:
        Response from create_campaign tool.
    """
    return await call_tool(
        session,
        "create_campaign",
        {"intake_data": config, "owner_id": owner_id},
    )


async def submit_demo_results(
    session: ClientSession,
    campaign_id: str,
    results: list[dict[str, Any]],
    submitted_by: str,
) -> dict[str, Any]:
    """Submit results with proper parameters.

    This is a helper that wraps the submit_results tool call with the
    correct parameter names.

    Args:
        session: Active MCP client session.
        campaign_id: UUID string of the campaign.
        results: List of result dictionaries.
        submitted_by: UUID string of the user submitting results.

    Returns:
        Response from submit_results tool.
    """
    return await call_tool(
        session,
        "submit_results",
        {
            "campaign_id": campaign_id,
            "results": results,
            "submitted_by": submitted_by,
            "source": "api",
        },
    )
