# bo-mcp-server

MCP (Model Context Protocol) server for Bayesian Optimization. Exposes a full BO workflow as MCP tools for use with Claude, other AI agents, or any MCP-compatible client.

## Scope

- 20 MCP tools covering the complete optimization lifecycle (create, suggest, submit, diagnose, compare)
- 5 MCP resources for campaign, suggestion, and audit event inspection
- SQLAlchemy ORM with PostgreSQL and SQLite support
- Protocol-neutral operations layer reusable by REST, CLI, or direct Python import
- Pluggable backend via `BOBackend` protocol (defaults to BoTorch)

## Installation

```bash
# Basic (SQLite, stdio transport)
pip install bo-mcp-server

# With PostgreSQL support
pip install "bo-mcp-server[postgres]"
```

Requires Python >= 3.13.

## Quick Start

### With Claude Code

The `.mcp.json` at the repo root auto-configures Claude Code:

```bash
cd bo-mcp-ui && claude
```

### Standalone

```bash
# stdio transport (for Claude Desktop / agent integration)
bo-mcp-server

# SSE transport (for network access)
bo-mcp-server --transport sse --host 0.0.0.0 --port 8001
```

### Docker

```bash
docker compose up mcp db
```

### Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/bo_mcp.db` | Database connection string |
| `BO_MCP_LOG_LEVEL` | `INFO` | Logging verbosity |
| `MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*,[::1]:*,mcp:*` | DNS rebinding protection (comma-separated) |
| `MCP_ALLOWED_ORIGINS` | `http://127.0.0.1:*,http://localhost:*,http://[::1]:*,http://mcp:*` | CORS origins (comma-separated) |

## MCP Tools (20)

### Campaign Management

- `bo_create_campaign` -- Create a new optimization campaign
- `bo_list_campaigns` -- List campaigns with filtering and pagination
- `bo_validate_intake` -- Dry-run validation of campaign spec (no side effects)
- `bo_pause_campaign` -- Pause a running campaign
- `bo_resume_campaign` -- Resume a paused campaign
- `bo_terminate_campaign` -- Terminate a campaign

### Suggestion & Results

- `bo_generate_suggestions` -- Generate next experiment batch
- `bo_submit_results` -- Submit experimental results
- `bo_list_suggestions` -- List suggestions with status filtering
- `bo_update_suggestion_status` -- Accept, reject, or expire a suggestion
- `bo_list_results` -- List results with pagination
- `bo_export_campaign` -- Export campaign data as CSV
- `bo_upload_results_file` -- Upload results from CSV file

### Diagnostics & Analysis

- `bo_get_diagnostics` -- Model health, convergence, LOO-CV metrics
- `bo_get_suggestion_explanation` -- Why was this suggestion made?
- `bo_compare_campaigns` -- Side-by-side campaign comparison
- `bo_discover_transfer_candidates` -- Find campaigns for transfer learning
- `bo_batch_get_status` -- Batch status for multiple campaigns
- `bo_list_capabilities` -- List backend features and server version
- `bo_health_check` -- Server health and connectivity check

## MCP Resources (5)

- `campaign://{campaign_id}` -- Campaign details (markdown)
- `campaigns://list` -- All campaigns
- `suggestions://{campaign_id}` -- Suggestions for a campaign
- `suggestion://{suggestion_id}` -- Individual suggestion details
- `events://{campaign_id}` -- Audit trail of tool calls

## Architecture

```text
MCP tool (thin wrapper, ~30 lines)
  |
  v
Operation (business logic, protocol-neutral)
  |
  v
BOBackend protocol  -->  BoTorchBackend (default)
  |                      BayBEBackend (optional)
  v
Database (SQLAlchemy async, PostgreSQL or SQLite)
```

## Package Structure

```text
bo_mcp_server/
    server.py             # FastMCP server setup
    cli.py                # CLI entry point
    audit.py              # Audit logging
    cache.py              # Diagnostics cache
    converters.py         # Domain <-> engine type converters
    errors.py             # Structured error codes with recovery actions
    response_formatter.py # Verbosity-based response filtering
    backend.py            # Backend provider (get_backend())
    domain/               # Pydantic domain models
    storage/              # SQLAlchemy ORM + repositories
    operations/           # Protocol-neutral business logic
    tools/                # MCP tool handlers (thin wrappers)
    resources/            # MCP resource handlers
    migrations/           # Alembic database migrations
```

## Development

```bash
# Run tests (excluding slow/postgres/docker)
cd packages/bo-mcp-server
uv run pytest -m "not slow and not postgres and not docker"

# Run with PostgreSQL (via Docker)
docker compose up db
DATABASE_URL=postgresql+asyncpg://bo_user:bo_password@localhost:5432/bo_mcp bo-mcp-server
```
