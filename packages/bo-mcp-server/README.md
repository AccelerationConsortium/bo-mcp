# bo-mcp-server

MCP (Model Context Protocol) server for Bayesian Optimization. Use with Claude or any MCP-compatible client.

## Installation

```bash
pip install bo-mcp-server
```

## Quick Start

### With Claude Desktop

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "bayesian-optimization": {
      "command": "bo-mcp-server",
      "args": ["--transport", "stdio"]
    }
  }
}
```

### Standalone

```bash
# Run with stdio transport (for Claude integration)
bo-mcp-server --transport stdio

# Run with SSE transport (for network access)
bo-mcp-server --transport sse --host 0.0.0.0 --port 8001

# Show help
bo-mcp-server --help
```

### CLI Reference

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--transport` | `stdio`, `sse` | `stdio` | Transport protocol |
| `--host` | IP address | `0.0.0.0` | SSE server host |
| `--port` | Integer | `8001` | SSE server port |

## MCP Tools

### Core Tools
- `validate_intake` - Validate campaign configuration
- `create_campaign` - Create new optimization campaign
- `generate_suggestions` - Generate next experiment batch
- `submit_results` - Submit experimental results
- `get_diagnostics` - Get model health and progress metrics

### Data Upload
- `upload_results_file` - Upload results from CSV file

### Explainability
- `get_suggestion_explanation` - Get detailed explanation of why a suggestion was made

### Campaign Lifecycle
- `pause_campaign` - Pause an active campaign
- `resume_campaign` - Resume a paused campaign
- `terminate_campaign` - Permanently terminate a campaign

### Analysis & Strategy
- `compare_campaigns` - Compare 2-10 campaigns for relative performance
- `discover_transfer_candidates` - Auto-discover campaigns for transfer learning

For detailed input/output schemas with all fields documented, see [TOOL_SCHEMAS.md](TOOL_SCHEMAS.md).

## MCP Resources

- `campaign://{id}` - Campaign state and metadata
- `campaigns://list` - List all campaigns
- `suggestions://{campaign_id}` - Pending suggestions for a campaign
- `suggestion://{id}` - Suggestion with provenance

## Package Structure

```
bo_mcp_server/
├── __init__.py           # Package entry point
├── server.py             # FastMCP server setup
├── cli.py                # CLI entry point
├── domain/               # Domain models (Pydantic)
│   ├── campaign.py
│   ├── campaign_spec.py
│   ├── result.py
│   ├── suggestion.py
│   └── user.py
├── storage/              # Database layer (SQLAlchemy)
│   ├── base.py
│   ├── database.py
│   ├── models.py
│   └── repositories.py
├── tools/                # MCP tool implementations
│   ├── create_campaign.py
│   ├── validate_intake.py
│   ├── generate_suggestions.py
│   ├── submit_results.py
│   ├── get_diagnostics.py
│   ├── upload_results_file.py
│   ├── get_suggestion_explanation.py
│   └── campaign_lifecycle.py
└── resources/            # MCP resource implementations
    ├── campaign_resource.py
    └── suggestion_resource.py
```

## Development

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Type checking
pyright src/bo_mcp_server
```
