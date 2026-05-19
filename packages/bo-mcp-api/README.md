# bo-mcp-api

REST API layer for the Bayesian Optimization MCP server. Provides HTTP endpoints for web frontends and scripts that prefer REST over the MCP protocol.

## Scope

- FastAPI-based REST proxy over `bo-mcp-server` operations
- CORS-enabled for browser-based frontends
- OpenAPI documentation via Swagger UI (`/docs`)
- Shares the same operations layer as the MCP server -- identical behavior guaranteed
- Planned for extraction into a separate GUI repository

## Installation

```bash
pip install bo-mcp-api
```

Requires Python >= 3.13 and `bo-mcp-server`.

## Quick Start

```bash
# Start the REST API (default port 8000)
bo-mcp-api

# With custom port
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up api db
```

## API Endpoints

### Campaigns

- `POST /api/campaigns` -- Create campaign
- `GET /api/campaigns` -- List campaigns

### Suggestions

- `POST /api/suggestions/{campaign_id}/generate` -- Generate next batch
- `GET /api/suggestions/{campaign_id}` -- List suggestions

### Results

- `POST /api/results/{campaign_id}` -- Submit results
- `GET /api/results/{campaign_id}` -- List results

### Diagnostics

- `GET /api/diagnostics/{campaign_id}` -- Get diagnostics

### Health

- `GET /health` -- Server health check

Full OpenAPI schema available at `/docs` when the server is running.

## Development

```bash
cd packages/bo-mcp-api
uv run pytest
```
