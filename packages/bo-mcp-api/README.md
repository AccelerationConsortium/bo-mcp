# bo-mcp-api

REST API proxy for the BO MCP server. Enables web frontend integration.

## Installation

```bash
pip install bo-mcp-api
```

## Quick Start

```bash
bo-mcp-api run --port 8000
```

## API Endpoints

- `POST /campaigns` - Create campaign
- `GET /campaigns` - List campaigns
- `GET /campaigns/{id}` - Get campaign details
- `POST /campaigns/{id}/suggestions` - Generate suggestions
- `POST /campaigns/{id}/results` - Submit results
- `GET /campaigns/{id}/diagnostics` - Get diagnostics
