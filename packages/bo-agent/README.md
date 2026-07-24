# BO Agent

This package runs the BO-MCP conversational agent locally. The agent expects the
repository's database, REST API, and MCP HTTP server to be running first.

## 1. Start the required Docker services

From the repository root, create the shared Docker network if it does not already
exist, then start `db`, `api`, and `mcp`:

```bash
docker network inspect bo-mcp-network >/dev/null 2>&1 \
  || docker network create bo-mcp-network
docker compose up -d --build db api mcp
```

Confirm that all three services are running and that `db`, `api`, and `mcp` are
healthy:

```bash
docker compose ps db api mcp
```

With the default port configuration, the agent uses:

- REST API: `http://127.0.0.1:8000`
- OpenAPI schema: `http://127.0.0.1:8000/openapi.json`
- MCP endpoint (streamable HTTP): `http://127.0.0.1:8001/mcp`

## 2. Configure the agent

Open another terminal and change to this package:

```bash
cd packages/bo-agent
cp .env.example .env
```

Edit `.env` and replace the placeholder values. For the default local Docker
stack, the BO-MCP settings should be:

```env
BO_MCP_API_URL=http://127.0.0.1:8000
BO_MCP_OPENAPI_URL=http://127.0.0.1:8000/openapi.json
BO_MCP_URL=http://127.0.0.1:8001/mcp
BO_MCP_API_KEY=dev-api-key-12345
```

Also provide valid `OPENAI_API_KEY` and `LOGFIRE_TOKEN` values as needed.

Do not commit `.env`; it may contain secrets.

## 3. Run the agent

From the `packages/bo-agent` terminal, run:

```bash
uv run uvicorn agent:app --reload --port 8899
```

Follow the Uvicorn output in that terminal. When it prints the listening URL,
open it in a browser. With the command above, it will normally be:

```text
http://127.0.0.1:8899
```

Keep both Docker and the Uvicorn terminal running while using the agent. Runtime
errors, reload messages, and the final URL are printed in the Uvicorn terminal.

To stop the required Docker services later, run this from the repository root:

```bash
docker compose stop db api mcp
```
