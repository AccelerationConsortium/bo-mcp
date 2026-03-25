# BO-MCP-UI

Bayesian Optimization MCP Service with Web UI - A multi-objective optimization platform using BoTorch and the Model Context Protocol (MCP).

## Features

### Core Optimization
- **Multi-objective Bayesian Optimization** using BoTorch with qLogNEHVI acquisition
- **Single-objective Bayesian Optimization** (v1.0.1) with qLogNEI/qLogEI acquisition
- **Support for mixed parameter types**: continuous, discrete, and categorical
- **Constraint handling**: sum constraints, linear constraints

### Acquisition Functions
- **qLogNEHVI** - Multi-objective Noisy Expected Hypervolume Improvement (default)
- **qLogNParEGO** (v1.1) - Multi-objective with Chebyshev scalarization
- **qLogNEI** (v1.0.1) - Single-objective Noisy Expected Improvement
- **qLogEI** (v1.0.1) - Single-objective Expected Improvement (noiseless)

### Advanced Features (v1.1)
- **Input Warping** - Kumaraswamy CDF warping for non-stationary objectives
- **LOO Cross-Validation** - Leave-one-out CV for model quality assessment
- **Automatic acquisition selection** - AUTO mode picks the right acquisition function

### Integration & Visualization
- **MCP Server** for direct integration with LLM agents (Claude Code, etc.)
- **REST API** (FastAPI) for web GUI integration
- **React Web UI** for campaign management and visualization
- **Pareto front tracking** with hypervolume progress metrics
- **Interactive visualizations** with Plotly

## Quick Start for AI Agents

### 1. Installation

```bash
git clone <repo-url> && cd bo-mcp-ui && uv sync
```

### 2. Verify Prerequisites

```bash
uv run python scripts/check_prerequisites.py
# Expected: All checks pass - ready to run MCP server
```

### 3. Verify Installation

```bash
uv run bo-mcp-server --verify
# Expected: {"status": "ok", "version": "0.1.0", "tools": 12, "database": "connected"}
```

### 4. Claude Code Configuration

Add to `~/.claude.json` (or Claude Desktop's MCP configuration):

```json
{
  "mcpServers": {
    "bo-mcp": {
      "command": "uv",
      "args": ["run", "bo-mcp-server"],
      "cwd": "/absolute/path/to/bo-mcp-ui"
    }
  }
}
```

**Alternative**: Run via script (equivalent functionality):
```json
{
  "mcpServers": {
    "bo-mcp": {
      "command": "uv",
      "args": ["run", "python", "scripts/run_mcp_server.py"],
      "cwd": "/absolute/path/to/bo-mcp-ui"
    }
  }
}
```

See [Using the MCP Server Standalone](#using-the-mcp-server-standalone) for advanced options and network transport.

## Project Structure

```
bo-mcp/
├── packages/
│   ├── bo-engine/              # BO engine (standalone, no deps on other packages)
│   │   ├── src/bo_engine/
│   │   │   ├── models.py       # GP model creation/fitting
│   │   │   ├── acquisition.py  # qLogNEHVI acquisition function
│   │   │   ├── constraints.py  # Sum/linear constraint handling
│   │   │   ├── suggestions.py  # Suggestion generation
│   │   │   ├── diagnostics.py  # Hypervolume, Pareto front, model health
│   │   │   ├── transforms.py   # Parameter transformations
│   │   │   ├── feature_importance.py
│   │   │   └── types.py        # Internal types (OptimizationSpec, etc.)
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   ├── bo-mcp-server/          # MCP server (depends on bo-engine)
│   │   ├── src/bo_mcp_server/  # Standalone package structure
│   │   │   ├── server.py       # FastMCP server setup
│   │   │   ├── cli.py          # CLI entry point
│   │   │   ├── tools/          # MCP tool implementations
│   │   │   ├── resources/      # MCP resource implementations
│   │   │   ├── storage/        # SQLAlchemy database layer
│   │   │   └── domain/         # Domain models (Campaign, Suggestion, etc.)
│   │   ├── tests/
│   │   └── pyproject.toml
│   │
│   └── bo-mcp-api/             # REST API (depends on bo-mcp-server)
│       ├── src/api/
│       │   ├── main.py         # FastAPI app
│       │   ├── routes/         # API endpoints
│       │   └── schemas/        # Request/response schemas
│       ├── tests/
│       └── pyproject.toml
│
├── apps/
│   └── frontend/               # React + TypeScript + Vite
│       ├── src/
│       ├── package.json
│       └── vite.config.ts
│
├── scripts/                    # Utility scripts
│   ├── create_test_user.py     # Create dev user
│   ├── toy_example.py          # Chemical yield optimization demo
│   ├── mcp_standalone_example.py # Direct MCP usage example
│   └── run_mcp_server.py       # Start MCP server standalone
├── pyproject.toml              # uv workspace root
├── uv.lock
├── docker-compose.yml          # Full stack deployment
└── Dockerfile.api              # API container build
```

### Package Dependencies

```
bo-engine         (no deps on other packages)
     ↑
bo-mcp-server     (depends on bo-engine)
     ↑
bo-mcp-api        (depends on bo-mcp-server)
     ↑
frontend          (calls bo-mcp-api via HTTP)
```

### User Installation Options

> **Note**: These packages are not yet published to PyPI. Install from the local repository after cloning.

```bash
# Clone the repository first
git clone <repo-url>
cd bo-mcp-ui

# Just the BO engine for custom integrations
pip install ./packages/bo-engine

# MCP server for Claude/agent use
pip install ./packages/bo-mcp-server
bo-mcp-server --transport stdio

# Full stack with REST API
pip install ./packages/bo-mcp-api
```

## Quick Start

### Prerequisites

- Python 3.11 or 3.12
- Node.js 18+ (for frontend)
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Option 1: Run with Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repo-url>
cd bo-mcp-ui

# Build and run all services
docker-compose up --build
```

Access the application:
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **MCP SSE**: http://localhost:8001

### Option 2: Run Locally (Development)

#### Backend Setup

```bash
# Create virtual environment with Python 3.11+
uv venv --python 3.11
uv sync

# Install workspace packages in editable mode
uv pip install -e packages/bo-engine -e packages/bo-mcp-server -e packages/bo-mcp-api

# Start the API server
uv run uvicorn api.main:app --reload --port 8000
```

The API will be available at http://localhost:8000

#### Frontend Setup

```bash
cd apps/frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at http://localhost:5173

## Using the Web UI

### Step 1: Set Your API Key

When you first open the UI, you'll be prompted to enter an API key.

For development, use: `dev-api-key-12345`

### Step 2: Create a Campaign

1. Click **"New Campaign"** in the navigation
2. Fill in the campaign details:
   - **Name**: e.g., "Chemical Yield Optimization"
   - **Description**: What you're optimizing
   - **Batch Size**: Number of suggestions per iteration (default: 3)

3. Add **Parameters** (the design variables):
   - **Continuous**: Real numbers within a range (e.g., temperature: 50-150)
   - **Discrete**: Integer values within a range
   - **Categorical**: Choose from a list (e.g., catalyst: Pt, Pd, Rh)

4. Add **Objectives** (what to optimize):
   - Set direction: maximize or minimize
   - Add unit (optional)

5. Add **Constraints** (optional):
   - Sum constraints for mixture parameters

6. Click **"Create Campaign"**

### Step 3: Generate Suggestions

1. On the campaign detail page, click **"Generate New Suggestions"**
2. The system will generate optimized parameter combinations
3. Each suggestion shows the recommended parameter values

### Step 4: Submit Results

1. After running your experiments, click **"Submit Result"** next to each suggestion
2. Enter the measured objective values
3. The results are recorded and used to improve future suggestions

### Step 5: View Progress

- **Pareto Front tab**: See the trade-off between objectives
- **Results tab**: View all submitted results
- **Diagnostics**: Monitor optimization health and hypervolume progress

### Step 6: Iterate

Repeat steps 3-5. The optimizer learns from each result and suggests better experiments.

## Running Examples

### Toy Example: Chemical Yield Optimization

```bash
uv run python scripts/toy_example.py
```

This demonstrates:
1. Creating a campaign with 3 parameters (temperature, pressure, catalyst)
2. Running 5 optimization iterations
3. Tracking Pareto front and hypervolume progress

### MCP Standalone Example

```bash
uv run python scripts/mcp_standalone_example.py
```

This shows how to use MCP tools directly without the REST API.

## Demo Scripts

The project includes comprehensive demo scripts showcasing all BO capabilities. These demos use the actual MCP server protocol and demonstrate how minimal user input leads to intelligent, automatic method selection.

### MCP Protocol Demos (v2.1)

These demos spawn the MCP server and communicate via stdio transport, showing the full production workflow:

| Script | Feature | Command |
|--------|---------|---------|
| `mcp_protocol_simple.py` | Minimal single-objective BO | `uv run python scripts/demos/mcp_protocol_simple.py` |
| `mcp_protocol_multi_objective.py` | Multi-objective Pareto optimization | `uv run python scripts/demos/mcp_protocol_multi_objective.py` |
| `mcp_protocol_high_dim.py` | High-dimensional (25 params) with TuRBO | `uv run python scripts/demos/mcp_protocol_high_dim.py` |
| `mcp_protocol_mixed_params.py` | Mixed parameter types | `uv run python scripts/demos/mcp_protocol_mixed_params.py` |
| `mcp_protocol_constrained.py` | Sum-to-one constraints | `uv run python scripts/demos/mcp_protocol_constrained.py` |

**Key Features Demonstrated:**
- **Automatic method selection**: The system automatically chooses the appropriate model and acquisition function
- **Transparent decisions**: Each demo prints what model/acquisition was selected and why
- **Minimal configuration**: Users only specify parameters and objectives; the rest is handled automatically

### Direct Function Import Demos

These demos import MCP tools directly for testing specific features:

| Script | Feature | Command |
|--------|---------|---------|
| `mcp_single_objective_demo.py` | Single-objective BO | `uv run python scripts/demos/mcp_single_objective_demo.py` |
| `mcp_multi_objective_demo.py` | Multi-objective BO | `uv run python scripts/demos/mcp_multi_objective_demo.py` |
| `mcp_categorical_params_demo.py` | Categorical parameters | `uv run python scripts/demos/mcp_categorical_params_demo.py` |
| `mcp_mixture_constraints_demo.py` | Mixture constraints | `uv run python scripts/demos/mcp_mixture_constraints_demo.py` |
| `mcp_turbo_demo.py` | TuRBO high-dimensional | `uv run python scripts/demos/mcp_turbo_demo.py` |
| `mcp_outcome_constraints_demo.py` | Outcome constraints | `uv run python scripts/demos/mcp_outcome_constraints_demo.py` |
| `mcp_cost_aware_demo.py` | Cost-aware BO (EIpu) | `uv run python scripts/demos/mcp_cost_aware_demo.py` |
| `mcp_multifidelity_demo.py` | Multi-fidelity (qMFKG) | `uv run python scripts/demos/mcp_multifidelity_demo.py` |
| `mcp_transfer_learning_demo.py` | Transfer learning (RGPE) | `uv run python scripts/demos/mcp_transfer_learning_demo.py` |
| `mcp_saasbo_demo.py` | SAASBO high-dimensional | `uv run python scripts/demos/mcp_saasbo_demo.py` |
| `mcp_input_warping_demo.py` | Input warping | `uv run python scripts/demos/mcp_input_warping_demo.py` |
| `mcp_qlogparego_demo.py` | qLogNParEGO | `uv run python scripts/demos/mcp_qlogparego_demo.py` |

### Adaptive BO Demo (v2.2) - NEW

The adaptive BO demo showcases intelligent Bayesian Optimization with automatic method selection, SQLite tracking, and interactive Plotly visualizations.

```bash
# Simple 2D Branin optimization
uv run python scripts/demos/adaptive_bo_demo.py

# 6D Hartmann function
uv run python scripts/demos/adaptive_bo_demo.py --benchmark hartmann6

# Scalable Levy function in 10D
uv run python scripts/demos/adaptive_bo_demo.py --benchmark levy --dims 10

# More iterations with larger batches
uv run python scripts/demos/adaptive_bo_demo.py --iterations 20 --batch-size 5
```

**Key Features:**
- **Minimal configuration**: Just select a benchmark function
- **Automatic method selection**: System adapts model/acquisition based on data
- **SQLite tracking**: All inputs, results, and model performance stored in `data/adaptive_bo_demo.db`
- **Batch BO**: Configurable batch size for parallel evaluations
- **Interactive visualizations**: Plotly dashboard generated at `data/plots/<run_id>/dashboard.html`

**Available Benchmarks:**
- Single-objective: `branin`, `hartmann6`, `levy`, `ackley`, `rosenbrock`
- Multi-objective: `branin_currin`, `zdt1`

### Advanced Feature Demos

| Script | Feature | Command |
|--------|---------|---------|
| `multifidelity_mcp_demo.py` | Multi-fidelity BO with cost tracking | `uv run python scripts/multifidelity_mcp_demo.py` |
| `transfer_learning_mcp_demo.py` | Transfer learning across campaigns | `uv run python scripts/transfer_learning_mcp_demo.py` |
| `saasbo_mcp_demo.py` | High-dimensional with sparsity | `uv run python scripts/saasbo_mcp_demo.py` |

### Method Selection Decision Matrix

The system automatically selects the optimal BO method based on your problem:

| Problem Characteristic | Model | Acquisition | Strategy |
|----------------------|-------|-------------|----------|
| 1 objective, ≤20 params | SingleTaskGP | qLogNEI | L-BFGS-B |
| 1 objective, >20 params | SingleTaskGP | qLogNEI | TuRBO |
| 1 objective, ≥50 params | SaasFullyBayesianGP | qLogEI | SAASBO |
| 2+ objectives | ModelListGP | qLogNEHVI | L-BFGS-B |
| With fidelity param | MFGP | qMFKG | Cost-aware |
| With prior campaigns | RGPE | qLogNEI | Transfer |
| Categorical params | MixedSingleTaskGP | qLogNEI | Mixed |
| <2×n_params observations | Any | Sobol | Initial design |

## API Usage

### Create a Campaign via API

```python
import httpx

BASE_URL = "http://localhost:8000/api"
headers = {"X-API-Key": "dev-api-key-12345"}

campaign_data = {
    "intake": {
        "name": "Chemical Yield Optimization",
        "description": "Optimize yield while minimizing cost",
        "parameters": [
            {"name": "temperature", "type": "continuous", "bounds": [50.0, 150.0]},
            {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
            {"name": "catalyst", "type": "categorical", "categories": ["Pt", "Pd", "Rh"]}
        ],
        "objectives": [
            {"name": "yield", "direction": "maximize", "unit": "%"},
            {"name": "cost", "direction": "minimize", "unit": "USD"}
        ],
        "batch_size": 3
    }
}

response = httpx.post(f"{BASE_URL}/campaigns", json=campaign_data, headers=headers)
campaign_id = response.json()["campaign_id"]
```

### Generate Suggestions

```python
response = httpx.post(f"{BASE_URL}/suggestions/{campaign_id}/generate", headers=headers)
suggestions = response.json()["suggestions"]
```

### Submit Results

```python
results_data = {
    "results": [
        {
            "parameter_values": {"temperature": 100.0, "pressure": 5.0, "catalyst": "Pd"},
            "objective_values": {"yield": 92.5, "cost": 180.0}
        }
    ],
    "source": "api"
}
response = httpx.post(f"{BASE_URL}/results/{campaign_id}", json=results_data, headers=headers)
```

### Get Diagnostics

```python
response = httpx.get(f"{BASE_URL}/diagnostics/{campaign_id}", headers=headers)
diagnostics = response.json()
print(f"Hypervolume: {diagnostics['hypervolume']}")
print(f"Pareto points: {diagnostics['n_pareto_points']}")
```

## Using the MCP Server Standalone

The BO-MCP system is designed with MCP as the primary interface. You can use it directly without the REST API or web frontend.

### Run as MCP Server (for Claude Code / LLM Agents)

```bash
# For Claude Code (stdio transport):
uv run python scripts/run_mcp_server.py

# For network access (SSE transport):
uv run python scripts/run_mcp_server.py --transport sse --host 0.0.0.0 --port 8001
```

### Configuring Claude Code

Add to your Claude Code MCP configuration file. The location depends on your setup:
- **Claude Code CLI**: `~/.claude.json` (in the `mcpServers` section)
- **Claude Desktop**: See [MCP documentation](https://modelcontextprotocol.io/) for configuration location

```json
{
  "mcpServers": {
    "bo-mcp": {
      "command": "uv",
      "args": ["run", "python", "scripts/run_mcp_server.py"],
      "cwd": "/path/to/bo-mcp-ui"
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|------|-------------|
| `health_check` | Check MCP server health and connectivity |
| `create_campaign` | Create a new optimization campaign |
| `list_campaigns` | List campaigns with optional filtering |
| `generate_suggestions` | Generate next batch of experiment suggestions |
| `submit_results` | Submit experimental results |
| `upload_results_file` | Upload results from CSV file |
| `get_diagnostics` | Get campaign progress, Pareto front, and health status |
| `get_suggestion_explanation` | Get detailed explanation of why a suggestion was made |
| `manage_campaign_lifecycle` | Pause, resume, or terminate a campaign |
| `compare_campaigns` | Compare 2-10 campaigns for relative performance |
| `discover_transfer_candidates` | Auto-discover campaigns for transfer learning |
| `batch_get_status` | Get status for multiple campaigns in one call |

### Available MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `campaign://{id}` | Get campaign details |
| `campaigns://list` | List all campaigns |
| `suggestions://{campaign_id}` | Get pending suggestions for a campaign |
| `suggestion://{id}` | Get specific suggestion details |

### Connecting a Custom Agent (Python)

If you're building your own LLM agent or automation that needs to use the BO-MCP server, here's how to connect programmatically.

#### Option A: Using the MCP Python SDK (Recommended)

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_optimization():
    # Connect to the MCP server via stdio
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "scripts/run_mcp_server.py"],
        cwd="/path/to/bo-mcp-ui"
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()

            # List available tools
            tools = await session.list_tools()
            print(f"Available tools: {[t.name for t in tools.tools]}")

            # Create a campaign
            result = await session.call_tool(
                "create_campaign",
                arguments={
                    "name": "My Optimization",
                    "description": "Optimizing process parameters",
                    "parameters": [
                        {"name": "temperature", "type": "continuous", "bounds": [50.0, 150.0]},
                        {"name": "pressure", "type": "continuous", "bounds": [1.0, 10.0]},
                        {"name": "catalyst", "type": "categorical", "categories": ["Pt", "Pd", "Rh"]}
                    ],
                    "objectives": [
                        {"name": "yield", "direction": "maximize"},
                        {"name": "cost", "direction": "minimize"}
                    ],
                    "batch_size": 3
                }
            )
            campaign_id = result.content[0].text  # Extract campaign ID from response
            print(f"Created campaign: {campaign_id}")

            # Generate suggestions
            suggestions = await session.call_tool(
                "generate_suggestions",
                arguments={"campaign_id": campaign_id}
            )
            print(f"Suggestions: {suggestions.content[0].text}")

            # Submit results after running experiments
            await session.call_tool(
                "submit_results",
                arguments={
                    "campaign_id": campaign_id,
                    "results": [
                        {
                            "parameter_values": {"temperature": 100.0, "pressure": 5.0, "catalyst": "Pd"},
                            "objective_values": {"yield": 92.5, "cost": 180.0}
                        }
                    ]
                }
            )

            # Get diagnostics
            diagnostics = await session.call_tool(
                "get_diagnostics",
                arguments={"campaign_id": campaign_id}
            )
            print(f"Diagnostics: {diagnostics.content[0].text}")

if __name__ == "__main__":
    asyncio.run(run_optimization())
```

#### Option B: Direct HTTP with SSE Transport

For network-based connections, start the server with SSE transport:

```bash
uv run python scripts/run_mcp_server.py --transport sse --host 0.0.0.0 --port 8001
```

Then connect via HTTP:

```python
import httpx
import json

MCP_URL = "http://localhost:8001"

# Tool calls are sent as JSON-RPC 2.0 requests
def call_tool(tool_name: str, arguments: dict) -> dict:
    response = httpx.post(
        f"{MCP_URL}/mcp/v1/tools/call",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        }
    )
    return response.json()

# Create campaign
result = call_tool("create_campaign", {
    "name": "Network Optimization",
    "parameters": [
        {"name": "x", "type": "continuous", "bounds": [0.0, 1.0]}
    ],
    "objectives": [
        {"name": "y", "direction": "maximize"}
    ]
})
print(result)
```

#### Option B1: Using from PydanticAI

If another PydanticAI agent needs BO-MCP tools over the network, connect to the SSE endpoint with `MCPServerSSE` and pass it as a toolset:

```python
import asyncio

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerSSE

server = MCPServerSSE("http://localhost:8001/sse")

agent = Agent(
    "openai:gpt-5.2",
    toolsets=[server],
)

async def main():
    async with server:
        result = await agent.run("List the available BO tools and create a campaign.")
        print(result.output)

if __name__ == "__main__":
    asyncio.run(main())
```

Use the URL that matches where the calling agent is running:

- **Same host as Docker Compose**: `http://localhost:8001/sse`
- **Another container on the same Compose network**: `http://mcp:8001/sse`
- **Another machine on the network**: `http://<docker-host-ip>:8001/sse`

The `mcp` hostname only resolves inside the same Docker network. External machines must use the Docker host's reachable IP address or DNS name, and port `8001` must be open.

The BO-MCP server allows `localhost` and the Docker Compose service hostname `mcp` by default for SSE Host-header validation, so `http://mcp:8001/sse` works for container-to-container access on the same Compose network.

#### Option C: Using the Server Directly in Python (No MCP Protocol)

For tighter integration, you can import and use the MCP server's internal functions directly:

```python
import asyncio
from bo_mcp_server.storage import init_database
from bo_mcp_server.tools.create_campaign import create_campaign
from bo_mcp_server.tools.generate_suggestions import generate_suggestions
from bo_mcp_server.tools.submit_results import submit_results
from bo_mcp_server.tools.get_diagnostics import get_diagnostics

async def main():
    # Initialize database
    await init_database()

    # Create campaign (returns campaign ID)
    result = await create_campaign(
        name="Direct Integration",
        description="Using server functions directly",
        parameters=[
            {"name": "x", "type": "continuous", "bounds": [0.0, 10.0]},
            {"name": "y", "type": "continuous", "bounds": [0.0, 10.0]}
        ],
        objectives=[
            {"name": "f", "direction": "minimize"}
        ],
        batch_size=2
    )
    campaign_id = result["campaign_id"]
    print(f"Campaign ID: {campaign_id}")

    # Generate suggestions
    suggestions = await generate_suggestions(campaign_id=campaign_id)
    print(f"Suggestions: {suggestions}")

    # Submit results
    await submit_results(
        campaign_id=campaign_id,
        results=[
            {"parameter_values": {"x": 5.0, "y": 5.0}, "objective_values": {"f": 25.0}}
        ]
    )

    # Get diagnostics
    diag = await get_diagnostics(campaign_id=campaign_id)
    print(f"Best value: {diag.get('best_objectives')}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Full Optimization Workflow Example

Here's a complete example showing the typical optimization loop:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json

async def optimization_loop(n_iterations: int = 5):
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "scripts/run_mcp_server.py"],
        cwd="/path/to/bo-mcp-ui"
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # 1. Create campaign
            result = await session.call_tool(
                "create_campaign",
                arguments={
                    "name": "Process Optimization",
                    "parameters": [
                        {"name": "temp", "type": "continuous", "bounds": [20.0, 100.0]},
                        {"name": "time", "type": "discrete", "bounds": [1, 60]}
                    ],
                    "objectives": [
                        {"name": "quality", "direction": "maximize"}
                    ],
                    "batch_size": 2
                }
            )
            campaign_id = json.loads(result.content[0].text)["campaign_id"]

            # 2. Optimization loop
            for iteration in range(n_iterations):
                print(f"\n--- Iteration {iteration + 1} ---")

                # Generate suggestions
                suggestions_result = await session.call_tool(
                    "generate_suggestions",
                    arguments={"campaign_id": campaign_id}
                )
                suggestions = json.loads(suggestions_result.content[0].text)["suggestions"]

                # Simulate running experiments (replace with actual experiments)
                results = []
                for sugg in suggestions:
                    params = sugg["parameter_values"]
                    # Your experiment code here - this is just a simulation
                    quality = 100 - (params["temp"] - 60)**2 / 100 - (params["time"] - 30)**2 / 50
                    results.append({
                        "parameter_values": params,
                        "objective_values": {"quality": quality}
                    })

                # Submit results
                await session.call_tool(
                    "submit_results",
                    arguments={
                        "campaign_id": campaign_id,
                        "results": results
                    }
                )

                # Check progress
                diag_result = await session.call_tool(
                    "get_diagnostics",
                    arguments={"campaign_id": campaign_id}
                )
                diagnostics = json.loads(diag_result.content[0].text)
                print(f"Best so far: {diagnostics.get('best_objectives')}")
                print(f"Pareto points: {diagnostics.get('n_pareto_points')}")

if __name__ == "__main__":
    asyncio.run(optimization_loop())
```

### MCP Tool Schemas

For reference, here are the detailed schemas for each MCP tool:

#### create_campaign
Creates a new optimization campaign.
```json
{
  "name": "string (required)",
  "description": "string (optional)",
  "parameters": [...],
  "objectives": [...],
  "constraints": [...],
  "batch_size": 3
}
```

#### generate_suggestions
Generates the next batch of experiment suggestions.
```json
{
  "campaign_id": "uuid-string"
}
```

#### submit_results
Submits experimental results.
```json
{
  "campaign_id": "uuid-string",
  "results": [
    {
      "parameter_values": {"param1": value1, "param2": value2},
      "objective_values": {"obj1": value1, "obj2": value2}
    }
  ]
}
```

#### get_diagnostics
Returns campaign progress and health metrics.
```json
{
  "campaign_id": "uuid-string"
}
```

Response includes:
- `hypervolume`: Current hypervolume indicator
- `hypervolume_history`: Progress over iterations
- `pareto_front`: Current Pareto-optimal points
- `n_pareto_points`: Number of Pareto points
- `n_results`: Total results submitted
- `model_health`: GP model diagnostics

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific package tests
uv run pytest packages/bo-engine/tests -v
uv run pytest packages/bo-mcp-server/tests -v
```

### Type Checking

```bash
uv run pyright packages/*/src
```

### Linting

```bash
# Check
uv run ruff check packages/

# Auto-fix
uv run ruff check packages/ --fix

# Format
uv run ruff format packages/
```

### Frontend Linting

```bash
cd apps/frontend
npm run lint
```

## Environment Variables

### Backend

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/bo_mcp.db` | Database connection string |
| `SQL_ECHO` | `false` | Enable SQL query logging |

### Frontend

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_URL` | `/api` | Backend API URL |

## GPU Acceleration

The BO engine automatically detects and uses CUDA GPUs when available, with zero configuration required.

### Design Principles

1. **Zero-config auto-detection**: CUDA GPU usage is automatic
2. **Graceful fallback**: If GPU unavailable, continues on CPU silently
3. **Numerical stability**: Uses float64 (required by BoTorch for GP precision)
4. **MPS opt-in only**: Apple Silicon MPS is NOT auto-enabled due to float64 incompatibility

### GPU Environment Variables

| Variable | Description | Values |
|----------|-------------|--------|
| `BO_ENGINE_DEVICE` | Force specific device | `cuda`, `mps`, `cpu` |
| `BO_ENGINE_DISABLE_GPU` | Disable GPU acceleration | `1`, `true` |
| `CUDA_VISIBLE_DEVICES` | Standard CUDA control | Device indices |

### Usage

```python
from bo_engine import get_device, get_device_info, clear_cache

# Check current device
print(get_device())  # cuda, cpu (mps only if forced)

# Get detailed device info
print(get_device_info())

# Clear GPU memory cache
clear_cache()
```

### Notes

- **MPS (Apple Silicon)**: NOT auto-enabled because BoTorch requires float64 for numerical stability, which MPS doesn't support. Use `BO_ENGINE_DEVICE=mps` to force MPS if you accept float32 precision.
- **CUDA OOM**: The engine automatically catches out-of-memory errors, clears cache, and retries on CPU.

## Troubleshooting

### "Unauthorized" errors
- Make sure you've set your API key in the UI (click "Set API Key" in the nav)
- For development, use: `dev-api-key-12345`

### Frontend can't connect to backend
- Ensure the backend is running on port 8000
- Check that the Vite proxy is configured correctly
- In Docker, ensure the `depends_on` health check passes

### "greenlet" module not found
- Run `uv sync` to install all dependencies
- This is required for async SQLAlchemy

### Python version issues
- This project requires Python 3.11 or 3.12
- Use `uv venv --python 3.11` to create the virtual environment

### Pareto chart not showing
- Submit at least 2-3 results first
- Ensure objectives have different values

### MCP Connection Issues

#### "Tool not found" errors
- Ensure database is initialized: the server creates tables on first run
- Check that `scripts/run_mcp_server.py` runs without errors standalone:
  ```bash
  uv run python scripts/run_mcp_server.py
  ```

#### Server crashes on startup
- Check Python version: `python --version` (need 3.11+)
- Ensure dependencies are installed: `uv sync`
- Check for missing environment variables in error output

#### Claude Code can't find the server
- Verify `cwd` path in config is an absolute path and correct
- Test manually first:
  ```bash
  cd /path/to/bo-mcp-ui && uv run python scripts/run_mcp_server.py
  ```
- Check that `uv` is available in your PATH

#### SSE transport not working
- Ensure no firewall is blocking the port (default: 8001)
- Try with explicit host: `--host 0.0.0.0 --port 8001`

## License

[Add license information]
