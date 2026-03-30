# MCP Demo Scripts

This directory contains demonstration scripts for the BO-MCP server.

## Two Approaches

### 1. MCP Protocol Demos (Recommended)

These demos spawn the MCP server as a subprocess and communicate via the actual
MCP protocol (stdio transport). This is the recommended approach as it demonstrates
the real integration pattern for MCP clients.

**Features:**
- Minimal user input required
- Transparent model/acquisition selection
- Shows exactly what choices the system makes

```bash
uv run python scripts/demos/mcp_protocol_simple.py
```

### 2. Direct Function Import Demos

These demos import Python functions directly, bypassing the MCP protocol layer.
Useful for understanding the internal APIs but not representative of real MCP usage.

```bash
uv run python scripts/demos/mcp_single_objective_demo.py
```

---

## MCP Protocol Demos (v2.1)

| Script | Feature | Description |
|--------|---------|-------------|
| `mcp_protocol_simple.py` | Simplest BO | Minimal input, auto model/acquisition |
| `mcp_protocol_multi_objective.py` | Multi-objective | Auto-selects ModelListGP + qLogNEHVI |
| `mcp_protocol_high_dim.py` | High-dimensional | Auto-activates TuRBO (25 params) |
| `mcp_protocol_mixed_params.py` | Mixed parameters | Continuous, discrete, categorical |
| `mcp_protocol_constrained.py` | Constraints | Sum-to-one mixture constraint |

### Shared Utilities

`mcp_client_utils.py` provides:
- `connect_to_mcp_server()` - Spawn and connect to MCP server
- `call_tool()` - Call MCP tools with JSON response parsing
- `print_method_selection()` - Display transparent model selection

---

## Direct Function Import Demos

### Basic Features

| Script | Feature | Description |
|--------|---------|-------------|
| `mcp_single_objective_demo.py` | Single-objective BO | Basic optimization with qLogNEI acquisition |
| `mcp_multi_objective_demo.py` | Multi-objective BO | Pareto front discovery with qLogNEHVI |
| `mcp_categorical_params_demo.py` | Mixed parameters | Continuous, discrete, and categorical parameters |
| `mcp_mixture_constraints_demo.py` | Sum constraints | Mixture formulations (components sum to 1) |

### Advanced Features (v1.1+)

| Script | Feature | Description |
|--------|---------|-------------|
| `mcp_turbo_demo.py` | TuRBO | Trust region BO for high dimensions (20+ params) |
| `mcp_outcome_constraints_demo.py` | Outcome constraints | Constraints learned from data |
| `mcp_cost_aware_demo.py` | Cost-aware BO | EIpu acquisition for varying costs |
| `mcp_input_warping_demo.py` | Input warping | Kumaraswamy transforms for non-stationarity |
| `mcp_qlogparego_demo.py` | qLogNParEGO | Alternative multi-objective acquisition |

### V2.0 Features

| Script | Feature | Description |
|--------|---------|-------------|
| `mcp_multifidelity_demo.py` | Multi-fidelity BO | qMFKG for cheap/expensive evaluations |
| `mcp_transfer_learning_demo.py` | Transfer learning | RGPE from prior campaigns |
| `mcp_saasbo_demo.py` | SAASBO | Sparse priors for 50+ dimensions |

## Feature Summary

### MCP Tools Used

Each demo uses the following MCP tools:
- `create_campaign` - Create optimization campaign
- `generate_suggestions` - Get next experimental designs
- `submit_results` - Record experimental outcomes
- `get_diagnostics` - Check optimization progress

### Campaign Configuration Options

| Option | Values | Description |
|--------|--------|-------------|
| `acquisition_method` | `auto`, `qLogNEI`, `qLogNEHVI`, `qLogNParEGO`, `EIpu`, `qMFKG`, `SAASBO` | Acquisition function |
| `use_turbo` | `true/false` | Enable TuRBO for high dimensions |
| `use_saasbo` | `true/false` | Enable SAASBO for sparse problems |
| `use_input_warping` | `true/false` | Enable input transforms |
| `use_cost_aware` | `true/false` | Enable cost-aware acquisition |
| `fidelity_parameter` | Object | Multi-fidelity configuration |
| `transfer_learning` | Object | Transfer from prior campaigns |
| `outcome_constraints` | Array | Learned feasibility constraints |
| `constraints` | Array | Hard constraints (sum, bounds) |

## Example Usage (MCP Protocol)

```python
import asyncio
from mcp_client_utils import connect_to_mcp_server, call_tool

async def optimize():
    config = {
        "name": "My Optimization",
        "parameters": [{"name": "x", "type": "continuous", "bounds": [0, 1]}],
        "objectives": [{"name": "y", "direction": "minimize"}],
    }

    async with connect_to_mcp_server() as session:
        # Create campaign
        result = await call_tool(session, "create_campaign", {"config": config})
        campaign_id = result["campaign_id"]

        # Get suggestions (shows transparent method selection)
        suggestions = await call_tool(
            session, "generate_suggestions", {"campaign_id": campaign_id}
        )
        print(f"Model: {suggestions['method_selection']['model_type']}")
        print(f"Acquisition: {suggestions['method_selection']['acquisition_function']}")

asyncio.run(optimize())
```

## Requirements

- Python 3.11+
- Dependencies installed via `uv sync`
- Database initialized (handled automatically by demos)
