# BO-MCP

Bayesian Optimization as a service via the Model Context Protocol (MCP).

## Architecture

| Package | Description |
| ------- | ----------- |
| `bo-engine` | Core BO algorithms (BoTorch). Defines the `BOBackend` protocol. No dependencies on other packages. |
| `bo-engine-baybe` | BayBE backend. Depends on `bo-engine` for the protocol + types. |
| `bo-mcp-server` | MCP server, operations layer, database. Depends on `bo-engine`; discovers backends via entry points. |
| `bo-mcp-api` | REST API (FastAPI). Depends on `bo-mcp-server`. |

```text
bo-engine  <──  bo-engine-baybe         (both implement BOBackend)
    ^
    |
bo-mcp-server  ──>  get_backend()  ──>  BoTorchBackend | BayBEBackend
    ^
    |
bo-mcp-api
```

The **operations layer** (`bo_mcp_server.operations`) sits between MCP tools and
the backend, handling orchestration (campaign lookup, state transitions, caching,
provenance) so that tools stay thin (~30 lines each).

## Quick Start

### Path 1: Claude Code (recommended)

The repository ships a `.mcp.json` that auto-configures the MCP server.

```bash
cd bo-mcp-ui && claude
```

Claude Code will detect the server and expose all 20 BO tools automatically.

### Path 2: Standalone MCP server

```bash
pip install ./packages/bo-mcp-server   # or: uv pip install ./packages/bo-mcp-server
bo-mcp-server                          # stdio transport (default)
bo-mcp-server --transport sse --port 8001  # network transport
```

Add to any MCP client config (Claude Desktop, custom agent, etc.):

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

### Path 3: BO engine as a library

```bash
pip install ./packages/bo-engine
```

```python
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ObjectiveSpec, OptimizationSpec, ParameterSpec, ParameterType,
)

backend = BoTorchBackend()
spec = OptimizationSpec(
    parameters=[
        ParameterSpec(name="temperature", type=ParameterType.CONTINUOUS, bounds=(50.0, 150.0)),
        ParameterSpec(name="pressure", type=ParameterType.CONTINUOUS, bounds=(1.0, 10.0)),
    ],
    objectives=[ObjectiveSpec(name="yield", minimize=False)],
)
initial_points = backend.generate_initial_design(spec, n_points=5)
```

## Installation

### Prerequisites

- **Python >= 3.13**
- [uv](https://github.com/astral-sh/uv) (recommended package manager)

### Development setup

```bash
git clone <repo-url> && cd bo-mcp-ui
uv sync          # installs all workspace packages in editable mode
```

### Docker

```bash
docker-compose up --build
```

Services: API on `:8000`, MCP SSE on `:8001`. Source is bind-mounted so Python
code changes are picked up without rebuilding. Rebuild only when dependencies
or Dockerfiles change.

## MCP Tools

20 tools grouped by workflow stage:

| Category | Tool | Description |
|----------|------|-------------|
| **Setup** | `bo_create_campaign` | Create a new optimization campaign |
| | `bo_validate_intake` | Validate campaign spec before creation |
| | `bo_list_capabilities` | List supported features and backends |
| **Exploration** | `bo_list_campaigns` | List campaigns with optional filtering |
| | `bo_list_results` | List submitted results for a campaign |
| | `bo_list_suggestions` | List suggestions for a campaign |
| | `bo_compare_campaigns` | Compare 2-10 campaigns side by side |
| | `bo_export_campaign` | Export full campaign data |
| **Optimization** | `bo_generate_suggestions` | Generate next batch of suggestions |
| | `bo_get_suggestion_explanation` | Explain why a suggestion was made |
| | `bo_update_suggestion_status` | Mark suggestion as accepted/rejected/pending |
| **Results** | `bo_submit_results` | Submit experimental results |
| | `bo_upload_results_file` | Upload results from a CSV file |
| **Diagnostics** | `bo_get_diagnostics` | Campaign progress, Pareto front, model health |
| | `bo_health_check` | Server health and connectivity |
| | `bo_batch_get_status` | Status for multiple campaigns in one call |
| **Lifecycle** | `bo_pause_campaign` | Pause a running campaign |
| | `bo_resume_campaign` | Resume a paused campaign |
| | `bo_terminate_campaign` | Terminate a campaign |
| **Transfer** | `bo_discover_transfer_candidates` | Find campaigns suitable for transfer learning |

5 MCP resources: `campaign://{campaign_id}`, `campaigns://list`, `suggestions://{campaign_id}`, `suggestion://{suggestion_id}`, `events://{campaign_id}`.

See [TOOL_SCHEMAS.md](packages/bo-mcp-server/TOOL_SCHEMAS.md) for full input/output schemas.

## Multi-Backend Architecture

The `BOBackend` protocol (`bo_engine.backend`) defines the contract all backends
implement. Methods accept and return plain Python types (no tensors, no DataFrames).

| Backend | Package | Strengths |
|---------|---------|-----------|
| **BoTorchBackend** (default) | `bo-engine` | Full BoTorch/GPyTorch stack: multi-objective, TuRBO, SAASBO, transfer learning, multi-fidelity |
| **BayBEBackend** | `bo-engine-baybe` | BayBE integration with its own search-space and surrogate model abstractions |

### Selecting a backend

1. **Global default** -- set `BO_BACKEND=baybe` (defaults to `botorch`).
2. **Per-campaign** -- set `spec.backend` when creating a campaign.
3. **Entry-point discovery** -- third-party packages can register backends under
   the `bo_mcp.backends` entry-point group.

## Algorithm Selection

The BoTorch backend automatically selects the optimal model and acquisition
function based on your problem characteristics:

| Problem Characteristic | Model | Acquisition | Strategy |
|----------------------|-------|-------------|----------|
| 1 objective, <=20 params | SingleTaskGP | qLogNEI | L-BFGS-B |
| 1 objective, >20 params | SingleTaskGP | qLogNEI | TuRBO |
| 1 objective, >=50 params | SaasFullyBayesianGP | qLogEI | SAASBO |
| 2+ objectives | ModelListGP | qLogNEHVI | L-BFGS-B |
| With fidelity param | MFGP | qMFKG | Cost-aware |
| With prior campaigns | RGPE | qLogNEI | Transfer |
| Categorical params | MixedSingleTaskGP | qLogNEI | Mixed |
| <2*n_params observations | Any | Sobol | Initial design |

No configuration required -- the system inspects parameters, objectives, and
observation count, then chooses accordingly.

## Development

### Testing

```bash
# All fast tests (run on every PR)
uv run pytest -m "not slow and not nightly"

# Per-package
uv run pytest packages/bo-engine/tests -v
uv run pytest packages/bo-engine-baybe/tests -v
uv run pytest packages/bo-mcp-server/tests -v
uv run pytest packages/bo-mcp-api/tests -v

# Slow tests (MCMC-based, main branch only)
uv run pytest -m slow

# Nightly (statistical tests, multiple seeds)
uv run pytest -m nightly
```

### Lint, format, type-check

```bash
uv run ruff check packages/          # lint
uv run ruff check packages/ --fix    # lint + auto-fix
uv run ruff format packages/         # format
uv run ty check                      # type checking
```

### Pre-commit hooks

Configured in `.pre-commit-config.yaml`: **uv-lock**, **ruff**, **ruff-format**, **ty**.

```bash
pre-commit install     # one-time setup
pre-commit run --all   # manual run
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BO_BACKEND` | `botorch` | Default backend (`botorch` or `baybe`) |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/bo_mcp.db` | Database connection string |
| `SQL_ECHO` | `false` | Log SQL queries |
| `BO_MCP_LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `MCP_ALLOWED_HOSTS` | `127.0.0.1:*,localhost:*,[::1]:*,mcp:*` | Allowed Host headers for SSE (comma-separated) |
| `MCP_ALLOWED_ORIGINS` | `http://127.0.0.1:*,http://localhost:*,http://[::1]:*,http://mcp:*` | CORS origins for SSE (comma-separated) |
| `BO_ENGINE_DEVICE` | *(auto)* | Force compute device: `cuda`, `mps`, `cpu` |
| `BO_ENGINE_DISABLE_GPU` | *(unset)* | Set to `1` to disable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | *(system)* | Standard CUDA device selection |

## Troubleshooting

**Python version errors** -- This project requires Python >= 3.13.
Use `uv venv --python 3.13` to create the virtual environment.

**"greenlet" module not found** -- Run `uv sync` to install all dependencies
(greenlet is required for async SQLAlchemy).

**MCP server won't start** -- Verify manually:

```bash
uv run bo-mcp-server --verify
```

**Claude Code can't find the server** -- Ensure `.mcp.json` exists at the repo
root and that `uv` is on your PATH. If using a custom config, verify the `cwd`
is an absolute path.

**SSE transport blocked** -- Check firewall rules for the port (default 8001).
Add extra hostnames via `MCP_ALLOWED_HOSTS` if connecting from containers or
proxies.

**GPU / MPS issues** -- BoTorch requires float64; Apple MPS does not support it.
MPS is not auto-enabled. Force it with `BO_ENGINE_DEVICE=mps` if you accept
float32 precision. CUDA OOM errors trigger automatic CPU fallback.

## Package READMEs

- [bo-engine](packages/bo-engine/README.md)
- [bo-engine-baybe](packages/bo-engine-baybe/README.md)
- [bo-mcp-server](packages/bo-mcp-server/README.md) (includes [AGENT_COOKBOOK.md](packages/bo-mcp-server/AGENT_COOKBOOK.md))
- [bo-mcp-api](packages/bo-mcp-api/README.md)

## License

MIT -- see [LICENSE](LICENSE).
