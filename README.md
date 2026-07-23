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

Claude Code will detect the server and expose all of its BO tools automatically.

### Path 2: Standalone MCP server

```bash
pip install ./packages/bo-mcp-server   # or: uv pip install ./packages/bo-mcp-server
bo-mcp-server                          # stdio transport (default)
bo-mcp-server --transport sse --port 8001  # network transport (binds 127.0.0.1)
```

The SSE transport binds loopback by default; pass `--host 0.0.0.0` explicitly to
expose it on the network. Note that SSE itself is currently **unauthenticated**
and not tenant-isolated (tracked as a known gap) — a non-loopback bind exposes
every tool to anyone who can reach the port, so keep it loopback or front it
with an authenticating proxy.

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

### Large categorical search spaces on BayBE

BayBE enumerates the full Cartesian product of all discrete/categorical
parameters and persists it inside the campaign state. To keep large
combinatorial campaigns usable, the BayBE backend budgets the enumerated
discrete subspace before construction:

- **Below budget** (default: ≈32 MiB estimated serialized state and
  10 000 candidates), the search space builds exactly as before.
- **Above budget**, the discrete subspace is built from a bounded,
  **deterministically subsampled** candidate list (seeded from the spec
  content, so rebuilds regenerate the identical set). Observed and
  pending configurations are always included so duplicate protection
  keeps working. The campaign runs with a prominent warning, and
  `backend="auto"` prefers a backend that does not enumerate the space.

Tuning: `backend_options["baybe"].max_searchspace_state_bytes` /
`max_candidates` override the budget per campaign. Mitigations for very
wide categorical spaces: `parameter_options["baybe"].encoding="INT"`
(one column per parameter instead of one per category), fewer
categories, or `backend="botorch"`.

As defense in depth, the server also refuses to persist any backend
state larger than `MAX_BACKEND_STATE_BYTES` (default 256 MiB — well
below PostgreSQL's 1 GiB protocol limit), returning a typed
`BACKEND_STATE_TOO_LARGE` (E109) envelope with recovery guidance instead
of a dead database connection.

## Development setup

```bash
git clone https://github.com/willigott/bo-mcp && cd bo-mcp
uv sync          # installs all workspace packages in editable mode
```

For a full local development and test environment across all workspace packages:

```bash
uv sync --all-packages --all-groups --extra dev --extra test
```

Avoid using `--all-extras` for the default dev/test setup. It also enables
feature extras such as `bo-engine[shap]` and `bo-engine-baybe[insights]`, which
pull in the SHAP/Numba/llvmlite dependency stack and may be incompatible with
the active Python version.

### Docker

```bash
docker network inspect bo-mcp-network >/dev/null 2>&1 || docker network create bo-mcp-network
docker compose up --build
```

Services: frontend on `127.0.0.1:3001`, API on `127.0.0.1:8000`, MCP SSE on
`127.0.0.1:8001`. Source is bind-mounted so Python code changes are picked up
without rebuilding. Rebuild only when dependencies or Dockerfiles change.

To run alongside gpu4pyscf, use the slot-aware launcher instead of calling
Compose directly:

```bash
./dev-up-bo-mcp 3 up --build
```

The launcher exports `COMPOSE_PROJECT_NAME=bo-mcp-s3`, shifts host ports by
`slot * 100`, and creates/joins `BO_MCP_NETWORK_NAME=akg4pyscf-gpu-s3-net`.
With slot `3`, host access is:

```text
Frontend: http://127.0.0.1:3301
API:      http://127.0.0.1:8300
MCP SSE:  http://127.0.0.1:8301/sse
```

Containers on the shared network can reach BO-MCP by Docker service name:
`api:8000`, `mcp:8001`, `frontend:80`, and `db:5432`.

## API stability

The REST prefix `/api/v1` and the MCP tool contracts carry **no
backward-compatibility guarantee** while this project is pre-1.0 with no
external consumers. Breaking payload changes land directly on `/api/v1`;
the signal for them is the `schema_version` integer in every envelope
response (see the version history at `RESPONSE_SCHEMA_VERSION` in
`bo_mcp_server/response_formatter.py`). Bundled clients (demo scripts,
frontend) are expected to move atomically with the server — there is no
migration window.

## MCP Tools

22 tools grouped by workflow stage:

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
| | `bo_check_progress` | Poll progress of a long-running suggestion generation call |
| **Lifecycle** | `bo_pause_campaign` | Pause a running campaign |
| | `bo_resume_campaign` | Resume a paused campaign |
| | `bo_terminate_campaign` | Terminate a campaign |
| | `bo_reopen_campaign` | Reopen a completed/terminated campaign |
| **Transfer** | `bo_discover_transfer_candidates` | Find campaigns suitable for transfer learning |

8 MCP resources: `campaign://{campaign_id}`, `campaigns://recent`, `campaigns://recent/{filters}`, `campaigns://list`, `campaigns://list/{filters}`, `suggestions://{campaign_id}`, `suggestion://{suggestion_id}`, `events://{campaign_id}`.

See [TOOL_SCHEMAS.md](packages/bo-mcp-server/TOOL_SCHEMAS.md) for full input/output schemas.

## Multi-Backend Architecture

The `BOBackend` protocol (`bo_engine.backend`) defines the contract all backends
implement. Methods accept and return plain Python types (no tensors, no DataFrames).

| Backend | Package | Strengths |
|---------|---------|-----------|
| **BoTorchBackend** (legacy fallback) | `bo-engine` | Full BoTorch/GPyTorch stack: multi-objective, TuRBO, constraints, cost-aware. SAASBO, multi-fidelity and RGPE transfer learning ship as standalone modules (`bo_engine.saasbo`, `bo_engine.multifidelity`, `bo_engine.transfer_learning`) and are not yet routed through campaign suggestions — specs carrying `saasbo_config` / `fidelity_parameter` / `transfer_learning` are rejected with typed errors |
| **BayBEBackend** (default) | `bo-engine-baybe` | BayBE integration with its own search-space and surrogate model abstractions |

### Selecting a backend

1. **Global default** -- set `BO_BACKEND=botorch` to opt into the legacy backend (defaults to `baybe`).
2. **Per-campaign** -- set `spec.backend` when creating a campaign.
3. **Entry-point discovery** -- third-party packages can register backends under
   the `bo_mcp.backends` entry-point group.

### Molecular / substance parameters (BayBE)

BayBE can encode a categorical parameter whose labels are molecules as a
`SubstanceParameter` (SMILES -> cheminformatics descriptors). Declare it via the
BayBE slot of `parameter_options` (`role: "substance"` + a `substance_data`
SMILES map + an optional `substance_encoding`); `backend="auto"` routes such
specs to BayBE automatically (a pinned `backend="botorch"` is rejected -- no
chemistry kernel). See
[bo-engine-baybe/README.md](packages/bo-engine-baybe/README.md#molecular--substance-parameters-baybe-only)
and the runnable
[example payload](docs/examples/substance_solvent_screening.json).

## Algorithm Selection

The BoTorch backend automatically selects the optimal model and acquisition
function based on your problem characteristics:

| Problem Characteristic | Model | Acquisition | Strategy |
|----------------------|-------|-------------|----------|
| 1 objective, <=20 params | SingleTaskGP | qLogNEI | L-BFGS-B |
| 1 objective, >20 params | SingleTaskGP | qLogNEI | TuRBO |
| 2+ objectives | ModelListGP | qLogNEHVI | L-BFGS-B |
| Categorical params | MixedSingleTaskGP | qLogNEI | Mixed |
| <2*n_params observations | Any | Sobol | Initial design |

SAASBO (`SaasFullyBayesianGP`, for 50+ parameters), multi-fidelity
(`qMFKG`) and RGPE transfer learning are implemented as standalone modules
but not dispatched by the campaign pipeline: a spec setting
`saasbo_config`, `fidelity_parameter` or `transfer_learning` fails fast
with a typed error instead of silently downgrading. Drive
`bo_engine.saasbo.generate_saasbo_suggestions` /
`bo_engine.multifidelity.generate_multifidelity_suggestions` /
`bo_engine.transfer_learning.generate_rgpe_suggestions` directly for those
workflows (campaign-level transfer is supported via BayBE's native
`TaskParameter` mechanism).

No configuration required -- the system inspects parameters, objectives, and
observation count, then chooses accordingly.

## Development

### Testing

Each package has its own `tests/conftest.py`; running pytest from the repo
root collides their identically-named `tests.conftest` modules, so always run
per package (as CI does):

```bash
# All fast tests (run on every PR)
uv run pytest packages/bo-engine/tests -v -m "not slow and not nightly"
uv run pytest packages/bo-engine-baybe/tests -v -m "not slow and not nightly"
uv run pytest packages/bo-mcp-server/tests -v -m "not slow and not nightly"
uv run pytest packages/bo-mcp-api/tests -v -m "not slow and not nightly"

# Slow tests (MCMC-based, main branch only)
uv run pytest packages/bo-engine/tests -m slow

# Nightly (statistical tests, multiple seeds)
uv run pytest packages/bo-engine/tests -m nightly
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
| `BO_BACKEND` | `baybe` | Default backend (`baybe` or legacy `botorch`) |
| `DATABASE_URL` | source checkout: `sqlite+aiosqlite:///<project>/data/bo_mcp.db`; installed package: SQLite file in the OS user data directory (via `platformdirs`, e.g. `~/.local/share/bo-mcp-server` on Linux, `~/Library/Application Support/bo-mcp-server` on macOS) | Database connection string; both defaults are absolute, so stdio launches from any directory use the same file |
| `SQL_ECHO` | `false` | Log SQL queries |
| `BO_MCP_LOG_LEVEL` | `INFO` | Logging verbosity |
| `BO_MCP_NETWORK_NAME` | `bo-mcp-network` | External Docker network used by Compose |
| `BO_MCP_FRONTEND_PORT` | `3001` | Host port for the frontend container |
| `BO_MCP_API_PORT` | `8000` | Host port for the API container |
| `BO_MCP_SSE_PORT` | `8001` | Host port for MCP SSE |
| `MCP_ALLOWED_HOSTS` | *(see below)* | Allowed Host headers for SSE |
| `MCP_ALLOWED_ORIGINS` | *(see below)* | CORS origins for SSE |
| `BO_ENGINE_DEVICE` | *(auto)* | Force compute device: `cuda`, `mps`, `cpu` |
| `BO_ENGINE_DISABLE_GPU` | *(unset)* | Set to `1` to disable GPU acceleration |
| `CUDA_VISIBLE_DEVICES` | *(system)* | Standard CUDA device selection |

`MCP_ALLOWED_HOSTS` defaults to `127.0.0.1:*,localhost:*,[::1]:*,mcp:*`.
`MCP_ALLOWED_ORIGINS` defaults to `http://127.0.0.1:*,http://localhost:*,http://[::1]:*,http://mcp:*`.
Both accept comma-separated patterns; set them to extend defaults for proxies or containers.

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
proxies. Remember the server binds `127.0.0.1` unless started with an explicit
`--host`.

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
