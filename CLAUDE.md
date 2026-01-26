# BO-MCP-UI Development Guidelines

<!-- ## Project Architecture

This is a monorepo with three Python packages:

- **bo-engine**: Core Bayesian Optimization engine using PyTorch/BoTorch (no dependencies on other packages from this repository)
- **bo-mcp-server**: MCP (Model Context Protocol) server for AI agent integration (depends on bo-engine)
- **bo-mcp-api**: REST API layer using FastAPI (depends on bo-mcp-server)
- **apps/frontend**: React + TypeScript + Vite web UI

### Key Directories

```
packages/bo-engine/src/bo_engine/       # BO algorithms, GP models, acquisition functions
packages/bo-mcp-server/src/bo_mcp_server/  # MCP server, tools, resources
  ├── tools/                            # MCP tool implementations (create_campaign, etc.)
  ├── resources/                        # MCP resource implementations
  ├── domain/                           # Domain models (Campaign, Suggestion, Result)
  └── storage/                          # SQLAlchemy database layer
packages/bo-mcp-api/src/api/            # FastAPI routes and schemas
```

### Package Import Pattern

Always use the `bo_mcp_server` package (not `mcp_server`):

```python
from bo_mcp_server.domain import Campaign, CampaignSpec
from bo_mcp_server.storage import get_session, init_database
from bo_mcp_server.tools.create_campaign import create_campaign
``` -->

## Development Guidelines

- you act as a senior software architect, senior software engineer and senior software developer
- you further act as a domain expert for Bayesian Optimization who understands the underlying algorithms, including the mathematical foundations, and also have much experience with practical implementations
- when asked to use a specific tool, always look up their documentation and tutorials online
- when writing code, make sure that each function executes one task
- when writing code, make sure that the cognitive load for the reader of the code is minimized
- follow the KISS principle
- follow the DRY principle
- do not hardcode numbers in the code, but provide them via configuration or constant files
- when writing tests, do online research and identify use cases that can be referenced. add the reference to the test in the doc string. the reader should get confidence in the implemented solution.
- when running tests, never skip them if they fail or let them pass trivially but always implement actual functionality that is being tested. make suggestions and a plan how to fix the tests that fail
- when fixing a failed test, always address the root cause.
- when adding new functionality or if you change existing one, add and update tests i.e. keep the functionality of the code in sync with the test suite

## Python Guidelines

- always use uv for package management
- always use formatting (using ruff), linting (using ruff), typing (using pyright) and testing (using pytest) before letting me review your solution
- always use an ORM (SQLAlchemy) when creating databases and when you interact with them
- put all imports at the top of a file, don't put them within functions
- do not use wild card imports ever
- for exception handling, always try to use specific exceptions rather than just "Exception". If you don't use a specific one, justify your decision when you summarize your activities

## Testing Guidelines

See [TESTING.md](TESTING.md) for the complete testing strategy. Key points:

- **Fast tests:** `uv run pytest -m "not slow and not nightly"` - run on every PR
- **Slow tests:** `uv run pytest -m slow` - run on main branch only (MCMC-based)
- **Nightly tests:** `uv run pytest -m nightly` - statistical tests run nightly

### Handling Stochastic Tests

BO algorithms are inherently stochastic. Tests should use:

1. **Invariant assertions** that always hold (bounds, monotonicity)
2. **Calibrated tolerances** from `scripts/calibrate_test_tolerances.py`
3. **Statistical testing** with `@pytest.mark.nightly` for mean/percentile checks

### Test Markers

- `@pytest.mark.slow` - Tests > 30s (MCMC, cross-validation)
- `@pytest.mark.nightly` - Statistical tests (multiple seeds)
- `@pytest.mark.smoke` - Fast critical path tests
- `@pytest.mark.tutorial` - BoTorch tutorial reproduction
<!-- - all async code uses SQLAlchemy 2.0 async patterns with `async_sessionmaker`

## Running the Project

```bash
# Setup
uv sync

# Run MCP server (for Claude integration)
uv run python scripts/run_mcp_server.py

# Run API server (for web UI)
uv run uvicorn api.main:app --reload --port 8000

# Run tests
uv run pytest

# Linting and formatting
uv run ruff check packages/ --fix
uv run ruff format packages/
uv run pyright packages/*/src
``` -->