# bo-engine-baybe

BayBE backend for the Bayesian Optimization MCP server.

Implements the `BOBackend` protocol using [BayBE](https://github.com/emdgroup/baybe) as the optimization engine, providing native support for categorical parameters, mixed search spaces, and chemical encodings.

## Installation

```bash
pip install bo-engine-baybe
```

## Usage

```python
from bo_engine_baybe import BayBEBackend

backend = BayBEBackend()
designs = backend.generate_initial_design(spec, n_points=5)
```
