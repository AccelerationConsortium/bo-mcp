# bo-engine-baybe

BayBE backend for Bayesian Optimization. Implements the `BOBackend` protocol using [BayBE](https://github.com/emdgroup/baybe) (Bayesian Back End by Merck KGaA) as the optimization engine.

## Scope

- Alternative backend to the default BoTorch backend
- Native support for categorical parameters with smart encodings (OHE, custom)
- Mixed search spaces (continuous + discrete + categorical) handled natively
- Multi-objective optimization via BayBE's `ParetoObjective` (qLogNEHVI)
- Campaign state serialization via `Campaign.to_json()`/`from_json()`
- BayBE-native posterior stats, acquisition values, and model introspection
- Delegates to `bo-engine` for hypervolume, duplicate detection, and batch diversity (BayBE doesn't provide these)

## Installation

```bash
pip install bo-engine-baybe
```

With optional SHAP-based feature importance:

```bash
pip install bo-engine-baybe[insights]
```

Requires Python >= 3.13, BayBE >= 0.14, and `bo-engine`.

## Quick Start

```python
from bo_engine.types import OptimizationSpec, ParameterSpec, ObjectiveSpec, ParameterType
from bo_engine_baybe import BayBEBackend

spec = OptimizationSpec(
    parameters=[
        ParameterSpec(name="temp", type=ParameterType.CONTINUOUS, bounds=(200.0, 400.0)),
        ParameterSpec(
            name="solvent",
            type=ParameterType.CATEGORICAL,
            categories=["Water", "Ethanol", "DMF"],
        ),
    ],
    objectives=[ObjectiveSpec(name="yield", minimize=False)],
    batch_size=3,
)

backend = BayBEBackend()
designs = backend.generate_initial_design(spec, n_points=5)
```

## BayBE Features Used

| BayBE API | What It Does |
| --------- | ------------ |
| `Campaign.recommend()` | Core suggestion generation |
| `Campaign.posterior_stats()` | Predicted mean/std per suggestion |
| `Campaign.acquisition_values()` | Per-suggestion acquisition function values |
| `Campaign.get_surrogate().to_botorch()` | Model hyperparameter extraction |
| `Campaign.to_json()` / `from_json()` | State persistence across calls |
| `ParetoObjective` | Multi-objective via qLogNEHVI |
| `allow_recommending_already_measured=False` | Prevents re-suggesting measured points (discrete spaces) |
| `SHAPInsight.from_campaign()` | Feature importance (optional, requires `[insights]`) |

## Supported Features

```python
backend.supported_features
# frozenset({MULTI_OBJECTIVE, CONSTRAINTS, CATEGORICAL, MIXED_SEARCH_SPACE})
```

Not supported (use BoTorch backend instead): multi-fidelity, TuRBO, SAASBO, cost-aware, input warping, outcome constraints.

## Package Structure

```text
bo_engine_baybe/
    __init__.py     # Exports BayBEBackend
    backend.py      # BayBEBackend class implementing BOBackend protocol
    converters.py   # OptimizationSpec <-> BayBE type converters
```
