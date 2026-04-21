# bo-engine

Bayesian Optimization engine built on BoTorch. Provides the core optimization algorithms, model fitting, diagnostics, and the `BOBackend` protocol that enables pluggable backends.

## Scope

- Pure optimization library with no server, database, or network dependencies
- Defines the `BOBackend` protocol that all backends (BoTorch, BayBE, etc.) implement
- Provides `BoTorchBackend` as the default, full-featured backend
- All functions accept and return plain Python types (dicts, dataclasses) — no framework lock-in at the API boundary

## Installation

```bash
pip install bo-engine
```

Requires Python >= 3.13 and PyTorch >= 2.0.

## Quick Start

```python
from bo_engine.types import OptimizationSpec, ParameterSpec, ObjectiveSpec, ParameterType
from bo_engine.botorch_backend import BoTorchBackend

spec = OptimizationSpec(
    parameters=[
        ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
    ],
    objectives=[ObjectiveSpec(name="y", minimize=True)],
    batch_size=2,
)

backend = BoTorchBackend()

# Generate initial space-filling design
designs = backend.generate_initial_design(spec, n_points=5)

# After collecting results, generate model-guided suggestions
from bo_engine.types import ObservationData
observations = [
    ObservationData(parameter_values=d, objective_values={"y": evaluate(d)})
    for d in designs
]
batch = backend.generate_suggestions(spec, observations, batch_size=2, iteration=1)
```

## Features

### Core Optimization

- Single-objective (qLogNEI/qLogEI) and multi-objective (qLogNEHVI) acquisition
- Mixed parameter types: continuous, discrete, categorical
- Constraint handling: sum, linear, outcome constraints

### Advanced Methods

- **TuRBO** -- Trust Region BO for high-dimensional problems (20+ parameters)
- **SAASBO** -- Sparse Axis-Aligned Subspace BO for very high-dimensional problems (50+)
- **Multi-fidelity** -- Cost-aware optimization with qMFKG
- **Transfer Learning** -- RGPE ensemble for leveraging prior campaign data
- **Input Warping** -- Kumaraswamy CDF for non-stationary objectives

### Diagnostics

- Hypervolume and Pareto front tracking
- LOO cross-validation (batch, approximate, k-fold)
- Feature importance via inverse lengthscales and SHAP
- Model health assessment and convergence detection
- Prediction intervals and calibration metrics

### Multi-Backend Protocol

- `BOBackend` protocol in `bo_engine.backend` defines the interface
- `BoTorchBackend` in `bo_engine.botorch_backend` is the default implementation
- `Feature` enum for capability advertisement (10 features)
- Third-party backends implement the same protocol (see `bo-engine-baybe`)

## Package Structure

```text
bo_engine/
    backend.py           # BOBackend protocol, Feature enum, SuggestionBatch
    botorch_backend.py   # Default BoTorch backend implementation
    types.py             # OptimizationSpec, ObservationData, ParameterSpec, etc.
    suggestions.py       # Core suggestion generation
    acquisition.py       # Acquisition function creation
    diagnostics.py       # Hypervolume, Pareto, health metrics, LOO-CV
    models.py            # GP model fitting
    constants.py         # Configurable thresholds
    transforms.py        # Parameter encoding and normalization
    constraints.py       # Constraint handling
    turbo.py             # TuRBO trust region management
    saasbo.py            # SAASBO high-dimensional optimization
    multifidelity.py     # Multi-fidelity BO
    transfer_learning.py # RGPE transfer learning
    batch_diversity.py   # Batch diversity metrics and enforcement
    convergence.py       # Convergence detection
    cross_validation.py  # LOO-CV and k-fold CV
```
