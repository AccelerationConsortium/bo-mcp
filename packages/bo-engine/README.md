# bo-engine

Bayesian Optimization engine using BoTorch for multi-objective optimization.

## Installation

```bash
pip install bo-engine
```

## Features

### Core Optimization
- Multi-objective optimization with qLogNEHVI acquisition
- Single-objective optimization with qLogNEI/qLogEI acquisition
- ModelListGP with Matérn 5/2 kernel and ARD
- Constraint handling (sum, linear, outcome constraints)
- Mixed parameter types (continuous, discrete, categorical)

### Advanced Methods
- **TuRBO** - Trust Region Bayesian Optimization for high-dimensional problems (≥20 parameters)
- **SAASBO** - Sparse Axis-Aligned Subspace BO for very high-dimensional problems (≥50 parameters)
- **Multi-fidelity BO** - Cost-aware optimization with qMFKG acquisition
- **Transfer Learning** - RGPE for leveraging prior campaign data
- **Input Warping** - Kumaraswamy CDF warping for non-stationary objectives

### Diagnostics
- Hypervolume and Pareto front tracking
- LOO cross-validation for model quality assessment
- Feature importance via inverse lengthscales and SHAP
- Model health assessment

### GPU Acceleration
- Automatic GPU detection and acceleration (v2.3)
- Device management utilities

## Usage

```python
from bo_engine import generate_initial_design, generate_next_batch
from bo_engine.diagnostics import compute_hypervolume, compute_pareto_front
from bo_engine.feature_importance import compute_feature_importance

# Generate initial Sobol design
suggestions = generate_initial_design(campaign_spec, n_suggestions=10)

# After collecting results, generate next batch
next_suggestions = generate_next_batch(campaign_spec, results, batch_size=3)
```

## Configuration Constants

The engine provides configurable constants for tuning behavior. Import from `bo_engine.constants`:

```python
from bo_engine import (
    # High-dimensional thresholds
    TURBO_MIN_DIMENSIONS,          # Min params for TuRBO (default: 20)
    SAASBO_MIN_DIMENSIONS,         # Min params for SAASBO (default: 50)

    # Model training thresholds
    MIN_OBSERVATIONS_FOR_MODEL,    # Min observations before model training (default: 2)
    MIN_OBSERVATIONS_FOR_LOO_CV,   # Min observations for LOO-CV (default: 5)

    # Confidence thresholds
    CONFIDENCE_HIGH_UNCERTAINTY_THRESHOLD,    # (default: 0.1)
    CONFIDENCE_MEDIUM_UNCERTAINTY_THRESHOLD,  # (default: 0.3)

    # TuRBO configuration
    TURBO_INITIAL_LENGTH,          # Initial trust region length (default: 0.8)
    TURBO_SUCCESS_TOLERANCE,       # Successes before expansion (default: 10)
)
```
