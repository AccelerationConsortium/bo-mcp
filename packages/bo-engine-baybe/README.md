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

This pulls in `baybe[chem]` (RDKit + scikit-fingerprints) by default, so
molecular / substance parameters (see below) work out of the box.

With optional SHAP-based feature importance:

```bash
pip install bo-engine-baybe[insights]
```

Requires Python >= 3.13, `baybe[chem] >= 0.14`, and `bo-engine`.

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

## Molecular / substance parameters (BayBE only)

BayBE can treat a categorical parameter whose labels are molecules as a
[`SubstanceParameter`](https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html):
each label maps to a SMILES string, which BayBE turns into cheminformatics
descriptors so the GP reasons over molecular structure instead of treating the
labels as unordered categories.

You opt in per parameter through the BayBE slot of `parameter_options` — there
is **no** neutral `SUBSTANCE` parameter type, mirroring how `TaskParameter` is
modeled. The base parameter stays `categorical`; the labels you declare in
`categories` are the experimental values you submit and observe (e.g.
`"ethanol"`), **not** the SMILES.

```json
{
  "name": "solvent",
  "type": "categorical",
  "categories": ["water", "ethanol", "methanol", "acetone", "toluene"],
  "parameter_options": {
    "baybe": {
      "role": "substance",
      "substance_data": {
        "water": "O",
        "ethanol": "CCO",
        "methanol": "CO",
        "acetone": "CC(=O)C",
        "toluene": "Cc1ccccc1"
      },
      "substance_encoding": "MORDRED"
    }
  }
}
```

- **`role`**: set to `"substance"` (default `"categorical"`; `"task"` selects a
  `TaskParameter`).
- **`substance_data`**: a `{category_label: SMILES}` map that must cover every
  declared category. Malformed SMILES are rejected at intake (a clear
  capability error), not deep inside RDKit at suggestion time.
- **`substance_encoding`**: one of `MORDRED` (default — BayBE's ~1800-descriptor
  block), `ECFP` (a lighter Morgan-style fingerprint), `RDKIT2DDESCRIPTORS`, or
  `RDKITFINGERPRINT`. Omit it to use the default.

Caveats:

- **BayBE-only.** This is a BayBE-native capability. `backend="auto"` routes a
  substance spec to BayBE automatically; a pinned `backend="botorch"` is
  rejected (BoTorch has no chemistry kernel and would silently treat the SMILES
  labels as opaque one-hot categories). This veto cannot be bypassed via
  `acknowledge_degradations`.
- **`baybe[chem]` required.** RDKit + scikit-fingerprints ship as default
  dependencies of this package, so substance parameters work out of the box.

A complete, runnable intake payload lives at
[`docs/examples/substance_solvent_screening.json`](../../docs/examples/substance_solvent_screening.json).

## Package Structure

```text
bo_engine_baybe/
    __init__.py     # Exports BayBEBackend
    backend.py      # BayBEBackend class implementing BOBackend protocol
    converters.py   # OptimizationSpec <-> BayBE type converters
```
