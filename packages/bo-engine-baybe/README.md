# bo-engine-baybe

BayBE backend for Bayesian Optimization. Implements the `BOBackend` protocol using [BayBE](https://github.com/emdgroup/baybe) (Bayesian Back End by Merck KGaA) as the optimization engine.

## Scope

- Default backend of BO-MCP (BoTorch remains as the legacy fallback)
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

## Custom representations (BayBE only)

When you already have a precomputed numeric representation for each label —
quantum-chemistry descriptors, learned embeddings, measured properties — feed it
directly via `role="custom"`, which maps to BayBE's
[`CustomDiscreteParameter`](https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
As with `substance`, the base parameter stays `categorical` and the declared
`categories` remain the experimental values you submit; the descriptor table is
only how BayBE encodes them for the GP.

```json
{
  "name": "ligand",
  "type": "categorical",
  "categories": ["L1", "L2", "L3", "L4"],
  "parameter_options": {
    "baybe": {
      "role": "custom",
      "custom_descriptors": {
        "L1": {"volume": 1.0, "charge": -0.2},
        "L2": {"volume": 2.5, "charge": 0.1},
        "L3": {"volume": 3.1, "charge": 0.4},
        "L4": {"volume": 4.8, "charge": -0.7}
      },
      "decorrelate": true
    }
  }
}
```

- **`custom_descriptors`**: a `{category_label: {descriptor: value}}` map that
  must cover every declared category (extra/missing labels are rejected at
  intake). Each label's dict becomes a row of the descriptor table.
- **`decorrelate`**: `true` (default) drops highly correlated descriptor columns,
  `false` keeps the table as-is, or a float in `(0, 1)` sets the correlation
  threshold. Mirrors BayBE's `CustomDiscreteParameter.decorrelate`.
- The table must satisfy BayBE's `CustomDiscreteParameter` rules, all enforced at
  campaign creation as clear capability errors (not a deferred crash):
  - keys match the declared `categories` exactly — no missing or extra labels;
  - at least 2 categories;
  - every value numeric and finite (no null / NaN / inf);
  - no descriptor column constant across labels (a single-value column carries no
    information);
  - no two labels sharing an identical descriptor vector (duplicate **rows** →
    ambiguous representation).
  Note the last two are separate: a constant *column* and a duplicate *row* are
  different rejections. Give each label a distinct, informative vector.
- **BayBE-only**, same routing as substance: `backend="auto"` routes to BayBE; a
  pinned `backend="botorch"` is rejected (BoTorch would one-hot the labels and
  silently drop the representation) and the veto cannot be acknowledged away. No
  `baybe[chem]` needed — the representation is supplied, not computed.

## Package Structure

```text
bo_engine_baybe/
    __init__.py     # Exports BayBEBackend
    backend.py      # BayBEBackend class implementing BOBackend protocol
    converters.py   # OptimizationSpec <-> BayBE type converters
```
