"""Typed BayBE-specific option schemas.

The neutral :class:`bo_engine.types.ParameterSpec` and
:class:`bo_engine.types.OptimizationSpec` carry per-backend metadata via
free-form ``dict`` slots (``parameter_options`` / ``backend_options``).
This module defines the BayBE-native shape inside those slots so the
backend can validate user-supplied data once at intake time rather than
crashing with a ``KeyError`` deep inside the converter.

Two Pydantic submodels live here:

* :class:`BayBEParameterOptions` — per-parameter knobs (encoding,
  ``TaskParameter`` active values, substance descriptors,
  candidate-table membership).
* :class:`BayBEBackendOptions` — per-campaign knobs (recommender
  configuration overrides, candidate-table search-space mode).

Both classes use Pydantic's ``"forbid"`` extras policy so misspelled
keys become explicit validation errors instead of silently disappearing
into the opaque dict. The backend's ``validate_capabilities`` consumes
``extract_baybe_*_options`` to surface validation failures as
:class:`~bo_engine.backend_base.CapabilityReport` entries.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BayBEParameterEncoding(StrEnum):
    """Categorical-encoding choices supported by BayBE.

    Mirrors :class:`baybe.parameters.enum.CategoricalEncoding` so the
    neutral spec can carry the user's choice through validation without
    importing BayBE in the domain model.
    """

    OHE = "OHE"
    INT = "INT"


class BayBEParameterRole(StrEnum):
    """Role of a categorical-family BayBE parameter.

    ``categorical`` is the default vanilla one-hot/integer-encoded
    parameter, ``task`` switches to :class:`baybe.parameters.TaskParameter`
    for transfer-learning across related campaigns, ``substance``
    switches to :class:`baybe.parameters.SubstanceParameter` for
    cheminformatics descriptors, and ``custom`` switches to
    :class:`baybe.parameters.CustomDiscreteParameter` so the caller can
    supply a precomputed numeric representation per label (e.g. from
    quantum chemistry).
    """

    CATEGORICAL = "categorical"
    TASK = "task"
    SUBSTANCE = "substance"
    CUSTOM = "custom"


class BayBESubstanceEncoding(StrEnum):
    """Substance encoding strategies supported by BayBE.

    Mirrors :class:`baybe.parameters.enum.SubstanceEncoding` (the full
    scikit-fingerprints-backed set as of BayBE 0.15) **minus the
    deprecated ``RDKIT`` alias** (BayBE warns and maps it to
    ``RDKIT2DDESCRIPTORS``). Every member is pinned against the installed
    BayBE by a guard test so a BayBE rename fails CI instead of a user
    campaign. Defaults follow BayBE's own defaults; the converter passes
    the string through to :class:`baybe.parameters.SubstanceParameter`.
    """

    ATOMPAIR = "ATOMPAIR"
    AUTOCORR = "AUTOCORR"
    AVALON = "AVALON"
    BCUT2D = "BCUT2D"
    E3FP = "E3FP"
    ECFP = "ECFP"
    ELECTROSHAPE = "ELECTROSHAPE"
    MORGAN_FP = "MORGAN_FP"
    ERG = "ERG"
    ESTATE = "ESTATE"
    FUNCTIONALGROUPS = "FUNCTIONALGROUPS"
    GETAWAY = "GETAWAY"
    GHOSECRIPPEN = "GHOSECRIPPEN"
    KLEKOTAROTH = "KLEKOTAROTH"
    LAGGNER = "LAGGNER"
    LAYERED = "LAYERED"
    LINGO = "LINGO"
    MACCS = "MACCS"
    MAP = "MAP"
    MHFP = "MHFP"
    MORSE = "MORSE"
    MQNS = "MQNS"
    MORDRED = "MORDRED"
    PATTERN = "PATTERN"
    PHARMACOPHORE = "PHARMACOPHORE"
    PHYSIOCHEMICALPROPERTIES = "PHYSIOCHEMICALPROPERTIES"
    PUBCHEM = "PUBCHEM"
    RDF = "RDF"
    RDKITFINGERPRINT = "RDKITFINGERPRINT"
    RDKIT2DDESCRIPTORS = "RDKIT2DDESCRIPTORS"
    SECFP = "SECFP"
    TOPOLOGICALTORSION = "TOPOLOGICALTORSION"
    USR = "USR"
    USRCAT = "USRCAT"
    VSA = "VSA"
    WHIM = "WHIM"


# Encoding applied to a ``role=substance`` parameter when the caller leaves
# ``substance_encoding`` unset. MORDRED matches BayBE's own
# ``SubstanceParameter`` default (a ~1800-descriptor physicochemical block);
# kept here as a named constant so :func:`spec_to_parameters` never hardcodes
# the string and the documented default lives in exactly one place.
DEFAULT_SUBSTANCE_ENCODING = BayBESubstanceEncoding.MORDRED


class BayBEParameterOptions(BaseModel):
    """Typed BayBE-native parameter metadata.

    Stored as ``parameter_options["baybe"]`` on a neutral
    :class:`~bo_mcp_server.domain.campaign_spec.InputParameter` (or
    :class:`~bo_engine.types.ParameterSpec`). Only fields relevant to the
    parameter's role are consumed by :func:`spec_to_parameters`; foreign
    keys are rejected via ``extra="forbid"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: BayBEParameterRole = Field(
        default=BayBEParameterRole.CATEGORICAL,
        description=(
            "Which BayBE parameter class this becomes: 'categorical' (plain "
            "one-hot/integer-encoded), 'task' (TaskParameter, for "
            "transfer-learning across related campaigns; see "
            "`active_values`), 'substance' (SubstanceParameter, "
            "cheminformatics descriptors; see `substance_data`/"
            "`substance_encoding`), or 'custom' (CustomDiscreteParameter, a "
            "caller-supplied numeric representation; see "
            "`custom_descriptors`)."
        ),
    )
    encoding: BayBEParameterEncoding | None = Field(
        default=None,
        description=(
            "role=categorical only. 'OHE' (one-hot) or 'INT' (integer) "
            "encoding of the category levels. None keeps BayBE's own "
            "default (OHE)."
        ),
    )
    # Restrict recommendations to a subset of the declared categories while
    # keeping the full set measurable. Valid for every categorical-family
    # role (categorical / task / substance / custom); membership in the
    # declared categories is validated at intake, and the tuple must be
    # non-empty (an empty subset would silently mean "all categories").
    active_values: tuple[str, ...] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Restrict recommendations to this subset of the declared "
            "categories while keeping every declared category measurable. "
            "Valid for every categorical-family role (categorical / task / "
            "substance / custom); every entry must already be a declared "
            "category (checked at intake) and the tuple must be non-empty."
        ),
    )
    substance_data: dict[str, str] | None = Field(
        default=None,
        description=(
            "role=substance only. Maps each declared category label to its "
            "SMILES string, e.g. {'water': 'O', 'ethanol': 'CCO'}. Required "
            "for role='substance'; every declared category must have an "
            "entry."
        ),
    )
    substance_encoding: BayBESubstanceEncoding | None = Field(
        default=None,
        description=(
            "role=substance only. Molecular fingerprint/descriptor "
            "encoding computed from `substance_data`. None uses BayBE's "
            "default (MORDRED — a ~1800-descriptor physicochemical block)."
        ),
    )
    # role=substance only: keyword passthroughs for the fingerprint
    # computation and (for conformer-based encodings) the conformer
    # generation — forwarded verbatim to BayBE's SubstanceParameter.
    kwargs_fingerprint: dict[str, Any] | None = Field(
        default=None,
        description=(
            "role=substance only. Keyword arguments forwarded verbatim to "
            "the underlying scikit-fingerprints fingerprint computation "
            "selected by `substance_encoding`."
        ),
    )
    kwargs_conformer: dict[str, Any] | None = Field(
        default=None,
        description=(
            "role=substance only. Keyword arguments forwarded verbatim to "
            "conformer generation, for the `substance_encoding` choices "
            "that require 3D conformers (e.g. GETAWAY, WHIM, MORSE, RDF, "
            "AUTOCORR)."
        ),
    )
    # Numerical-discrete parameters only: measurement-matching slack used
    # by BayBE when assigning measured values to grid points. Must stay
    # below half the smallest gap between declared values (BayBE's own
    # validator); checked against the declared grid at intake.
    tolerance: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Numerical-discrete parameters only. Measurement-matching "
            "slack BayBE uses when assigning a measured value to its "
            "nearest declared grid point. Must stay below half the "
            "smallest gap between declared values (checked at intake); "
            "None keeps BayBE's own default (no slack)."
        ),
    )
    custom_descriptors: dict[str, dict[str, float]] | None = Field(
        default=None,
        description=(
            "role=custom only. Precomputed numeric representation per category: "
            "{category label: {descriptor name: value}}. Each label becomes one row "
            "of the table BayBE's CustomDiscreteParameter encodes. Rules (enforced at "
            "campaign creation, rejected with a clear error): keys must match the "
            "declared `categories` exactly (no missing/extra labels); at least 2 "
            "categories; every value numeric and finite (no null/NaN/inf); no "
            "descriptor column may be constant across labels (carries no information); "
            "and no two labels may share an identical descriptor vector (ambiguous "
            "representation). Give each label a distinct, informative vector."
        ),
    )
    decorrelate: bool | float = Field(
        default=True,
        description=(
            "role=custom only. Mirrors BayBE CustomDiscreteParameter.decorrelate: "
            "true drops descriptor columns correlated above the default "
            "threshold (0.7), false keeps the table as-is, or a float in "
            "(0, 1) sets a custom correlation threshold."
        ),
    )


class BayBEInitialRecommender(StrEnum):
    """Initial-design (pre-model) recommender selection.

    ``random`` keeps the historical :class:`baybe.recommenders.RandomRecommender`
    default; ``fps`` selects farthest-point sampling and the clustering
    members select the corresponding clustering recommenders. All
    non-random members require a purely discrete/enumerable search space
    (their BayBE ``compatibility`` is ``SearchSpaceType.DISCRETE``) —
    validated at intake by the capability layer. The sequential/streaming
    meta-recommenders are deliberately not exposed:
    ``StreamingSequentialMetaRecommender`` is non-serializable and would
    break the JSON state envelope.
    """

    RANDOM = "random"
    FPS = "fps"
    KMEANS = "kmeans"
    PAM = "pam"
    GMM = "gmm"


class BayBEHybridSampler(StrEnum):
    """Sampling strategy for the discrete part of hybrid-space optimization.

    Mirrors :class:`baybe.utils.sampling_algorithms.DiscreteSamplingMethod`;
    pinned by a guard test against renames (the ``BayBESubstanceEncoding``
    precedent).
    """

    RANDOM = "Random"
    FPS = "FPS"


class BayBEBayesianRecommenderOptions(BaseModel):
    """Tuning knobs for BayBE's GP-phase :class:`BotorchRecommender`.

    All fields default to ``None`` = keep BayBE's own defaults. Note that
    ``sampling_percentage`` applies to **hybrid** spaces only — purely
    discrete spaces are scored exhaustively by BayBE regardless.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequential_continuous: bool | None = Field(
        default=None,
        description=(
            "Whether to apply sequential-greedy (True) or joint batch (False) "
            "optimization in **continuous** search spaces only — discrete/"
            "hybrid spaces always optimize sequentially regardless of this "
            "flag. None keeps BayBE's default (True)."
        ),
    )
    hybrid_sampler: BayBEHybridSampler | None = Field(
        default=None,
        description=(
            "Discrete-subspace sampling strategy for **hybrid** (mixed "
            "discrete/continuous) spaces only: 'Random' or 'FPS' "
            "(farthest-point sampling). None keeps BayBE's default (no "
            "sampling — the full discrete subspace is used, and "
            "`sampling_percentage` is then ignored)."
        ),
    )
    sampling_percentage: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of the discrete subspace to sample per acquisition "
            "step, for **hybrid** spaces only when `hybrid_sampler` is set "
            "(purely discrete spaces are scored exhaustively regardless). "
            "None keeps BayBE's default (1.0 — the full subspace)."
        ),
    )
    n_restarts: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of gradient-based multi-start restarts for continuous "
            "acquisition optimization; does not affect purely discrete "
            "optimization. None keeps BayBE's default (10)."
        ),
    )
    n_raw_samples: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of raw samples drawn for the initialization heuristic "
            "that seeds `n_restarts`; does not affect purely discrete "
            "optimization. None keeps BayBE's default (64)."
        ),
    )


class BayBERecommenderConfig(BaseModel):
    """Recommender configuration overrides.

    ``switch_after`` delays the initial → BO recommender switch and takes
    precedence over the neutral ``OptimizationSpec.initial_design_size``
    knob (explicit BayBE option wins); without either, BayBE switches
    after the first measurement. ``initial_recommender`` selects the
    space-filling phase recommender, and ``bayesian`` tunes the GP-phase
    :class:`BotorchRecommender`. The overall graph stays the two-phase
    meta-recommender.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    switch_after: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of measurements after which the two-phase meta-"
            "recommender switches from `initial_recommender` to the "
            "GP-phase BotorchRecommender. Takes precedence over the "
            "neutral `initial_design_size` when both are set."
        ),
    )
    initial_recommender: BayBEInitialRecommender = Field(
        default=BayBEInitialRecommender.RANDOM,
        description=(
            "Space-filling recommender used before the switch to the "
            "GP-phase recommender (see `switch_after`). 'random' works on "
            "any search space; 'fps'/'kmeans'/'pam'/'gmm' require a purely "
            "discrete/enumerable search space."
        ),
    )
    bayesian: BayBEBayesianRecommenderOptions | None = Field(
        default=None,
        description="Tuning knobs for the GP-phase BotorchRecommender.",
    )


class BayBESurrogateKind(StrEnum):
    """Curated surrogate-model choices.

    Deliberately a subset of BayBE's surrogate zoo: the bandit surrogate
    requires binary targets, and ``CustomONNXSurrogate`` needs a
    user-supplied model artifact with no intake path — both are excluded
    by design. Multi-objective campaigns replicate a non-GP surrogate per
    target via ``CompositeSurrogate.from_replication``.
    """

    GP = "gp"
    RANDOM_FOREST = "random_forest"
    NGBOOST = "ngboost"
    BAYESIAN_LINEAR = "bayesian_linear"
    MEAN_PREDICTION = "mean_prediction"


class BayBEGPPreset(StrEnum):
    """GP hyperparameter presets (``GaussianProcessSurrogate.from_preset``)."""

    BAYBE = "BAYBE"
    BOTORCH = "BOTORCH"
    CHEN = "CHEN"
    EDBO = "EDBO"
    EDBO_SMOOTHED = "EDBO_SMOOTHED"
    HVARFNER = "HVARFNER"


class BayBEKernelKind(StrEnum):
    """Curated GP kernel choices.

    ``matern``/``rbf`` are the historical stationary/smooth defaults.
    ``linear``/``periodic``/``polynomial``/``rq``/``rff`` extend the zoo for
    non-stationary or periodic objective surfaces Matern/RBF can't
    represent well (mirrors ``baybe.kernels.basic``). Composite kernels
    (``AdditiveKernel``/``ProductKernel``) and the transfer-learning
    ``IndexKernel`` family are deliberately excluded: they need a
    recursive/multi-task config shape, not a single enum member.
    """

    MATERN = "matern"
    RBF = "rbf"
    LINEAR = "linear"
    PERIODIC = "periodic"
    POLYNOMIAL = "polynomial"
    RQ = "rq"
    RFF = "rff"


# Smoothness values accepted by the Matern kernel family (gpytorch contract).
# ``float`` isn't a valid ``Literal`` parameter per PEP 586 (and rejected by
# ty's `invalid-type-form` check), so the allowed set is enforced by
# ``validate_kernel_fields`` and separately surfaced to schema consumers via
# ``json_schema_extra`` below.
MATERN_ALLOWED_NU: tuple[float, ...] = (0.5, 1.5, 2.5)


class BayBEKernelConfig(BaseModel):
    """GP kernel selection (wrapped in a ScaleKernel by the converter).

    ``nu`` parameterizes ``kind='matern'`` only. ``period_length``
    parameterizes ``kind='periodic'`` only and is optional (omit for
    gpytorch's own initial value). ``power`` and ``num_samples`` are
    *required* companions for ``kind='polynomial'`` and ``kind='rff'``
    respectively. ``linear`` and ``rq`` take no extra parameters here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BayBEKernelKind = Field(
        description=(
            "Base kernel family. 'matern' (optionally tuned via `nu`) and "
            "'rbf' are smooth/stationary defaults. 'periodic' suits cyclic "
            "parameters (optional `period_length`). 'linear' and 'rq' "
            "(rational quadratic) suit non-stationary surfaces. "
            "'polynomial' requires `power`. 'rff' (random Fourier "
            "features, an RBF approximation) requires `num_samples`."
        )
    )
    nu: float | None = Field(
        default=None,
        description=(
            "Matern smoothness. Only valid when kind='matern'; rejected on "
            f"every other kind. Must be one of {list(MATERN_ALLOWED_NU)}."
        ),
        json_schema_extra={"enum": [*MATERN_ALLOWED_NU, None]},
    )
    period_length: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Periodic kernel's initial period length. Only valid when "
            "kind='periodic'; rejected on every other kind. Optional — "
            "omit to use gpytorch's own initial value."
        ),
    )
    power: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Polynomial kernel's power (degree). Required when "
            "kind='polynomial'; rejected on every other kind."
        ),
    )
    num_samples: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of random Fourier frequencies to draw. Required when "
            "kind='rff'; rejected on every other kind."
        ),
    )

    @model_validator(mode="after")
    def validate_kernel_fields(self) -> BayBEKernelConfig:
        """Per-kind parameters only apply to (and are required by) their kind."""
        if self.nu is not None and self.kind != BayBEKernelKind.MATERN:
            msg = "nu is only valid for the matern kernel"
            raise ValueError(msg)
        if self.nu is not None and self.nu not in MATERN_ALLOWED_NU:
            msg = f"matern nu must be one of {list(MATERN_ALLOWED_NU)}"
            raise ValueError(msg)
        if self.period_length is not None and self.kind != BayBEKernelKind.PERIODIC:
            msg = "period_length is only valid for the periodic kernel"
            raise ValueError(msg)
        if self.power is not None and self.kind != BayBEKernelKind.POLYNOMIAL:
            msg = "power is only valid for the polynomial kernel"
            raise ValueError(msg)
        if self.kind == BayBEKernelKind.POLYNOMIAL and self.power is None:
            msg = "power is required for the polynomial kernel"
            raise ValueError(msg)
        if self.num_samples is not None and self.kind != BayBEKernelKind.RFF:
            msg = "num_samples is only valid for the rff kernel"
            raise ValueError(msg)
        if self.kind == BayBEKernelKind.RFF and self.num_samples is None:
            msg = "num_samples is required for the rff kernel"
            raise ValueError(msg)
        return self


class BayBESurrogateConfig(BaseModel):
    """Surrogate-model configuration (``backend_options['baybe'].surrogate``)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: BayBESurrogateKind = Field(
        default=BayBESurrogateKind.GP,
        description=(
            "Surrogate model family. 'gp' (Gaussian process, the only kind "
            "`gp_preset`/`kernel` apply to) is the default and the only "
            "kind that meaningfully handles continuous search spaces; "
            "'random_forest'/'ngboost'/'bayesian_linear'/'mean_prediction' "
            "are non-differentiable alternatives valid on discrete spaces "
            "only."
        ),
    )
    gp_preset: BayBEGPPreset | None = Field(
        default=None,
        description=(
            "kind='gp' only. Named hyperparameter-prior preset ('BAYBE', "
            "'BOTORCH', 'CHEN', 'EDBO', 'EDBO_SMOOTHED', 'HVARFNER') "
            "combined with `kernel` via `GaussianProcessSurrogate."
            "from_preset`. None keeps BayBE's own dimension/parameter-"
            "adaptive default prior selection."
        ),
    )
    kernel: BayBEKernelConfig | None = Field(
        default=None,
        description=(
            "kind='gp' only. Curated GP kernel selection; combined with "
            "`gp_preset` if both are set. None keeps BayBE's own default "
            "kernel (a dimension-scaled Matern 5/2)."
        ),
    )

    @model_validator(mode="after")
    def validate_gp_only_fields(self) -> BayBESurrogateConfig:
        """Preset/kernel parameterize the GP surrogate only."""
        if self.kind != BayBESurrogateKind.GP and (
            self.gp_preset is not None or self.kernel is not None
        ):
            msg = "gp_preset/kernel are only valid with kind='gp'"
            raise ValueError(msg)
        return self


class BayBEExplainerKind(StrEnum):
    """Explainer backends for SHAP-based feature importance.

    Mirrors ``baybe.insights.shap.EXPLAINERS`` (SHAP explainers plus the
    non-SHAP LIME/MAPLE attributions); pinned by a guard test. The
    non-Kernel SHAP explainers reject categorical experimental
    representations — combine them with ``use_comp_rep=True`` on
    categorical campaigns.
    """

    KERNEL = "KernelExplainer"
    EXACT = "ExactExplainer"
    PERMUTATION = "PermutationExplainer"
    PARTITION = "PartitionExplainer"
    ADDITIVE = "AdditiveExplainer"
    LIME_TABULAR = "LimeTabular"
    MAPLE = "Maple"


class BayBEInsightsOptions(BaseModel):
    """Diagnostics/insights options (``backend_options['baybe'].insights``).

    ``explainer`` selects the SHAP/LIME/MAPLE explainer backend (BayBE
    default: KernelExplainer). ``use_comp_rep`` explains the computational
    (encoded) representation instead of the experimental one.
    ``include_row_level`` adds bounded per-observation attributions to the
    diagnostics payload: rows follow the campaign's measurement order and
    carry no per-row identifiers, and on multi-target campaigns only the
    *first* target's attributions are returned. ``row_level_max_rows``
    overrides the payload bound (default:
    ``bo_engine_baybe.constants.DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS``). Plots
    are deliberately not exposed — binary images do not fit the JSON tool
    contract.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    explainer: BayBEExplainerKind | None = Field(
        default=None,
        description=(
            "SHAP/LIME/MAPLE explainer backend. None keeps BayBE's default "
            "(KernelExplainer). Non-Kernel SHAP explainers reject "
            "categorical experimental representations — combine them with "
            "`use_comp_rep=true` on categorical campaigns."
        ),
    )
    use_comp_rep: bool = Field(
        default=False,
        description=(
            "Explain the computational (encoded) representation instead "
            "of the experimental one. Required for non-Kernel SHAP "
            "explainers on categorical campaigns."
        ),
    )
    include_row_level: bool = Field(
        default=False,
        description=(
            "Add bounded per-observation attributions to the diagnostics "
            "payload. Rows follow the campaign's measurement order and "
            "carry no per-row identifiers; on multi-target campaigns only "
            "the first target's attributions are returned."
        ),
    )
    row_level_max_rows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Row cap for `include_row_level`. None keeps the named "
            "default (`bo_engine_baybe.constants."
            "DEFAULT_MAX_ROW_LEVEL_SHAP_ROWS`)."
        ),
    )


class BayBEBackendOptions(BaseModel):
    """Typed BayBE-native campaign-level options.

    Stored as ``backend_options["baybe"]`` on a neutral
    :class:`~bo_mcp_server.domain.campaign_spec.CampaignSpec`. Validated at
    intake so misshaped payloads cannot reach
    :class:`~bo_engine_baybe.backend.BayBEBackend.generate_suggestions`.

    The ``allow_recommending_*`` toggles default to ``None`` = keep the
    backend's historical behavior (``False`` for purely discrete spaces,
    BayBE's AUTO otherwise — which resolves the pending flag to ``True``
    on spaces with a continuous part). An explicit
    ``allow_recommending_pending_experiments=False`` is honored on purely
    discrete spaces only: BayBE forbids it whenever the search space has
    a continuous subspace ("for algorithmic reasons" — pending points are
    handled by the acquisition function there, not by candidate
    exclusion), so the capability layer reports that combination
    UNSUPPORTED at intake. ``measurements_must_be_within_tolerance``
    maps to BayBE's ``add_measurements(numerical_measurements_must_be_within_tolerance=...)``
    (default ``True``); relaxing it accepts out-of-tolerance numeric
    values instead of raising — a data-quality trade-off the caller opts
    into explicitly.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    recommender: BayBERecommenderConfig | None = Field(
        default=None,
        description="Recommender graph configuration overrides.",
    )
    surrogate: BayBESurrogateConfig | None = Field(
        default=None,
        description=(
            "Surrogate-model configuration. None keeps BayBE's implicit "
            "default GP surrogate untouched."
        ),
    )
    insights: BayBEInsightsOptions | None = Field(
        default=None,
        description="SHAP/LIME/MAPLE feature-importance diagnostics configuration.",
    )
    allow_recommending_pending_experiments: bool | None = Field(
        default=None,
        description=(
            "Whether already-recommended-but-unmeasured points remain "
            "candidates. None keeps the backend's historical behavior "
            "(False on purely discrete spaces, BayBE's AUTO — which "
            "resolves to True — on spaces with a continuous part). An "
            "explicit False is honored on purely discrete spaces only: "
            "BayBE forbids it whenever the search space has a continuous "
            "subspace (pending points are handled by the acquisition "
            "function there, not by candidate exclusion), and that "
            "combination is reported UNSUPPORTED at intake."
        ),
    )
    allow_recommending_already_measured: bool | None = Field(
        default=None,
        description=(
            "Whether already-measured points remain candidates. None keeps "
            "the backend's historical behavior: False on purely discrete "
            "spaces (overriding BayBE's own AUTO, which always resolves "
            "this flag to True regardless of search-space type), or "
            "BayBE's AUTO (True) on spaces with a continuous part. An "
            "explicit False is rejected by BayBE (IncompatibilityError) "
            "whenever the search space has a continuous subspace — as "
            "with the other two `allow_recommending_*` flags, it is only "
            "honored on purely discrete spaces."
        ),
    )
    allow_recommending_already_recommended: bool | None = Field(
        default=None,
        description=(
            "Whether previously-recommended points remain candidates on "
            "later calls. None keeps the backend's historical behavior: "
            "False on purely discrete spaces, BayBE's AUTO (which also "
            "resolves to True) on spaces with a continuous part. An "
            "explicit False is rejected by BayBE (IncompatibilityError) "
            "whenever the search space has a continuous subspace — as "
            "with the other two `allow_recommending_*` flags, it is only "
            "honored on purely discrete spaces."
        ),
    )
    measurements_must_be_within_tolerance: bool | None = Field(
        default=None,
        description=(
            "Maps to BayBE's `add_measurements(numerical_measurements_"
            "must_be_within_tolerance=...)`. None keeps BayBE's default "
            "(True — raises on out-of-tolerance numeric values); False "
            "accepts out-of-tolerance values instead of raising, a "
            "data-quality trade-off the caller opts into explicitly."
        ),
    )
    # Large-categorical safeguard overrides. ``None`` keeps the named
    # defaults in :mod:`bo_engine_baybe.constants`
    # (DEFAULT_MAX_SEARCHSPACE_STATE_BYTES / DEFAULT_MAX_CANDIDATES); the
    # values bound the serialized size and row count of the enumerated
    # discrete subspace before deterministic subsampling kicks in.
    max_searchspace_state_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Byte-size bound on the serialized enumerated discrete "
            "subspace before deterministic subsampling kicks in. None "
            "keeps the named default (`bo_engine_baybe.constants."
            "DEFAULT_MAX_SEARCHSPACE_STATE_BYTES`)."
        ),
    )
    max_candidates: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Row-count bound on the enumerated discrete subspace before "
            "deterministic subsampling kicks in. None keeps the named "
            "default (`bo_engine_baybe.constants.DEFAULT_MAX_CANDIDATES`)."
        ),
    )


def extract_baybe_parameter_options(
    raw: dict[str, dict[str, Any]] | None,
) -> BayBEParameterOptions:
    """Return the validated BayBE per-parameter options.

    When ``raw`` is missing or has no ``"baybe"`` key, the default options
    are returned so callers do not have to guard against ``None``. Invalid
    payloads raise ``pydantic.ValidationError`` — the backend's
    ``validate_capabilities`` catches the error and surfaces it as a
    :class:`~bo_engine.backend_base.CapabilityReport` instead of failing
    suggestion generation.
    """
    if not raw or "baybe" not in raw:
        return BayBEParameterOptions()
    return BayBEParameterOptions.model_validate(raw["baybe"])


def extract_baybe_backend_options(
    raw: dict[str, dict[str, Any]] | None,
) -> BayBEBackendOptions:
    """Return the validated BayBE per-campaign options.

    Same defaulting/validation contract as
    :func:`extract_baybe_parameter_options`.
    """
    if not raw or "baybe" not in raw:
        return BayBEBackendOptions()
    return BayBEBackendOptions.model_validate(raw["baybe"])
