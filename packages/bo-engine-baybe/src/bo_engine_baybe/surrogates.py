"""Surrogate / kernel / noise-prior construction for the BayBE backend.

Builds the curated ``backend_options['baybe'].surrogate`` selection into a
BayBE surrogate object consumed by ``BotorchRecommender(surrogate_model=...)``.
The curation is deliberately narrow (see
:class:`~bo_engine_baybe.options.BayBESurrogateKind`); widening happens on
request, always behind the pin-guard tests.

Reference: BayBE surrogates userguide
(https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
"""

from __future__ import annotations

import logging
from typing import assert_never

from baybe.kernels import (
    MaternKernel,
    PeriodicKernel,
    PolynomialKernel,
    RBFKernel,
    RFFKernel,
    RQKernel,
    ScaleKernel,
)
from baybe.kernels.base import Kernel
from baybe.surrogates import (
    BayesianLinearSurrogate,
    GaussianProcessSurrogate,
    MeanPredictionSurrogate,
    NGBoostSurrogate,
    RandomForestSurrogate,
)
from baybe.surrogates.base import SurrogateProtocol

from bo_engine.types import OptimizationSpec
from bo_engine_baybe.options import (
    BayBEBackendOptions,
    BayBEKernelConfig,
    BayBEKernelKind,
    BayBESurrogateConfig,
    BayBESurrogateKind,
)

logger = logging.getLogger(__name__)

# Reason surfaced when a spec sets the neutral ``noise_prior_params`` on
# BayBE. Injecting a parameterized GP noise-prior likelihood requires a
# custom gpytorch object (or a custom likelihood-factory class), and
# BayBE's cattrs campaign serialization cannot restructure either — a
# campaign carrying one would fail every ``Campaign.from_json`` restore
# and silently rebuild from scratch each iteration. The option therefore
# stays BoTorch-only and is reported IGNORED here so ``backend="auto"``
# prefers BoTorch when the caller sets it.
NOISE_PRIOR_IGNORED_REASON = (
    "noise_prior_params targets the BoTorch backend. BayBE's campaign "
    "persistence (Campaign.to_json/from_json) cannot serialize a "
    "parameterized noise-prior likelihood, so honoring it would break "
    "state restoration; the option is ignored on BayBE."
)

# Default Matern smoothness when the kernel config omits ``nu`` — matches
# ``baybe.kernels.basic.MaternKernel``'s own attrs default (``nu=2.5``).
DEFAULT_MATERN_NU = 2.5

# The curated ``linear`` kind builds a degree-1 polynomial kernel (linear
# with a *learned intercept*, i.e. Bayesian linear regression) instead of
# the homogeneous ``LinearKernel`` (``k = v * x1^T x2``). The homogeneous
# form has zero prior variance at the input-space origin, and BayBE
# normalizes continuous parameters so the origin always lies in-bounds:
# the degenerate corner yields NaN acquisition gradients (botorch
# ``OptimizationGradientError``) on otherwise valid campaigns.
LINEAR_KERNEL_POWER = 1

# Human-readable label for BayBE's stock GP kernel: ``DefaultKernelFactory``
# builds ``ScaleKernel(MaternKernel(nu=2.5))`` with dimension-interpolated
# priors. Used by the backend's method metadata whenever no explicit
# kernel/preset is configured.
DEFAULT_KERNEL_DESCRIPTION = "Matern 5/2 (BayBE default GP surrogate)"

# Display names for the Matern smoothness values in MATERN_ALLOWED_NU,
# following the conventional fraction spelling (nu=2.5 -> "Matern 5/2").
MATERN_NU_LABELS: dict[float, str] = {0.5: "1/2", 1.5: "3/2", 2.5: "5/2"}

# Parameterless surrogate factories for the non-GP curated kinds.
_SIMPLE_SURROGATE_FACTORIES: dict[BayBESurrogateKind, type] = {
    BayBESurrogateKind.RANDOM_FOREST: RandomForestSurrogate,
    BayBESurrogateKind.NGBOOST: NGBoostSurrogate,
    BayBESurrogateKind.BAYESIAN_LINEAR: BayesianLinearSurrogate,
    BayBESurrogateKind.MEAN_PREDICTION: MeanPredictionSurrogate,
}


def _build_kernel(config: BayBEKernelConfig) -> ScaleKernel:
    """Build the configured base kernel wrapped in a ScaleKernel.

    The ScaleKernel wrapper mirrors BayBE's own default kernel factory so
    a curated kernel selection changes the similarity structure, not the
    output-scale handling. The ``match`` is exhaustive over
    :class:`BayBEKernelKind` (``assert_never`` makes a missing branch a
    type-check error), and the ``None`` re-checks on the required
    companions turn a violated validator invariant into an immediate,
    clearly-worded failure instead of an opaque library error.
    """
    base: Kernel
    match config.kind:
        case BayBEKernelKind.MATERN:
            nu = config.nu if config.nu is not None else DEFAULT_MATERN_NU
            base = MaternKernel(nu=nu)
        case BayBEKernelKind.RBF:
            base = RBFKernel()
        case BayBEKernelKind.LINEAR:
            # Linear with intercept — see LINEAR_KERNEL_POWER for why the
            # homogeneous LinearKernel is deliberately not used.
            base = PolynomialKernel(power=LINEAR_KERNEL_POWER)
        case BayBEKernelKind.PERIODIC:
            base = PeriodicKernel(period_length_initial_value=config.period_length)
        case BayBEKernelKind.POLYNOMIAL:
            if config.power is None:
                msg = "power is required for the polynomial kernel"
                raise ValueError(msg)
            base = PolynomialKernel(power=config.power)
        case BayBEKernelKind.RQ:
            base = RQKernel()
        case BayBEKernelKind.RFF:
            if config.num_samples is None:
                msg = "num_samples is required for the rff kernel"
                raise ValueError(msg)
            base = RFFKernel(num_samples=config.num_samples)
        case _:
            assert_never(config.kind)
    return ScaleKernel(base)


def _build_gp_surrogate(config: BayBESurrogateConfig) -> GaussianProcessSurrogate:
    """Build the (optionally preset/kernel configured) GP surrogate."""
    kernel = _build_kernel(config.kernel) if config.kernel is not None else None
    if config.gp_preset is not None:
        return GaussianProcessSurrogate.from_preset(
            config.gp_preset.value,
            kernel_or_factory=kernel,
        )
    if kernel is not None:
        return GaussianProcessSurrogate(kernel_or_factory=kernel)
    return GaussianProcessSurrogate()


def build_baybe_surrogate(
    spec: OptimizationSpec,
    options: BayBEBackendOptions,
) -> SurrogateProtocol | None:
    """Build the configured surrogate model, or ``None`` for BayBE's default.

    ``None`` is returned when no surrogate config is set, so the historical
    implicit-default-GP path stays byte-identical. Single-output surrogates
    are replicated per target automatically by BayBE's
    ``BotorchRecommender`` (it wraps them in a ``CompositeSurrogate``), so
    no replication handling is needed here. The neutral
    ``noise_prior_params`` is deliberately NOT consumed — see
    :data:`NOISE_PRIOR_IGNORED_REASON`.
    """
    _ = spec
    config = options.surrogate
    if config is None:
        return None
    if config.kind == BayBESurrogateKind.GP:
        return _build_gp_surrogate(config)
    return _SIMPLE_SURROGATE_FACTORIES[config.kind]()


def _kernel_label(config: BayBEKernelConfig) -> str:
    """Human-readable name of the kernel :func:`_build_kernel` constructs."""
    label: str
    match config.kind:
        case BayBEKernelKind.MATERN:
            nu = config.nu if config.nu is not None else DEFAULT_MATERN_NU
            label = f"Matern {MATERN_NU_LABELS[nu]}"
        case BayBEKernelKind.RBF:
            label = "RBF"
        case BayBEKernelKind.LINEAR:
            label = "Linear (with intercept)"
        case BayBEKernelKind.PERIODIC:
            label = "Periodic"
            if config.period_length is not None:
                label = f"Periodic (period_length={config.period_length})"
        case BayBEKernelKind.POLYNOMIAL:
            label = f"Polynomial (power={config.power})"
        case BayBEKernelKind.RQ:
            label = "Rational Quadratic"
        case BayBEKernelKind.RFF:
            label = f"Random Fourier Features (num_samples={config.num_samples})"
        case _:
            assert_never(config.kind)
    return label


def describe_configured_kernel(options: BayBEBackendOptions) -> str:
    """Kernel label for method/provenance metadata, derived from the config.

    Mirrors what :func:`build_baybe_surrogate` actually constructs so the
    ``method_info['kernel']`` label cannot drift from the fitted kernel:
    an explicit kernel config wins, then a GP preset's own kernel choice,
    then BayBE's stock default; non-GP surrogates have no GP kernel at
    all and say so instead of inheriting the default label.
    """
    surrogate = options.surrogate
    if surrogate is None:
        return DEFAULT_KERNEL_DESCRIPTION
    if surrogate.kind != BayBESurrogateKind.GP:
        return f"not applicable ({surrogate.kind.value} surrogate)"
    if surrogate.kernel is not None:
        label = _kernel_label(surrogate.kernel)
        if surrogate.gp_preset is not None:
            return f"{label} ({surrogate.gp_preset.value} GP preset priors)"
        return label
    if surrogate.gp_preset is not None:
        return f"{surrogate.gp_preset.value} GP preset default kernel"
    return DEFAULT_KERNEL_DESCRIPTION
