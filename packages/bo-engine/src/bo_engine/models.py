"""Gaussian Process model creation and fitting.

Supports:
- SingleTaskGP for single-objective optimization
- ModelListGP for multi-objective optimization
- Optional input warping with Kumaraswamy CDF for non-stationary objectives
- Automatic GPU acceleration when available

Output standardization convention
---------------------------------
All GPs in this module attach ``outcome_transform=Standardize(m=1)``. BoTorch
applies the transform during ``SingleTaskGP.__init__`` (the transform is set to
``train()`` mode and invoked on ``train_Y``), so ``model.train_targets`` holds
the standardized targets. ``fit_gpytorch_mll`` therefore calibrates the output
scale, kernel outputscale and noise against unit-scale targets, and
``model.posterior(X)`` automatically untransforms back to the user's original
scale. Use :func:`verify_standardization` to confirm the convention holds on a
fitted model -- e.g. in tests that rely on scale-invariant behavior.

For multi-objective problems ``create_model`` wraps independent ``SingleTaskGP``
instances in a ``ModelListGP`` so each objective is standardized separately.
This is the mechanism that keeps well-behaved acquisition geometry when
objectives span wildly different magnitudes (e.g. yield ~0.5 vs throughput
~500).

v1.0.1: Added create_single_task_model for single-objective
v1.1: Added input warping support
v2.3: Added GPU auto-detection and acceleration
v2.4: Documented output-standardization convention and added
      ``verify_standardization``.
"""

import logging
import math

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.kernels import CategoricalKernel
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import ChainedInputTransform, Normalize, Warp
from botorch.models.transforms.outcome import (
    ChainedOutcomeTransform,
    Log,
    OutcomeTransform,
    Standardize,
)
from botorch.posteriors import Posterior
from botorch.posteriors.transformed import TransformedPosterior
from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from gpytorch.priors import GammaPrior, Prior
from gpytorch.priors.torch_priors import LogNormalPrior
from torch import Tensor

from bo_engine.constants import (
    NOISE_PRIOR_GAMMA_CONCENTRATION,
    NOISE_PRIOR_GAMMA_RATE,
    NOISE_PRIOR_MIN_INFERRED,
    STANDARDIZATION_MEAN_TOLERANCE,
    STANDARDIZATION_STD_FLOOR,
    STANDARDIZATION_VAR_TOLERANCE,
    WARP_PRIOR_LOC,
    WARP_PRIOR_SCALE,
)
from bo_engine.device import ensure_device, get_device, to_device

logger = logging.getLogger(__name__)


class ModelFittingError(RuntimeError):
    """Raised when GP model fitting fails.

    Contains the original exception and a user-friendly message with recovery guidance.
    """

    def __init__(self, message: str, original_error: Exception) -> None:
        """Record the user-facing message and the underlying optimizer failure."""
        super().__init__(message)
        self.original_error = original_error


def _default_noise_prior() -> GammaPrior:
    """Return the default mildly informative ``GammaPrior`` for GP noise.

    Concentrations come from :mod:`bo_engine.constants` and are calibrated for
    targets that have been standardized to unit variance via
    ``Standardize(m=1)``.
    """
    return GammaPrior(NOISE_PRIOR_GAMMA_CONCENTRATION, NOISE_PRIOR_GAMMA_RATE)


class Negate(OutcomeTransform):
    """Sign-flip outcome transform (its own inverse).

    Exists so the ``Log`` transform can be used under the engine's
    maximization-form convention (see :mod:`bo_engine.types`): a minimize
    objective arrives at the model factory pre-negated, but ``Log`` needs
    the raw positive scale. Chaining ``Negate`` before ``Log`` un-negates
    the targets for the log step and re-negates the untransformed
    posterior, so the model's posterior stays on the (maximization-form)
    scale of the data the caller passed in.

    Although negation is mathematically linear, ``_is_linear`` stays at
    the base-class ``False``: BoTorch's contract for ``_is_linear=True``
    requires ``untransform_posterior`` to map a ``GPyTorchPosterior`` to a
    ``GPyTorchPosterior``, and this transform (like ``Log``) returns a
    ``TransformedPosterior``.
    """

    def subset_output(self, idcs: list[int]) -> OutcomeTransform:  # noqa: ARG002
        """Subset the transform along the output dimension.

        Negation applies uniformly to every output, so the subset is a
        fresh ``Negate``. Implemented so BoTorch's ``subset_model``
        utilities work on chains containing this transform.
        """
        new_tf = self.__class__()
        if not self.training:
            new_tf.eval()
        return new_tf

    def forward(
        self,
        Y: Tensor,  # noqa: N803
        Yvar: Tensor | None = None,  # noqa: N803
        X: Tensor | None = None,  # noqa: N803, ARG002
    ) -> tuple[Tensor, Tensor | None]:
        """Negate targets; observation noise is sign-invariant."""
        return -Y, Yvar

    def untransform(
        self,
        Y: Tensor,  # noqa: N803
        Yvar: Tensor | None = None,  # noqa: N803
        X: Tensor | None = None,  # noqa: N803, ARG002
    ) -> tuple[Tensor, Tensor | None]:
        """Negation is an involution, so untransform is forward."""
        return -Y, Yvar

    def untransform_posterior(
        self,
        posterior: Posterior,
        X: Tensor | None = None,  # noqa: N803, ARG002
    ) -> TransformedPosterior:
        """Negate the posterior mean and samples; variance is unchanged."""
        return TransformedPosterior(
            posterior=posterior,
            sample_transform=torch.neg,
            mean_transform=lambda mean, var: -mean,  # noqa: ARG005
            variance_transform=lambda mean, var: var,  # noqa: ARG005
        )


class DeltaMethodLog(Log):
    """``Log`` outcome transform that also propagates observation noise.

    BoTorch's stock :class:`~botorch.models.transforms.outcome.Log` raises
    ``NotImplementedError`` whenever observation noise (``Yvar``) is
    supplied, which makes the fixed-noise (known measurement-uncertainty)
    GP path incompatible with a log outcome stage. This subclass closes
    the gap with the first-order Taylor (delta method) approximation for
    the variance of a transformed random variable: for ``g(y) = log(y)``,

        ``Var[log Y] ~= g'(y)^2 * Var[Y] = Var[Y] / y^2``

    evaluated at the observed value ``y`` (Casella & Berger, *Statistical
    Inference*, 2nd ed., 2002, §5.5.4 "The Delta Method" -- the standard
    error-propagation rule). ``untransform`` applies the inverse map
    ``Var[Y] ~= exp(z)^2 * Var[Z]`` for ``z = log(y)``, so the pair
    round-trips.

    The approximation is only defined for strictly positive pre-log
    targets. The model factories enforce positivity for ``log_transform``
    objectives before construction (see
    :func:`_assert_positive_for_log_transform`); the transform
    re-validates whenever noise is supplied so standalone use fails with
    an actionable ``ValueError`` instead of emitting non-finite noise.

    ``Yvar`` propagation is implemented for the all-outputs case only
    (the engine chains one transform per single-output sub-model). The
    partial ``outputs`` subset with noise keeps the base class's
    ``NotImplementedError`` behavior.
    """

    def forward(
        self,
        Y: Tensor,  # noqa: N803
        Yvar: Tensor | None = None,  # noqa: N803
        X: Tensor | None = None,  # noqa: N803
    ) -> tuple[Tensor, Tensor | None]:
        """Log-transform targets; delta-method the observation noise."""
        y_tf, _ = super().forward(Y, None, X)
        if Yvar is None:
            return y_tf, None
        self._validate_yvar_supported()
        self._assert_positive_pre_log(Y)
        return y_tf, Yvar / Y.pow(2)

    def untransform(
        self,
        Y: Tensor,  # noqa: N803
        Yvar: Tensor | None = None,  # noqa: N803
        X: Tensor | None = None,  # noqa: N803
    ) -> tuple[Tensor, Tensor | None]:
        """Exponentiate targets; delta-method the noise back to the raw scale."""
        y_utf, _ = super().untransform(Y, None, X)
        if Yvar is None:
            return y_utf, None
        self._validate_yvar_supported()
        return y_utf, Yvar * y_utf.pow(2)

    def _validate_yvar_supported(self) -> None:
        """Reject the partial-``outputs`` + noise corner the base class rejects."""
        if self._outputs is not None:
            msg = (
                "DeltaMethodLog only propagates observation noise when all "
                "outputs are log-transformed; drop the `outputs` subset or "
                "the observation noise."
            )
            raise NotImplementedError(msg)

    @staticmethod
    def _assert_positive_pre_log(Y: Tensor) -> None:  # noqa: N803
        """Raise ``ValueError`` when the delta method's ``y > 0`` premise fails."""
        if not bool((Y > 0).all()):
            min_value = float(Y.min().item())
            msg = (
                "Delta-method noise propagation through the Log outcome "
                f"transform requires strictly positive targets; got "
                f"min={min_value}. Drop or shift non-positive observations "
                "before fitting."
            )
            raise ValueError(msg)


def _build_outcome_transform(
    log_transform: bool,
    *,
    negate_before_log: bool = False,
) -> OutcomeTransform:
    """Build the outcome transform stack for a single objective.

    By default the GP only standardizes targets (mean 0, unit variance).
    When ``log_transform`` is enabled a :class:`DeltaMethodLog` transform
    (a :class:`~botorch.models.transforms.outcome.Log` that additionally
    propagates observation noise via the delta method) is applied first
    via :class:`ChainedOutcomeTransform`, which makes the model behave
    reasonably on multi-decade objectives whose raw scale spans several
    orders of magnitude (e.g. reaction rates or contaminant
    concentrations). BoTorch un-applies both stages on the posterior so
    callers still see results in the scale of the targets they passed in.

    ``negate_before_log`` supports targets supplied in negated
    (maximization-form) shape: a :class:`Negate` stage is chained in front
    of ``Log`` so the log step sees the raw positive scale while the
    untransformed posterior stays on the negated scale of the input. It is
    only meaningful together with ``log_transform=True``; ``Standardize``
    alone is sign-agnostic.

    **Positivity is required.** :class:`Log` operates on the raw target
    values; passing a zero or negative target produces ``-inf`` / ``nan``
    and breaks the downstream ``Standardize`` mean estimate. Callers
    therefore must guarantee strictly positive raw targets for the
    objective when ``log_transform=True``. The model factories
    (:func:`create_single_task_model`, :func:`create_model`) enforce
    this at construction time so the failure mode is a clear
    ``ValueError`` instead of a numerical NaN cascade.

    **Known measurement uncertainty composes with the log stage.** When a
    fixed-noise GP is requested (``train_Yvar`` supplied), ``Negate``
    passes the variance through sign-unchanged, :class:`DeltaMethodLog`
    maps it into log space (``Var[log Y] ~= Var[Y] / y^2``), and
    ``Standardize`` rescales it by its stored ``stdvs**2`` -- so the
    likelihood receives noise in the same standardized-log space as the
    targets.
    """
    if log_transform:
        if negate_before_log:
            return ChainedOutcomeTransform(
                negate=Negate(), log=DeltaMethodLog(), standardize=Standardize(m=1)
            )
        return ChainedOutcomeTransform(log=DeltaMethodLog(), standardize=Standardize(m=1))
    return Standardize(m=1)


# Headroom added on top of ``-min(y)`` when computing the auto-shift; without it
# the shifted target still hits zero exactly and ``Log`` produces ``-inf``.
LOG_AUTO_SHIFT_EPSILON = 1e-6


def compute_log_auto_shift(train_y: Tensor) -> float:
    """Return the additive shift required to make ``train_y`` strictly positive.

    Returns ``0.0`` when the data is already strictly positive (so the
    auto-shift is a no-op and existing campaigns keep their behaviour).
    Otherwise returns ``-min(train_y) + LOG_AUTO_SHIFT_EPSILON`` so the
    smallest observation maps to ``+eps`` after shifting, well above the
    Log transform's ``-inf`` singularity.
    """
    if train_y.numel() == 0:
        return 0.0
    min_value = float(train_y.detach().min().item())
    if min_value > 0:
        return 0.0
    return -min_value + LOG_AUTO_SHIFT_EPSILON


def _assert_positive_for_log_transform(train_y: Tensor, objective_index: int) -> None:
    """Raise ``ValueError`` if any target row is non-positive.

    Enforced at construction time so the failure mode for a
    misconfigured ``log_transform`` campaign is a clear envelope at the
    boundary, not a NaN that propagates through the standardize step
    and surfaces as an opaque BoTorch fit error.
    """
    if not bool(torch.isfinite(train_y).all()):
        msg = (
            f"log_transform=True requires finite targets; objective[{objective_index}] "
            "has NaN or inf values. Drop or impute those rows before fitting."
        )
        raise ValueError(msg)
    if not bool((train_y > 0).all()):
        min_value = float(train_y.min().item())
        msg = (
            f"log_transform=True requires strictly positive targets for "
            f"objective[{objective_index}]; got min={min_value}. Either drop "
            "non-positive observations, pre-shift the target, or disable "
            "log_transform for this objective."
        )
        raise ValueError(msg)


def _prepare_log_targets(
    train_y: Tensor,
    *,
    auto_shift_for_log: bool,
    negate_before_log: bool,
) -> tuple[Tensor, float]:
    """Auto-shift and positivity-check targets destined for a ``Log`` stage.

    ``Log`` operates on the raw positive objective scale. When
    ``negate_before_log`` is set the supplied ``train_y`` is the negation of
    the raw objective (maximization-form convention), so the shift and the
    positivity check are evaluated on ``-train_y`` and the shift moves the
    negated targets in the opposite direction (a raw shift of ``+s`` is
    ``-s`` in negated form).

    Returns:
        Tuple of (possibly shifted ``train_y``, applied raw-scale shift).
    """
    applied_shift = 0.0
    if auto_shift_for_log:
        raw_y = -train_y if negate_before_log else train_y
        applied_shift = compute_log_auto_shift(raw_y)
        if applied_shift > 0:
            logger.warning(
                "log_transform=True with non-positive observations on "
                "objective[0]; applying auto_shift_for_log=%.6g so the Log "
                "transform is well-defined. Posterior predictions are stored "
                "on the shifted scale; subtract %.6g to recover raw values.",
                applied_shift,
                applied_shift,
            )
            train_y = train_y - applied_shift if negate_before_log else train_y + applied_shift

    raw_y = -train_y if negate_before_log else train_y
    _assert_positive_for_log_transform(raw_y, objective_index=0)
    return train_y, applied_shift


def _resolve_per_objective_flags(
    flags: bool | list[bool],
    n_objectives: int,
    name: str,
) -> list[bool]:
    """Broadcast a scalar flag or validate a per-objective flag list."""
    if isinstance(flags, bool):
        return [flags] * n_objectives
    if len(flags) != n_objectives:
        msg = (
            f"{name} list length must match the number of objectives; "
            f"got {len(flags)} flag(s) for {n_objectives} objective(s)."
        )
        raise ValueError(msg)
    return list(flags)


def _build_likelihood(noise_prior: Prior | None) -> GaussianLikelihood:
    """Build a Gaussian likelihood with an explicit, mildly informative prior.

    A floor of ``NOISE_PRIOR_MIN_INFERRED`` is applied via ``GreaterThan`` so
    the inferred noise cannot collapse to zero on multi-scale objectives — a
    known source of singular Cholesky factors during ``fit_gpytorch_mll``.
    """
    prior = noise_prior if noise_prior is not None else _default_noise_prior()
    return GaussianLikelihood(
        noise_prior=prior,
        noise_constraint=GreaterThan(NOISE_PRIOR_MIN_INFERRED),
    )


def _log_fitted_noise(model: SingleTaskGP | ModelListGP, *, fixed_noise: bool) -> None:
    """Emit a debug log of the post-fit noise hyperparameter for drift tracking.

    Targets are standardized, so values close to 0 mean the GP is treating the
    objective as nearly deterministic; values approaching 1 mean noise has
    absorbed most of the signal variance (a silent under-fit symptom).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return

    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    for idx, gp in enumerate(sub_models):
        likelihood = getattr(gp, "likelihood", None)
        if likelihood is None:
            continue
        noise = getattr(likelihood, "noise", None)
        if noise is None:
            continue
        flat = noise.detach().reshape(-1).to(dtype=torch.float64)
        logger.debug(
            "Fitted noise (model=%d, fixed=%s): min=%.4g mean=%.4g max=%.4g",
            idx,
            fixed_noise,
            float(flat.min().item()),
            float(flat.mean().item()),
            float(flat.max().item()),
        )


class _HammingCategoricalKernel(CategoricalKernel):
    """``CategoricalKernel`` whose one-hot block decays as ``exp(-Hamming / ls)``.

    BoTorch's stock :class:`CategoricalKernel` averages the per-column mismatch
    indicators (``delta.mean(-1)``). On a one-hot block a single category change
    flips exactly **two** columns (the old and new "1"), so the stock kernel
    reports a distance of ``2 / k`` (``k`` = number of categories) rather than
    the intended Hamming distance of 1 — the decay then depends on how many
    categories the parameter happens to have, and only coincides with the
    ordinal ``exp(-1 / ls)`` for binary parameters. This subclass divides the
    summed mismatch by two so a category change always maps to distance 1, i.e.
    ``K(catA, catB) = exp(-1 / lengthscale)`` for any ``k``. It keeps the
    one-hot encoding and a single shared lengthscale per block (no per-bit ARD),
    recovering the ordinal Hamming-kernel semantics the docstring promises.

    References:
        - Wan et al., "Think Global and Act Local: Bayesian Optimisation over
          High-Dimensional Categorical and Mixed Search Spaces", ICML 2021 —
          Hamming kernels for categorical / mixed spaces.
        - BoTorch ``CategoricalKernel`` — the averaged one-hot variant this
          compensates.
    """

    def forward(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
    ) -> Tensor:
        """Compute ``exp(-Hamming(x1, x2) / lengthscale)`` over a one-hot block."""
        if last_dim_is_batch:
            # Per-dimension output requested (no cross-column reduction);
            # defer to BoTorch's behavior, which the one-hot 2x factor does
            # not apply to because nothing is summed across the block here.
            return super().forward(x1, x2, diag=diag, last_dim_is_batch=True)
        delta = x1.unsqueeze(-2) != x2.unsqueeze(-3)
        # One-hot toggles two columns per category change, so the summed
        # mismatch is twice the Hamming distance — halve it back to 1.
        dists = (delta / self.lengthscale.unsqueeze(-2)).sum(-1) / 2.0
        res = torch.exp(-dists)
        if diag:
            res = torch.diagonal(res, dim1=-1, dim2=-2)
        return res


def build_mixed_kernel(
    n_total_dims: int,
    categorical_blocks: list[list[int]],
) -> Kernel:
    """Build an additive ``RBF(continuous) + CategoricalKernel(one_hot blocks)`` kernel.

    Used when :attr:`OptimizationSpec.use_categorical_kernel` is set on a spec
    that mixes continuous and one-hot categorical columns. The continuous block
    uses the same ARD ``RBFKernel`` shape that ``SingleTaskGP``'s default kernel
    uses (``ScaleKernel(RBFKernel(ard_num_dims=k))``); each categorical parameter
    gets its **own** :class:`_HammingCategoricalKernel` restricted to that
    parameter's one-hot columns via ``active_dims``. Per-block kernels mean each
    categorical parameter learns a single shared lengthscale (rather than one
    ARD lengthscale per one-hot bit, which over-parameterizes the kernel and
    couples unrelated parameters through a shared mismatch average). All parts
    are summed inside a single ``ScaleKernel`` so the model learns a joint
    outputscale.

    Args:
        n_total_dims: Total number of encoded input dimensions (continuous +
            one-hot categorical columns).
        categorical_blocks: One inner list per categorical parameter, holding
            that parameter's one-hot column indices (see
            :func:`bo_engine.transforms.get_categorical_blocks`).

    Returns:
        A composite :class:`ScaleKernel` ready to be assigned to
        ``SingleTaskGP.covar_module``.

    Raises:
        ValueError: If any categorical index falls outside ``[0, n_total_dims)``.
    """
    flat_indices = [i for block in categorical_blocks for i in block]
    if any(i < 0 or i >= n_total_dims for i in flat_indices):
        msg = (
            "categorical_blocks indices must all lie in "
            f"[0, {n_total_dims}); got {categorical_blocks}."
        )
        raise ValueError(msg)
    categorical_columns = set(flat_indices)
    cont_indices = [i for i in range(n_total_dims) if i not in categorical_columns]

    parts: list[Kernel] = []
    if cont_indices:
        parts.append(
            RBFKernel(
                ard_num_dims=len(cont_indices),
                active_dims=tuple(cont_indices),
            )
        )
    for block in categorical_blocks:
        if not block:
            continue
        # ``ard_num_dims`` omitted → one shared lengthscale for the whole block.
        parts.append(_HammingCategoricalKernel(active_dims=tuple(block)))

    if not parts:
        # Degenerate case (n_total_dims == 0); shouldn't happen for a real spec.
        return ScaleKernel(RBFKernel(ard_num_dims=n_total_dims))

    combined: Kernel = parts[0]
    for kernel in parts[1:]:
        combined = combined + kernel
    return ScaleKernel(combined)


def create_input_transform(
    n_dims: int,
    bounds: Tensor,
    use_input_warping: bool = False,
    *,
    warp_prior_loc: float = WARP_PRIOR_LOC,
    warp_prior_scale: float = WARP_PRIOR_SCALE,
) -> Normalize | ChainedInputTransform:
    """Create input transform with optional warping.

    Transform order is ``normalize → warp``: ``Normalize`` brings raw inputs
    into the unit hypercube, and ``Warp`` (Kumaraswamy CDF) reshapes that
    bounded space to absorb non-stationarity. This ordering matters because
    the Kumaraswamy CDF is defined on ``[0, 1]``; reversing it (warp →
    normalize) requires a custom Kumaraswamy prior calibrated against the
    user's raw input bounds, which is not the case here. The downstream
    ``LogNormalPrior`` on each ``concentration`` therefore assumes the
    *post-normalize* unit-cube domain.

    The ``LogNormalPrior(loc=0, scale=0.75)`` default is the BoTorch stock
    prior. Empirically calibrated alternatives can be passed via
    ``warp_prior_loc`` / ``warp_prior_scale`` for problems where the stock
    prior is too tight (recovers near-identity warps) or too loose (drives
    pathological boundary behaviour). See Eriksson & Snoek 2021 for a
    discussion of warp-prior calibration on non-stationary objectives.

    Args:
        n_dims: Number of input dimensions
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, add Kumaraswamy warping after normalization
        warp_prior_loc: ``loc`` argument for the LogNormal prior on each
            Kumaraswamy concentration parameter. Default matches the BoTorch
            stock prior calibrated for the unit-cube domain.
        warp_prior_scale: ``scale`` argument for the LogNormal prior. Smaller
            values pull both concentrations toward 1 (identity warp); larger
            values allow more pronounced shape transformations.

    Returns:
        Input transform (Normalize or ChainedInputTransform with Warp)
    """
    bounds = to_device(bounds)
    normalize = Normalize(d=n_dims, bounds=bounds)

    if not use_input_warping:
        return normalize

    # Create Kumaraswamy warping with learnable concentration parameters.
    # The prior operates on the post-normalize unit-cube domain (see the
    # docstring above for the chain-order justification).
    device = get_device()
    warp = Warp(
        d=n_dims,
        indices=list(range(n_dims)),
        concentration1_prior=LogNormalPrior(
            loc=torch.tensor(warp_prior_loc, device=device),
            scale=torch.tensor(warp_prior_scale, device=device),
        ),
        concentration0_prior=LogNormalPrior(
            loc=torch.tensor(warp_prior_loc, device=device),
            scale=torch.tensor(warp_prior_scale, device=device),
        ),
    )

    return ChainedInputTransform(
        normalize=normalize,
        warp=warp,
    )


def create_single_task_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool = False,
    *,
    categorical_blocks: list[list[int]] | None = None,
    auto_shift_for_log: bool = False,
    target_negated: bool = False,
) -> SingleTaskGP:
    """Create a SingleTaskGP for single-objective optimization.

    The returned model owns a ``Standardize(m=1)`` outcome transform. BoTorch
    standardizes ``train_Y`` during ``__init__`` so ``fit_gpytorch_mll`` sees
    unit-scale targets, and untransforms the posterior on inference. See the
    module docstring for the full convention.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,).
            Raw, unstandardized targets -- do not pre-standardize.
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation noise variance of shape
            (n_samples, 1) or (n_samples,). When supplied, BoTorch builds a
            ``FixedNoiseGaussianLikelihood`` internally and the noise
            hyperparameter is no longer trainable -- ``noise_prior`` is
            therefore ignored in this branch. Composes with
            ``log_transform=True``: the outcome chain's
            :class:`DeltaMethodLog` stage maps the variance into log space
            via the first-order delta method (``Var[log Y] ~= Var[Y] /
            y^2``), so known measurement uncertainty and log outcomes can
            be used together.
        noise_prior: Optional explicit GPyTorch ``Prior`` on the trainable
            noise hyperparameter. Defaults to a mildly informative
            ``GammaPrior`` calibrated for standardized targets (see
            ``bo_engine.constants``). Only used when ``train_yvar`` is None.
        log_transform: If True, apply ``Log`` before ``Standardize(m=1)`` so
            objectives spanning several orders of magnitude (e.g. reaction
            rates) train against a roughly homoskedastic scale. Requires
            strictly positive targets.
        categorical_blocks: Optional one-hot column indices grouped per
            categorical parameter (see
            :func:`bo_engine.transforms.get_categorical_blocks`). When
            supplied, the GP is built with an additive ``RBF(continuous) +
            CategoricalKernel(one_hot blocks)`` kernel via
            :func:`build_mixed_kernel` so categorical similarity is
            Hamming-style rather than Euclidean on the one-hot expansion.
            ``None`` (default) preserves the historical pure-RBF behavior.
        auto_shift_for_log: Only meaningful when ``log_transform=True``.
            ``True`` instructs the factory to compute a one-shot additive
            shift ``-min(raw_y) + LOG_AUTO_SHIFT_EPSILON`` when any
            training target is non-positive on the raw scale, apply it
            before fitting, and stash the shift on the returned model as
            ``model._auto_shift_for_log`` so downstream callers can back
            it out of posterior predictions. ``False`` (default) preserves
            the legacy "raise on non-positive" behavior.
        target_negated: ``True`` when ``train_y`` is the negation of the
            raw objective (the engine's maximization-form convention for a
            minimize objective; see :mod:`bo_engine.types`). Only consulted
            when ``log_transform=True``: the ``Log`` stage needs the raw
            positive scale, so the outcome chain gains a :class:`Negate`
            stage and the positivity / auto-shift checks run on
            ``-train_y``. ``Standardize`` alone is sign-agnostic.

    Returns:
        SingleTaskGP model (unfitted)
    """
    if train_yvar is not None:
        train_x, train_y, train_yvar, bounds = ensure_device(train_x, train_y, train_yvar, bounds)
    else:
        train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    # Ensure train_y has correct shape (n_samples, 1)
    if train_y.dim() == 1:
        train_y = train_y.unsqueeze(-1)
    if train_yvar is not None and train_yvar.dim() == 1:
        train_yvar = train_yvar.unsqueeze(-1)

    negate_before_log = log_transform and target_negated

    applied_shift = 0.0
    if log_transform:
        train_y, applied_shift = _prepare_log_targets(
            train_y,
            auto_shift_for_log=auto_shift_for_log,
            negate_before_log=negate_before_log,
        )

    input_transform = create_input_transform(
        n_dims=train_x.shape[-1],
        bounds=bounds,
        use_input_warping=use_input_warping,
    )

    kwargs: dict = {
        "train_X": train_x,
        "train_Y": train_y,
        "input_transform": input_transform,
        "outcome_transform": _build_outcome_transform(
            log_transform, negate_before_log=negate_before_log
        ),
    }
    if categorical_blocks:
        kwargs["covar_module"] = build_mixed_kernel(
            n_total_dims=int(train_x.shape[-1]),
            categorical_blocks=list(categorical_blocks),
        )
    if train_yvar is not None:
        # Heteroskedastic / known-uncertainty path. BoTorch routes Yvar
        # through a FixedNoiseGaussianLikelihood, so the noise hyperparameter
        # is non-trainable and any ``noise_prior`` would have no effect.
        kwargs["train_Yvar"] = train_yvar
    else:
        kwargs["likelihood"] = _build_likelihood(noise_prior)

    model = SingleTaskGP(**kwargs)
    # Stash the applied shift so downstream callers (suggestion provenance,
    # diagnostics) can subtract it before reporting predictions in the
    # user-facing scale.
    if applied_shift > 0:
        model._auto_shift_for_log = applied_shift  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
    return model


def create_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool | list[bool] = False,
    *,
    categorical_blocks: list[list[int]] | None = None,
    target_negated: bool | list[bool] = False,
) -> ModelListGP:
    """Create a ModelListGP for multi-objective optimization.

    Uses independent GP models for each objective. Each per-objective model
    owns its own ``Standardize(m=1)`` outcome transform, so objectives with
    very different magnitudes (e.g. yield ~0.5 vs throughput ~500) are still
    fit against unit-scale targets.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives).
            Raw, unstandardized targets -- do not pre-standardize.
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-objective noise variance of shape
            (n_samples, n_objectives). When supplied, every sub-model is
            built with a ``FixedNoiseGaussianLikelihood`` and ``noise_prior``
            is ignored. Sub-models with ``log_transform`` enabled map their
            variance column into log space via :class:`DeltaMethodLog`
            (first-order delta method).
        noise_prior: Optional GPyTorch ``Prior`` shared across sub-models.
            Defaults to a mildly informative ``GammaPrior`` for standardized
            targets. Only used when ``train_yvar`` is None.
        log_transform: If True, applies ``Log`` before ``Standardize`` on
            every sub-model. Pass a list aligned with the objective
            dimension to enable it per-objective; objectives with
            multi-decade magnitudes (e.g. concentrations) typically
            benefit while bounded ones (yield in [0, 1]) do not.
        categorical_blocks: One-hot column indices grouped per categorical
            parameter in ``train_x``. Used to route each block through its own
            ``CategoricalKernel`` instead of the default Matérn.
        target_negated: ``True`` for columns whose targets are the negation
            of the raw objective (maximization-form convention for minimize
            objectives; see :mod:`bo_engine.types`). Accepts a single bool
            or a per-objective list. Only consulted for objectives with
            ``log_transform`` enabled, where the ``Log`` stage needs the
            raw positive scale (see :class:`Negate`).

    Returns:
        ModelListGP with one GP per objective
    """
    if train_yvar is not None:
        train_x, train_y, train_yvar, bounds = ensure_device(train_x, train_y, train_yvar, bounds)
    else:
        train_x, train_y, bounds = ensure_device(train_x, train_y, bounds)

    n_objectives = train_y.shape[-1]
    n_dims = train_x.shape[-1]

    log_flags = _resolve_per_objective_flags(log_transform, n_objectives, "log_transform")
    negated_flags = _resolve_per_objective_flags(target_negated, n_objectives, "target_negated")

    models = []
    for i in range(n_objectives):
        # Create input transform (each model gets its own)
        input_transform = create_input_transform(
            n_dims=n_dims,
            bounds=bounds,
            use_input_warping=use_input_warping,
        )

        negate_before_log = log_flags[i] and negated_flags[i]
        if log_flags[i]:
            raw_column = -train_y[:, i : i + 1] if negate_before_log else train_y[:, i : i + 1]
            _assert_positive_for_log_transform(raw_column, objective_index=i)

        kwargs: dict = {
            "train_X": train_x,
            "train_Y": train_y[:, i : i + 1],
            "input_transform": input_transform,
            "outcome_transform": _build_outcome_transform(
                log_flags[i], negate_before_log=negate_before_log
            ),
        }
        if categorical_blocks:
            kwargs["covar_module"] = build_mixed_kernel(
                n_total_dims=n_dims,
                categorical_blocks=list(categorical_blocks),
            )
        if train_yvar is not None:
            kwargs["train_Yvar"] = train_yvar[:, i : i + 1]
        else:
            kwargs["likelihood"] = _build_likelihood(noise_prior)

        models.append(SingleTaskGP(**kwargs))

    return ModelListGP(*models)


def fit_single_task_model(model: SingleTaskGP) -> SingleTaskGP:
    """Fit a SingleTaskGP by maximizing marginal likelihood.

    Args:
        model: SingleTaskGP to fit

    Returns:
        Fitted model

    Raises:
        ModelFittingError: If fitting fails (singular matrix, numerical instability, etc.)
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except (RuntimeError, torch.linalg.LinAlgError) as e:
        msg = (
            f"GP model fitting failed: {e}. "
            "Consider adding more observations or reducing parameter count."
        )
        logger.exception(msg)
        raise ModelFittingError(msg, original_error=e) from e
    _log_fitted_noise(model, fixed_noise=_has_fixed_noise(model))
    return model


def fit_model(model: ModelListGP) -> ModelListGP:
    """Fit the GP model by maximizing marginal likelihood.

    Args:
        model: ModelListGP to fit

    Returns:
        Fitted model

    Raises:
        ModelFittingError: If fitting fails (singular matrix, numerical instability, etc.)
    """
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    try:
        fit_gpytorch_mll(mll)
    except (RuntimeError, torch.linalg.LinAlgError) as e:
        msg = (
            f"Multi-output GP model fitting failed: {e}. "
            "Consider adding more observations or reducing parameter count."
        )
        logger.exception(msg)
        raise ModelFittingError(msg, original_error=e) from e
    _log_fitted_noise(model, fixed_noise=_has_fixed_noise(model))
    return model


def _has_fixed_noise(model: SingleTaskGP | ModelListGP) -> bool:
    """Return True iff any sub-model uses a FixedNoiseGaussianLikelihood."""
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]
    for gp in sub_models:
        likelihood = getattr(gp, "likelihood", None)
        if likelihood is None:
            continue
        # FixedNoiseGaussianLikelihood exposes `noise_covar` of type FixedGaussianNoise
        noise_covar = getattr(likelihood, "noise_covar", None)
        if noise_covar is not None and not hasattr(noise_covar, "raw_noise"):
            return True
    return False


def create_and_fit_single_task_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool = False,
    *,
    categorical_blocks: list[list[int]] | None = None,
    auto_shift_for_log: bool = False,
    target_negated: bool = False,
) -> SingleTaskGP:
    """Create and fit a SingleTaskGP.

    Convenience function for single-objective optimization.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, 1) or (n_samples,)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation noise variance. See
            :func:`create_single_task_model`.
        noise_prior: Optional GPyTorch prior on the trainable noise
            hyperparameter. See :func:`create_single_task_model`.
        log_transform: Forwarded to :func:`create_single_task_model`.
        categorical_blocks: Forwarded to :func:`create_single_task_model`.
        auto_shift_for_log: Forwarded to :func:`create_single_task_model`.
        target_negated: Forwarded to :func:`create_single_task_model`.

    Returns:
        Fitted SingleTaskGP
    """
    model = create_single_task_model(
        train_x,
        train_y,
        bounds,
        use_input_warping=use_input_warping,
        train_yvar=train_yvar,
        noise_prior=noise_prior,
        log_transform=log_transform,
        categorical_blocks=categorical_blocks,
        auto_shift_for_log=auto_shift_for_log,
        target_negated=target_negated,
    )
    return fit_single_task_model(model)


def create_and_fit_model(
    train_x: Tensor,
    train_y: Tensor,
    bounds: Tensor,
    use_input_warping: bool = False,
    train_yvar: Tensor | None = None,
    noise_prior: Prior | None = None,
    log_transform: bool | list[bool] = False,
    *,
    categorical_blocks: list[list[int]] | None = None,
    target_negated: bool | list[bool] = False,
) -> ModelListGP:
    """Create and fit a ModelListGP.

    Convenience function combining create_model and fit_model.

    Args:
        train_x: Training inputs of shape (n_samples, n_dims)
        train_y: Training outputs of shape (n_samples, n_objectives)
        bounds: Parameter bounds of shape (2, n_dims)
        use_input_warping: If True, use Kumaraswamy input warping
        train_yvar: Optional per-observation per-objective noise variance.
            See :func:`create_model`.
        noise_prior: Optional GPyTorch prior shared across sub-models. See
            :func:`create_model`.
        log_transform: Per-objective ``Log → Standardize`` opt-in.
            Forwarded to :func:`create_model`.
        categorical_blocks: Forwarded to :func:`create_model`.
        target_negated: Forwarded to :func:`create_model`.

    Returns:
        Fitted ModelListGP
    """
    model = create_model(
        train_x,
        train_y,
        bounds,
        use_input_warping=use_input_warping,
        train_yvar=train_yvar,
        noise_prior=noise_prior,
        log_transform=log_transform,
        categorical_blocks=categorical_blocks,
        target_negated=target_negated,
    )
    return fit_model(model)


def get_model_predictions(
    model: ModelListGP | SingleTaskGP,
    test_x: Tensor,
) -> tuple[Tensor, Tensor]:
    """Get model predictions with uncertainty.

    Args:
        model: Fitted ModelListGP or SingleTaskGP
        test_x: Test inputs of shape (n_test, n_dims)

    Returns:
        Tuple of (mean, variance) each of shape (n_test, n_objectives) or (n_test, 1)
    """
    test_x = to_device(test_x)
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_x)
        mean = posterior.mean
        variance = posterior.variance

    # Reshape from (n_test, n_objectives, 1) to (n_test, n_objectives)
    mean = mean.squeeze(-1)
    variance = variance.squeeze(-1)

    return mean, variance


def extract_lengthscales(model: ModelListGP | SingleTaskGP) -> dict[int, Tensor]:
    """Extract lengthscales from model for feature importance analysis.

    For ARD kernels, lengthscales indicate relative importance of parameters.
    Smaller lengthscale = more important (more sensitive to changes).

    Args:
        model: Fitted GP model

    Returns:
        Dictionary mapping objective index to lengthscale tensor
    """
    lengthscales = {}

    if isinstance(model, SingleTaskGP):
        # Single model - access through covar_module attribute
        covar = model.covar_module  # type: ignore[attr-defined]
        ls = covar.base_kernel.lengthscale.detach()  # ty: ignore[call-non-callable, unresolved-attribute]
        lengthscales[0] = ls.squeeze()
    else:
        # ModelListGP
        for i, m in enumerate(model.models):
            covar = m.covar_module  # type: ignore[attr-defined]
            ls = covar.base_kernel.lengthscale.detach()
            lengthscales[i] = ls.squeeze()

    return lengthscales


def get_warping_parameters(model: SingleTaskGP) -> dict[str, Tensor] | None:
    """Extract warping parameters if input warping is used.

    Args:
        model: SingleTaskGP that may have input warping

    Returns:
        Dictionary with 'concentration0' and 'concentration1' tensors, or None
    """
    input_transform = model.input_transform

    # Check if it's a ChainedInputTransform with warping
    if isinstance(input_transform, ChainedInputTransform):
        for transform in input_transform.values():
            if isinstance(transform, Warp):
                c0 = transform.concentration0  # type: ignore[attr-defined]
                c1 = transform.concentration1  # type: ignore[attr-defined]
                return {
                    "concentration0": c0.detach(),  # ty: ignore[call-non-callable]
                    "concentration1": c1.detach(),  # ty: ignore[call-non-callable]
                }

    return None


def verify_standardization(
    model: ModelListGP | SingleTaskGP,
    *,
    mean_atol: float = STANDARDIZATION_MEAN_TOLERANCE,
    var_atol: float = STANDARDIZATION_VAR_TOLERANCE,
) -> list[dict[str, float]]:
    """Verify that BoTorch internalized ``Standardize`` on each sub-model.

    Checks that every GP's stored ``train_targets`` (what MLL is computed
    against) has ~zero mean and ~unit variance -- the invariant that
    downstream acquisition geometry, noise-prior calibration and TuRBO
    tolerances depend on. Logs a warning if the invariant does not hold
    within the given tolerances.

    Args:
        model: Fitted ``SingleTaskGP`` or ``ModelListGP``.
        mean_atol: Absolute tolerance on the standardized-target mean.
        var_atol: Absolute tolerance on ``|var(train_targets) - 1|``. Only
            asserted when the sub-model has at least two observations
            (variance is ill-defined for a single point).

    Returns:
        One diagnostic dict per sub-model with keys ``mean``, ``var``,
        ``n`` and ``standardized`` (bool).

    Raises:
        ValueError: If ``model`` has no ``train_targets`` (not an ExactGP).
    """
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    reports: list[dict[str, float]] = []
    for idx, gp in enumerate(sub_models):
        targets = getattr(gp, "train_targets", None)
        if targets is None:
            msg = f"Model {idx} has no train_targets -- cannot verify standardization."
            raise ValueError(msg)

        flat = targets.detach().reshape(-1).to(dtype=torch.float64)
        n = int(flat.numel())
        mean = float(flat.mean().item())
        # Sample (unbiased) variance to match BoTorch's `nanstd` normalization
        # -- it divides by n-1, so the post-standardization unbiased variance
        # is exactly 1.0 up to float jitter.
        var = float(flat.var(unbiased=True).item()) if n > 1 else float("nan")

        mean_ok = abs(mean) <= mean_atol
        var_ok = n <= 1 or abs(var - 1.0) <= var_atol
        standardized = mean_ok and var_ok
        if not standardized:
            logger.warning(
                "Model %d targets not unit-scale (mean=%.4g, var=%.4g, n=%d); "
                "outcome_transform may not be attached.",
                idx,
                mean,
                var,
                n,
            )

        reports.append(
            {
                "mean": mean,
                "var": var,
                "n": float(n),
                "standardized": float(standardized),
            }
        )

    return reports


def inspect_standardize_stdvs(
    model: ModelListGP | SingleTaskGP,
) -> list[float]:
    """Return the per-sub-model standardization stddev recorded by ``Standardize``.

    Read-only inspection helper used by :func:`post_fit_verification` to
    surface near-constant training data as a batch warning. Mutating
    ``Standardize.stdvs`` in place would break the forward/inverse
    transform round-trip — the GP was fit with one divisor and the
    posterior is un-transformed with a different one — so we no longer
    floor the stored value. Numerical instabilities from a tiny stddev
    surface to the caller through the warning path instead.

    Args:
        model: Fitted ``SingleTaskGP`` or ``ModelListGP``.

    Returns:
        Per-sub-model minimum stddev across the ``Standardize.stdvs``
        tensor; ``nan`` for sub-models that don't expose a Standardize
        transform.
    """
    if isinstance(model, ModelListGP):
        sub_models: list[SingleTaskGP] = list(model.models)  # ty: ignore[invalid-argument-type]
    else:
        sub_models = [model]

    raw_stds: list[float] = []
    for gp in sub_models:
        transform = getattr(gp, "outcome_transform", None)
        # ChainedOutcomeTransform exposes its children by attribute name; the
        # standardize step is canonically registered as "standardize".
        standardize = transform
        if transform is not None and hasattr(transform, "standardize"):
            standardize = transform.standardize
        stdvs = getattr(standardize, "stdvs", None)
        if stdvs is None:
            raw_stds.append(float("nan"))
            continue
        raw_stds.append(float(stdvs.detach().reshape(-1).min().item()))
    return raw_stds


def floor_standardize_stdvs(
    model: ModelListGP | SingleTaskGP,
) -> list[float]:
    """Deprecated alias of :func:`inspect_standardize_stdvs`.

    The previous implementation clamped ``Standardize.stdvs`` in place
    after fit, which broke the forward/inverse round-trip on
    near-constant data (the GP was fit with the tiny pre-clamp stddev,
    then the posterior was un-transformed with the floored value,
    distorting predictions on the user-facing scale). The function now
    only reports the stored stddev; downstream code should rely on
    :func:`post_fit_verification` warnings to flag near-constant data.
    """
    return inspect_standardize_stdvs(model)


def _label_for_subindex(idx: int, objective_names: list[str] | None) -> str:
    """Return the human-readable label used in standardization warnings."""
    if objective_names and idx < len(objective_names):
        return f"objective '{objective_names[idx]}'"
    return f"sub-model {idx}"


def post_fit_verification(
    model: ModelListGP | SingleTaskGP,
    *,
    objective_names: list[str] | None = None,
    std_floor: float = STANDARDIZATION_STD_FLOOR,
) -> list[str]:
    """Run the post-fit standardization checks and return human-readable warnings.

    Combines :func:`verify_standardization` (the invariant check) and
    :func:`inspect_standardize_stdvs` (the numerical inspection) so the
    suggestion pipeline can surface both signals to the caller as
    :attr:`bo_engine.backend.SuggestionBatch.warnings` entries.

    **No mutation.** A previous version of this helper clamped
    ``Standardize.stdvs`` in place after fit. That broke the forward /
    inverse transform round-trip on near-constant data — the GP was fit
    on standardized targets computed with the raw tiny stddev, but the
    posterior was un-transformed with the floored stddev. We now only
    warn; numerical instability from a near-constant column is the
    caller's responsibility to act on (drop the row, add jitter, or
    rerun with a different objective).

    Args:
        model: Fitted GP (or model list).
        objective_names: Optional names used to attribute warnings to a
            specific objective. When ``None``, warnings reference the
            sub-model index instead.
        std_floor: Threshold below which a sub-model's stddev triggers a
            warning. Defaults to :data:`STANDARDIZATION_STD_FLOOR`.

    Returns:
        List of warning strings (empty when every sub-model passes).
    """
    warnings_list: list[str] = []
    raw_stds = inspect_standardize_stdvs(model)
    for idx, raw_std in enumerate(raw_stds):
        if math.isnan(raw_std) or raw_std >= std_floor:
            continue
        label = _label_for_subindex(idx, objective_names)
        warnings_list.append(
            f"GP for {label} fit on near-constant data: raw stddev "
            f"{raw_std:.3e} is below the recommended floor {std_floor:.0e}. "
            "Posterior predictions may be numerically unstable; verify the "
            "objective is not a replicate-only column or add jitter to the "
            "training data."
        )

    reports = verify_standardization(model)
    for idx, report in enumerate(reports):
        if report["standardized"] >= 1.0:
            continue
        label = _label_for_subindex(idx, objective_names)
        warnings_list.append(
            f"Standardization invariant failed for {label}: "
            f"mean={report['mean']:.3g}, var={report['var']:.3g}. Posterior "
            "predictions may be on an unexpected scale; check that the "
            "outcome transform attached cleanly."
        )
    return warnings_list
