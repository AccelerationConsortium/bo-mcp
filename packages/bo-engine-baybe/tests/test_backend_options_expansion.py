"""Recommender / surrogate / campaign-toggle options on the BayBE backend.

Covers the typed ``backend_options['baybe']`` growth: initial-recommender
selection (FPS / clustering), BotorchRecommender tuning knobs, the curated
surrogate/kernel/preset surface with its build-and-catch intake guard,
fit-level coverage of every kernel kind (construction alone cannot catch
kernels that only fail during GP fitting or acquisition optimization),
the derived ``method_info['kernel']`` labels, the campaign-level
``allow_recommending_*`` / ``measurements_must_be_within_tolerance``
toggles, and campaign persistence round-trips per newly exposed
recommender/surrogate.

References: BayBE recommenders userguide
(https://emdgroup.github.io/baybe/stable/userguide/recommenders.html) and
surrogates userguide
(https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
import pytest
from baybe import Campaign
from baybe.kernels import PeriodicKernel, PolynomialKernel, RFFKernel, RQKernel, ScaleKernel
from baybe.recommenders import (
    BotorchRecommender,
    FPSRecommender,
    KMeansClusteringRecommender,
    TwoPhaseMetaRecommender,
)
from baybe.surrogates import CompositeSurrogate, GaussianProcessSurrogate

from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.constants import MAX_RFF_NUM_SAMPLES
from bo_engine_baybe.options import (
    KERNEL_COMPANION_FIELDS,
    MATERN_ALLOWED_NU,
    BayBEBackendOptions,
    BayBEInitialRecommender,
    BayBEKernelConfig,
    BayBEKernelKind,
    BayBESurrogateConfig,
    BayBESurrogateKind,
    extract_baybe_backend_options,
)
from bo_engine_baybe.state import _build_campaign
from bo_engine_baybe.surrogates import (
    DEFAULT_KERNEL_DESCRIPTION,
    LINEAR_KERNEL_POWER,
    MATERN_NU_LABELS,
    _build_kernel,
    build_baybe_surrogate,
    describe_configured_kernel,
)


def _meta(campaign: Campaign) -> TwoPhaseMetaRecommender:
    """Narrow the campaign's recommender to the two-phase meta graph."""
    recommender = campaign.recommender
    assert isinstance(recommender, TwoPhaseMetaRecommender)
    return recommender


def _bo_recommender(campaign: Campaign) -> BotorchRecommender:
    """Narrow the meta graph's GP-phase recommender."""
    recommender = _meta(campaign).recommender
    assert isinstance(recommender, BotorchRecommender)
    return recommender


_DISCRETE_PARAMS = [
    ParameterSpec(name="a", type=ParameterType.DISCRETE, values=[0.0, 1.0, 2.0]),
    ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
]
_CONTINUOUS_PARAMS = [
    ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
]

# One buildable payload per kernel kind, keyed by the enum: the coverage
# test below fails the moment a new BayBEKernelKind member lands without
# fit-level test wiring here.
_KERNEL_PAYLOADS: dict[BayBEKernelKind, dict[str, Any]] = {
    BayBEKernelKind.MATERN: {"kind": "matern", "nu": 1.5},
    BayBEKernelKind.RBF: {"kind": "rbf"},
    BayBEKernelKind.LINEAR: {"kind": "linear"},
    BayBEKernelKind.PERIODIC: {"kind": "periodic", "period_length": 1.5},
    BayBEKernelKind.POLYNOMIAL: {"kind": "polynomial", "power": 2},
    BayBEKernelKind.RQ: {"kind": "rq"},
    BayBEKernelKind.RFF: {"kind": "rff", "num_samples": 32},
}

# ``method_info['kernel']`` label expected for each payload above.
_EXPECTED_KERNEL_LABELS: dict[BayBEKernelKind, str] = {
    BayBEKernelKind.MATERN: "Matern 3/2",
    BayBEKernelKind.RBF: "RBF",
    BayBEKernelKind.LINEAR: "Linear (with intercept)",
    BayBEKernelKind.PERIODIC: "Periodic (period_length=1.5)",
    BayBEKernelKind.POLYNOMIAL: "Polynomial (power=2)",
    BayBEKernelKind.RQ: "Rational Quadratic",
    BayBEKernelKind.RFF: "Random Fourier Features (num_samples=32)",
}

# Kinds built on gpytorch classes with ``has_lengthscale = False``: their
# introspection reports a learned ``kernel_offset`` instead of
# ``lengthscales`` (https://docs.gpytorch.ai/en/stable/kernels.html).
_LENGTHSCALE_FREE_KINDS = frozenset({BayBEKernelKind.LINEAR, BayBEKernelKind.POLYNOMIAL})

# Kernel-specific learned hyperparameters that introspection must surface
# per kind (gpytorch parameter names: polynomial/linear ``offset``,
# periodic ``period_length``, rational-quadratic ``alpha``).
_EXPECTED_EXTRA_HYPERPARAMS: dict[BayBEKernelKind, str] = {
    BayBEKernelKind.LINEAR: "kernel_offset",
    BayBEKernelKind.POLYNOMIAL: "kernel_offset",
    BayBEKernelKind.PERIODIC: "kernel_period_length",
    BayBEKernelKind.RQ: "kernel_alpha",
}


def _spec(
    parameters: list[ParameterSpec],
    baybe_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=parameters,
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        backend_options={"baybe": baybe_options} if baybe_options is not None else None,
        **kwargs,
    )


class TestOptionsValidation:
    def test_round_trip_of_new_fields(self) -> None:
        options = BayBEBackendOptions.model_validate(
            {
                "recommender": {
                    "switch_after": 3,
                    "initial_recommender": "fps",
                    "bayesian": {"n_restarts": 20, "sampling_percentage": 0.5},
                },
                "surrogate": {"kind": "gp", "kernel": {"kind": "matern", "nu": 1.5}},
                "allow_recommending_already_measured": True,
                "measurements_must_be_within_tolerance": False,
            }
        )
        assert options.recommender is not None
        assert options.recommender.initial_recommender == BayBEInitialRecommender.FPS
        assert options.surrogate is not None
        assert options.surrogate.kernel is not None
        assert options.surrogate.kernel.nu == pytest.approx(1.5)

    def test_invalid_nu_rejected(self) -> None:
        with pytest.raises(ValueError, match="matern nu must be one of"):
            BayBEKernelConfig(kind=BayBEKernelKind.MATERN, nu=3.0)

    def test_nu_on_rbf_rejected(self) -> None:
        with pytest.raises(ValueError, match="nu is only valid for the matern kernel"):
            BayBEKernelConfig(kind=BayBEKernelKind.RBF, nu=2.5)

    def test_kernel_on_non_gp_rejected(self) -> None:
        with pytest.raises(ValueError, match="only valid with kind='gp'"):
            BayBESurrogateConfig(
                kind=BayBESurrogateKind.RANDOM_FOREST,
                kernel=BayBEKernelConfig(kind=BayBEKernelKind.RBF),
            )

    @pytest.mark.parametrize(
        ("payload", "match"),
        [
            ({"kind": "rbf", "period_length": 2.0}, "period_length is only valid"),
            ({"kind": "rbf", "power": 2}, "power is only valid"),
            ({"kind": "polynomial"}, "power is required"),
            ({"kind": "rbf", "num_samples": 100}, "num_samples is only valid"),
            ({"kind": "rff"}, "num_samples is required"),
            ({"kind": "polynomial", "power": 0}, "greater than or equal to 1"),
        ],
    )
    def test_kernel_companion_field_rules(self, payload: dict[str, Any], match: str) -> None:
        """Companion parameters are rejected off-kind and required on-kind.

        The required/optional split mirrors the installed BayBE kernel API
        (``baybe.kernels.basic``, pinned at 0.15.0): ``PolynomialKernel.power``
        and ``RFFKernel.num_samples`` are mandatory attrs fields while
        ``PeriodicKernel.period_length_initial_value`` defaults to ``None``
        (https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
        ``power=0`` is additionally rejected at intake even though BayBE
        permits it: gpytorch's polynomial kernel ``(x1 . x2 + c)^power``
        degenerates to a constant covariance at degree 0 — a GP that ignores
        its inputs and reduces the recommender to random sampling
        (https://docs.gpytorch.ai/en/stable/kernels.html).
        """
        with pytest.raises(ValueError, match=match):
            BayBEKernelConfig.model_validate(payload)

    def test_companion_table_covers_all_config_fields(self) -> None:
        """Every kernel parameter is owned by exactly one kind in the rules table."""
        assert set(KERNEL_COMPANION_FIELDS) == set(BayBEKernelConfig.model_fields) - {"kind"}

    def test_switch_after_unset_stays_none(self) -> None:
        """Omitted switch_after must be distinguishable from an explicit value.

        ``None`` marks "not set" so the phase resolver can apply the
        documented precedence (explicit `switch_after` > neutral
        `initial_design_size` > backend default) instead of letting the
        field default masquerade as a user choice whenever any other
        recommender sub-option is present.
        """
        options = BayBEBackendOptions.model_validate(
            {"recommender": {"initial_recommender": "fps"}}
        )
        assert options.recommender is not None
        assert options.recommender.switch_after is None

    def test_rff_num_samples_bounded(self) -> None:
        """num_samples accepts the cap and rejects cap+1 at intake.

        gpytorch's RFFKernel draws a ``randn(dims, num_samples)`` weight
        buffer at fit time and featurizes every evaluated point into
        ``2*num_samples`` columns
        (https://docs.gpytorch.ai/en/stable/kernels.html), so the intake
        cap is the guard against memory-exhausting payload values. The
        accepted boundary builds the real kernel object, not a mock.
        """
        config = BayBEKernelConfig(kind=BayBEKernelKind.RFF, num_samples=MAX_RFF_NUM_SAMPLES)
        kernel = _build_kernel(config)
        assert isinstance(kernel.base_kernel, RFFKernel)
        assert kernel.base_kernel.num_samples == MAX_RFF_NUM_SAMPLES
        with pytest.raises(ValueError, match="less than or equal to"):
            BayBEKernelConfig(kind=BayBEKernelKind.RFF, num_samples=MAX_RFF_NUM_SAMPLES + 1)


class TestRecommenderSelection:
    def test_fps_initial_recommender_is_built(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"recommender": {"initial_recommender": "fps"}})
        campaign = _build_campaign(spec)
        assert isinstance(_meta(campaign).initial_recommender, FPSRecommender)

    def test_kmeans_initial_recommender_is_built(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"recommender": {"initial_recommender": "kmeans"}})
        campaign = _build_campaign(spec)
        assert isinstance(_meta(campaign).initial_recommender, KMeansClusteringRecommender)

    def test_bayesian_knobs_reach_botorch_recommender(self) -> None:
        spec = _spec(
            _DISCRETE_PARAMS,
            {"recommender": {"bayesian": {"n_restarts": 21, "n_raw_samples": 99}}},
        )
        campaign = _build_campaign(spec)
        bo_rec = _bo_recommender(campaign)
        assert bo_rec.n_restarts == 21
        assert bo_rec.n_raw_samples == 99

    def test_non_random_initial_recommender_rejected_on_continuous_space(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS, {"recommender": {"initial_recommender": "fps"}})
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert any("initial_recommender" in r.key for r in result.unsupported)

    def test_random_initial_recommender_fine_on_continuous_space(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS, {"recommender": {"initial_recommender": "random"}})
        assert BayBEBackend().validate_capabilities(spec).is_compatible


class TestSurrogateSelection:
    def test_default_stays_none(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS)
        options = BayBEBackendOptions()
        assert build_baybe_surrogate(spec, options) is None

    def test_matern_kernel_gp(self) -> None:
        spec = _spec(
            _CONTINUOUS_PARAMS,
            {"surrogate": {"kind": "gp", "kernel": {"kind": "matern", "nu": 0.5}}},
        )
        options = extract_baybe_backend_options(spec.backend_options)
        assert isinstance(build_baybe_surrogate(spec, options), GaussianProcessSurrogate)
        # The campaign wraps it (CompositeSurrogate replication) but builds.
        assert _build_campaign(spec) is not None

    def test_linear_kernel_is_linear_with_intercept(self) -> None:
        """'linear' builds a degree-1 polynomial: linear plus learned intercept.

        The homogeneous linear kernel ``k(x, x') = v * x^T x'`` has zero
        prior variance at the input-space origin — the dot-product kernel
        needs the ``sigma_0^2`` offset term to model an intercept
        (Rasmussen & Williams 2006, GPML section 4.2.2,
        https://gaussianprocess.org/gpml/) — and BayBE normalizes
        continuous parameters so the origin always lies in-bounds; the
        pinned corner yields NaN acquisition gradients on misspecified
        data. gpytorch's ``PolynomialKernel`` with ``power=1`` is exactly
        the inhomogeneous (offset) form
        (https://docs.gpytorch.ai/en/stable/kernels.html).
        """
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.LINEAR))
        assert isinstance(kernel, ScaleKernel)
        assert isinstance(kernel.base_kernel, PolynomialKernel)
        assert kernel.base_kernel.power == LINEAR_KERNEL_POWER

    def test_periodic_kernel_default(self) -> None:
        """Omitted period_length defers to gpytorch's own initial value.

        ``baybe.kernels.basic.PeriodicKernel.period_length_initial_value``
        is an optional attrs field defaulting to ``None`` (BayBE 0.15.0;
        https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
        """
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.PERIODIC))
        assert isinstance(kernel.base_kernel, PeriodicKernel)
        assert kernel.base_kernel.period_length_initial_value is None

    def test_periodic_kernel_explicit_period_length(self) -> None:
        """An explicit period_length must round-trip into the built kernel."""
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.PERIODIC, period_length=3.5))
        assert isinstance(kernel.base_kernel, PeriodicKernel)
        assert kernel.base_kernel.period_length_initial_value == pytest.approx(3.5)

    def test_polynomial_kernel(self) -> None:
        """power reaches the built kernel (mandatory attrs field in BayBE 0.15.0)."""
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.POLYNOMIAL, power=3))
        assert isinstance(kernel.base_kernel, PolynomialKernel)
        assert kernel.base_kernel.power == 3

    def test_rq_kernel(self) -> None:
        """The rational quadratic kind needs no companion parameters."""
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.RQ))
        assert isinstance(kernel.base_kernel, RQKernel)

    def test_rff_kernel(self) -> None:
        """num_samples (the random Fourier feature count) reaches the kernel.

        Random Fourier features approximate a stationary kernel with a
        finite spectral sample (Rahimi & Recht 2007, "Random Features for
        Large-Scale Kernel Machines", NeurIPS 20); the count is a mandatory
        attrs field on ``baybe.kernels.basic.RFFKernel`` (BayBE 0.15.0).
        """
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.RFF, num_samples=50))
        assert isinstance(kernel.base_kernel, RFFKernel)
        assert kernel.base_kernel.num_samples == 50

    def test_kernel_payload_table_covers_every_kind(self) -> None:
        """A new BayBEKernelKind member without test wiring fails here first."""
        assert set(_KERNEL_PAYLOADS) == set(BayBEKernelKind)
        assert set(_EXPECTED_KERNEL_LABELS) == set(BayBEKernelKind)

    @pytest.mark.parametrize("kind", list(BayBEKernelKind), ids=lambda kind: kind.value)
    def test_every_kernel_kind_fits_and_recommends(self, kind: BayBEKernelKind) -> None:
        """Fit-level coverage for the whole kernel zoo, GP phase included.

        Campaign *construction* alone cannot catch kernels that only fail
        during hyperparameter fitting or gradient-based acquisition
        optimization (both run inside ``recommend``), so this drives
        ``generate_suggestions`` end-to-end and checks the introspected
        metadata: correct kernel label, complete hyperparameters,
        in-bounds suggestions, and no introspection warnings. References:
        BayBE surrogates userguide
        (https://emdgroup.github.io/baybe/stable/userguide/surrogates.html)
        and the gpytorch kernels API
        (https://docs.gpytorch.ai/en/stable/kernels.html).
        """
        spec = _spec(
            _CONTINUOUS_PARAMS,
            {"surrogate": {"kind": "gp", "kernel": _KERNEL_PAYLOADS[kind]}},
            random_seed=7,
        )
        observations = [
            ObservationData(parameter_values={"a": v}, objective_values={"y": (v - 0.3) ** 2})
            for v in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=2, iteration=1
        )

        assert len(batch.suggestions) == 2
        for suggestion in batch.suggestions:
            assert 0.0 <= suggestion["parameter_values"]["a"] <= 1.0
        assert not [w for w in batch.warnings if "introspection incomplete" in w]
        method_info = batch.method_info
        assert method_info["kernel"] == _EXPECTED_KERNEL_LABELS[kind]
        assert "noise_variance" in method_info
        assert "output_scale" in method_info
        if kind in _LENGTHSCALE_FREE_KINDS:
            assert "lengthscales" not in method_info
        else:
            assert "lengthscales" in method_info
        extra_key = _EXPECTED_EXTRA_HYPERPARAMS.get(kind)
        if extra_key is not None:
            assert extra_key in method_info

    def test_gp_preset(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS, {"surrogate": {"kind": "gp", "gp_preset": "EDBO"}})
        options = extract_baybe_backend_options(spec.backend_options)
        assert isinstance(build_baybe_surrogate(spec, options), GaussianProcessSurrogate)
        assert _build_campaign(spec) is not None

    def test_random_forest_surrogate(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": "random_forest"}})
        campaign = _build_campaign(spec)
        # BayBE wraps single-output surrogates in a CompositeSurrogate.
        assert isinstance(_bo_recommender(campaign)._surrogate_model, CompositeSurrogate)

    def test_non_gp_surrogate_rejected_on_continuous_space(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS, {"surrogate": {"kind": "random_forest"}})
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert any("non-differentiable" in r.reason for r in result.unsupported)

    def test_noise_prior_params_reports_ignored(self) -> None:
        spec = _spec(_CONTINUOUS_PARAMS, noise_prior_params=(1.1, 0.05))
        result = BayBEBackend().validate_capabilities(spec)
        assert result.is_compatible
        assert any("noise_prior_params" in w for w in result.warnings)


class TestKernelMethodInfoLabels:
    """``method_info['kernel']`` must reflect the configured surrogate.

    The label is derived from the same options the surrogate is built
    from (``describe_configured_kernel``), so it cannot drift from the
    fitted kernel the way a static string can; the historical default
    label survives only for genuinely unconfigured campaigns.
    """

    def test_default_options_keep_default_label(self) -> None:
        """Path-pinning: no surrogate config keeps the stock BayBE label."""
        assert describe_configured_kernel(BayBEBackendOptions()) == DEFAULT_KERNEL_DESCRIPTION

    def test_matern_nu_labels_pin_allowed_values(self) -> None:
        """A display name exists for exactly the allowed Matern nu values."""
        assert set(MATERN_NU_LABELS) == set(MATERN_ALLOWED_NU)

    @pytest.mark.parametrize("kind", list(BayBEKernelKind), ids=lambda kind: kind.value)
    def test_configured_kernel_labels(self, kind: BayBEKernelKind) -> None:
        options = extract_baybe_backend_options(
            {"baybe": {"surrogate": {"kind": "gp", "kernel": _KERNEL_PAYLOADS[kind]}}}
        )
        assert describe_configured_kernel(options) == _EXPECTED_KERNEL_LABELS[kind]

    def test_matern_without_nu_labels_the_default_smoothness(self) -> None:
        """Omitted nu is built as 2.5, so the label must say Matern 5/2."""
        options = extract_baybe_backend_options(
            {"baybe": {"surrogate": {"kind": "gp", "kernel": {"kind": "matern"}}}}
        )
        assert describe_configured_kernel(options) == "Matern 5/2"

    def test_non_gp_surrogate_has_no_gp_kernel_label(self) -> None:
        options = extract_baybe_backend_options({"baybe": {"surrogate": {"kind": "random_forest"}}})
        assert describe_configured_kernel(options) == "not applicable (random_forest surrogate)"

    def test_preset_only_label_names_the_preset(self) -> None:
        options = extract_baybe_backend_options(
            {"baybe": {"surrogate": {"kind": "gp", "gp_preset": "EDBO"}}}
        )
        assert describe_configured_kernel(options) == "EDBO GP preset default kernel"

    def test_kernel_with_preset_mentions_both(self) -> None:
        options = extract_baybe_backend_options(
            {"baybe": {"surrogate": {"kind": "gp", "gp_preset": "EDBO", "kernel": {"kind": "rbf"}}}}
        )
        assert describe_configured_kernel(options) == "RBF (EDBO GP preset priors)"

    def test_select_methods_fallback_uses_configured_label(self) -> None:
        """The pre-campaign fallback path derives the same label."""
        spec = _spec(_CONTINUOUS_PARAMS, {"surrogate": {"kind": "gp", "kernel": {"kind": "rq"}}})
        assert BayBEBackend().select_methods(spec, 3)["kernel"] == "Rational Quadratic"

    def test_select_methods_default_spec_keeps_default_label(self) -> None:
        """Path-pinning: unconfigured campaigns keep the historical label."""
        spec = _spec(_CONTINUOUS_PARAMS)
        assert BayBEBackend().select_methods(spec, 0)["kernel"] == DEFAULT_KERNEL_DESCRIPTION

    def test_select_methods_non_gp_reports_surrogate_kind(self) -> None:
        """The fallback metadata must not claim a GP for non-GP surrogates."""
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": "random_forest"}})
        info = BayBEBackend().select_methods(spec, 3)
        assert info["model_type"] == "BayBE random_forest"
        assert info["optimization_strategy"] == "BotorchRecommender (random_forest surrogate)"
        assert info["kernel"] == "not applicable (random_forest surrogate)"


class TestSelectMethodsWarmupPhase:
    """Fallback metadata must respect the configured warm-up phase.

    ``select_methods`` has no live campaign to introspect, so it must
    resolve the random→GP switch point exactly like campaign construction
    does (``TwoPhaseMetaRecommender.switch_after``; BayBE recommenders
    userguide,
    https://emdgroup.github.io/baybe/stable/userguide/recommenders.html)
    instead of assuming the GP phase from the first observation on:
    before the switch the configured initial recommender is active and no
    surrogate or acquisition function exists. The labels come from the
    same helpers as the live path, so a consumer polling fallback then
    live metadata within one phase sees identical strings.
    """

    @pytest.mark.parametrize(
        ("n_observations", "in_warmup"),
        [(0, True), (4, True), (5, False)],
    )
    def test_switch_after_boundary(self, n_observations: int, in_warmup: bool) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"recommender": {"switch_after": 5}})
        info = BayBEBackend().select_methods(spec, n_observations)
        if in_warmup:
            assert info["model_type"] == "none (space-filling)"
            assert info["acquisition_function"] == "none (space-filling)"
            assert (
                info["optimization_strategy"] == "RandomRecommender (space-filling, no surrogate)"
            )
            # "No acquisition function during warm-up" is definitive, not
            # a static-table guess — mirrors the live nonpredictive path.
            assert info["acquisition_function_inferred"] is False
        else:
            assert info["model_type"] == "BayBE GP"
            assert info["acquisition_function"] == "qLogNoisyExpectedImprovement"
            assert info["optimization_strategy"] == "BotorchRecommender (GP-based)"
            assert info["acquisition_function_inferred"] is True

    def test_configured_initial_recommender_is_named(self) -> None:
        """Warm-up strategy names the configured recommender, not always random."""
        spec = _spec(
            _DISCRETE_PARAMS,
            {"recommender": {"initial_recommender": "fps", "switch_after": 3}},
        )
        info = BayBEBackend().select_methods(spec, 2)
        assert info["optimization_strategy"] == "FPSRecommender (space-filling, no surrogate)"
        assert info["model_type"] == "none (space-filling)"

    def test_neutral_initial_design_size_is_respected(self) -> None:
        """The neutral warm-up knob resolves exactly like campaign construction."""
        spec = _spec(_DISCRETE_PARAMS, initial_design_size=4)
        assert BayBEBackend().select_methods(spec, 3)["model_type"] == "none (space-filling)"
        assert BayBEBackend().select_methods(spec, 4)["model_type"] == "BayBE GP"

    @pytest.mark.parametrize(
        "recommender_options",
        [
            {"initial_recommender": "fps"},
            {"bayesian": {"n_restarts": 20}},
        ],
    )
    def test_omitted_switch_after_defers_to_initial_design_size(
        self, recommender_options: dict[str, Any]
    ) -> None:
        """Unrelated recommender sub-options must not shrink the neutral warm-up.

        An omitted ``switch_after`` previously resolved to its field
        default (1) whenever *any* recommender sub-option was present,
        silently overriding ``initial_design_size`` — a campaign asking
        for a 4-point FPS warm-up got a single warm-up point. The
        explicit-set precedence must hold for campaign construction and
        the fallback metadata alike.
        """
        spec = _spec(_DISCRETE_PARAMS, {"recommender": recommender_options}, initial_design_size=4)
        assert _meta(_build_campaign(spec)).switch_after == 4
        assert BayBEBackend().select_methods(spec, 3)["model_type"] == "none (space-filling)"
        assert BayBEBackend().select_methods(spec, 4)["model_type"] == "BayBE GP"

    def test_explicit_switch_after_still_beats_initial_design_size(self) -> None:
        """Path-pinning: the documented explicit-set precedence is unchanged."""
        spec = _spec(_DISCRETE_PARAMS, {"recommender": {"switch_after": 2}}, initial_design_size=4)
        assert _meta(_build_campaign(spec)).switch_after == 2

    def test_default_switch_after_without_any_knob(self) -> None:
        """Path-pinning: with neither knob set, BayBE switches after one point."""
        assert _meta(_build_campaign(_spec(_DISCRETE_PARAMS))).switch_after == 1

    def test_fallback_matches_live_warmup_metadata(self) -> None:
        """Live and fallback metadata agree field-for-field during warm-up.

        Both paths now derive their labels from the same helpers, so the
        core method fields must be identical for the same campaign state —
        a consumer polling diagnostics (fallback) and then generating
        suggestions (live) must not observe metadata churn.
        """
        options = {"recommender": {"initial_recommender": "fps", "switch_after": 5}}
        spec = _spec(_DISCRETE_PARAMS, options, random_seed=2)
        observations = [
            ObservationData(parameter_values={"a": v, "c": "x"}, objective_values={"y": v**2})
            for v in (0.0, 1.0, 2.0)
        ]
        fallback = BayBEBackend().select_methods(spec, len(observations))
        live = (
            BayBEBackend()
            .generate_suggestions(spec=spec, observations=observations, batch_size=1, iteration=1)
            .method_info
        )
        for field in (
            "model_type",
            "optimization_strategy",
            "acquisition_function",
            "acquisition_function_inferred",
            "kernel",
        ):
            assert fallback[field] == live[field], field


class TestKernelHyperparameterDiagnostics:
    """Kernel-specific learned parameters must reach the diagnostics payload.

    For lengthscale-free kernels the offset is the principal learned
    kernel parameter, so dropping it from ``hyperparameters`` while the
    suggestion method metadata carries it would leave the diagnostics
    endpoint incomplete (gpytorch parameter names per kernel:
    https://docs.gpytorch.ai/en/stable/kernels.html).
    """

    _OBSERVATIONS: ClassVar[list[ObservationData]] = [
        ObservationData(parameter_values={"a": v}, objective_values={"y": (v - 0.3) ** 2})
        for v in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]

    def _hyperparameters(self, kernel_payload: dict[str, Any]) -> dict[str, Any]:
        spec = _spec(
            _CONTINUOUS_PARAMS,
            {"surrogate": {"kind": "gp", "kernel": kernel_payload}},
            random_seed=7,
        )
        diagnostics = BayBEBackend().compute_diagnostics(
            spec, self._OBSERVATIONS, sections=frozenset(["model"])
        )
        hyperparameters = diagnostics["hyperparameters"]
        assert hyperparameters is not None
        return hyperparameters

    def test_polynomial_offset_reaches_diagnostics(self) -> None:
        hp = self._hyperparameters({"kind": "polynomial", "power": 2})
        assert hp["kernel_type"] == "PolynomialKernel"
        assert hp["lengthscales"] is None
        assert math.isfinite(hp["kernel_offset"])
        assert hp["noise_variance"] is not None
        assert hp["output_scale"] is not None

    def test_periodic_period_length_reaches_diagnostics(self) -> None:
        hp = self._hyperparameters({"kind": "periodic", "period_length": 1.5})
        assert hp["kernel_type"] == "PeriodicKernel"
        assert all(math.isfinite(v) for v in hp["kernel_period_length"])

    def test_rq_alpha_reaches_diagnostics(self) -> None:
        hp = self._hyperparameters({"kind": "rq"})
        assert hp["kernel_type"] == "RQKernel"
        assert math.isfinite(hp["kernel_alpha"])


class TestCampaignToggles:
    def test_explicit_toggles_reach_campaign(self) -> None:
        spec = _spec(
            _DISCRETE_PARAMS,
            {
                "allow_recommending_already_measured": True,
                "allow_recommending_already_recommended": True,
            },
        )
        campaign = _build_campaign(spec)
        assert campaign.allow_recommending_already_measured is True
        assert campaign.allow_recommending_already_recommended is True

    def test_purely_discrete_defaults_unchanged(self) -> None:
        """Path-pinning: unset toggles keep the historical strict defaults."""
        spec = _spec(_DISCRETE_PARAMS)
        campaign = _build_campaign(spec)
        assert campaign.allow_recommending_already_measured is False
        assert campaign.allow_recommending_already_recommended is False
        assert campaign.allow_recommending_pending_experiments is False

    def test_pending_toggle_reaches_continuous_campaign_when_true(self) -> None:
        """An explicit opt-in is forwarded on spaces with a continuous part.

        Previously the pending toggle was forwarded on purely discrete
        spaces only, so a continuous campaign silently ran with BayBE's
        AUTO resolution regardless of the configured value.
        """
        continuous = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
        ]
        spec = _spec(continuous, {"allow_recommending_pending_experiments": True})
        campaign = _build_campaign(spec)
        assert campaign.allow_recommending_pending_experiments is True

    def test_pending_toggle_unset_keeps_auto_on_continuous_spaces(self) -> None:
        """Unset = BayBE's AUTO, which resolves to True off-discrete.

        BayBE forbids ``False`` on spaces with a continuous subspace ("for
        algorithmic reasons"), so the honest unset behavior is AUTO — the
        capability layer rejects an explicit ``False`` for these spaces.
        """
        continuous = [
            ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
        ]
        campaign = _build_campaign(_spec(continuous))
        assert campaign.allow_recommending_pending_experiments is True

    def test_pending_toggle_explicit_true_on_discrete_space(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"allow_recommending_pending_experiments": True})
        campaign = _build_campaign(spec)
        assert campaign.allow_recommending_pending_experiments is True

    def test_tolerance_toggle_accepts_off_grid_measurement(self) -> None:
        """Relaxed tolerance accepts an off-grid numeric value; strict raises.

        Mirrors BayBE's ``add_measurements(numerical_measurements_must_be_within_tolerance=...)``
        contract (campaign userguide,
        https://emdgroup.github.io/baybe/stable/userguide/campaigns.html).
        """
        off_grid = [
            ObservationData(parameter_values={"a": 0.4, "c": "x"}, objective_values={"y": 1.0})
        ]
        from bo_engine.backend_base import BackendError

        strict_spec = _spec(_DISCRETE_PARAMS, random_seed=1)
        with pytest.raises((BackendError, ValueError)):
            BayBEBackend().generate_suggestions(
                spec=strict_spec, observations=off_grid, batch_size=1, iteration=1
            )

        relaxed_spec = _spec(
            _DISCRETE_PARAMS,
            {"measurements_must_be_within_tolerance": False},
            random_seed=1,
        )
        batch = BayBEBackend().generate_suggestions(
            spec=relaxed_spec, observations=off_grid, batch_size=1, iteration=1
        )
        assert len(batch.suggestions) == 1


@pytest.mark.slow
class TestPersistenceRoundTrips:
    """Every newly exposed recommender/surrogate must survive to_json/from_json."""

    @pytest.mark.parametrize("initial", ["fps", "kmeans", "pam", "gmm"])
    def test_initial_recommender_round_trip(self, initial: str) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"recommender": {"initial_recommender": initial}})
        campaign = _build_campaign(spec)
        restored = Campaign.from_json(campaign.to_json())
        assert type(_meta(restored).initial_recommender) is type(
            _meta(campaign).initial_recommender
        )

    @pytest.mark.parametrize(
        "surrogate_options",
        [
            {"kind": "gp", "kernel": {"kind": "matern", "nu": 1.5}},
            {"kind": "gp", "kernel": {"kind": "linear"}},
            {"kind": "gp", "kernel": {"kind": "periodic", "period_length": 1.5}},
            {"kind": "gp", "kernel": {"kind": "rff", "num_samples": 32}},
            {"kind": "gp", "gp_preset": "EDBO"},
            {"kind": "random_forest"},
            {"kind": "mean_prediction"},
        ],
    )
    def test_surrogate_round_trip(self, surrogate_options: dict[str, Any]) -> None:
        parameters = _CONTINUOUS_PARAMS if surrogate_options["kind"] == "gp" else _DISCRETE_PARAMS
        spec = _spec(parameters, {"surrogate": surrogate_options})
        campaign = _build_campaign(spec)
        restored = Campaign.from_json(campaign.to_json())
        assert type(_bo_recommender(restored)._surrogate_model) is type(
            _bo_recommender(campaign)._surrogate_model
        )

    def test_fps_generates_suggestion_end_to_end(self) -> None:
        spec = _spec(
            _DISCRETE_PARAMS,
            {"recommender": {"initial_recommender": "fps", "switch_after": 5}},
            random_seed=2,
        )
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=[], batch_size=2, iteration=0
        )
        assert len(batch.suggestions) == 2

    @pytest.mark.parametrize("kind", ["random_forest", "bayesian_linear", "mean_prediction"])
    def test_non_gp_surrogate_end_to_end_metadata(self, kind: str) -> None:
        """Non-GP surrogates run and report honest, non-GP method metadata.

        BayBE's non-GP surrogates (random forest / Bayesian linear / mean
        prediction; surrogates userguide,
        https://emdgroup.github.io/baybe/stable/userguide/surrogates.html)
        expose no GP covariance module by construction, so the method
        metadata must name the configured surrogate instead of the GP
        labels, and the absence of a kernel must not surface as an
        introspection warning. ``ngboost`` follows the same path but is
        gated on module availability at intake (see
        TestNGBoostAvailabilityProbe), so it is not exercised here.
        """
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": kind}}, random_seed=2)
        observations = [
            ObservationData(parameter_values={"a": v, "c": "x"}, objective_values={"y": v**2})
            for v in (0.0, 1.0, 2.0)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert len(batch.suggestions) == 1
        method_info = batch.method_info
        assert method_info["model_type"] == f"BayBE {kind}"
        assert method_info["optimization_strategy"] == f"BotorchRecommender ({kind} surrogate)"
        assert method_info["kernel"] == f"not applicable ({kind} surrogate)"
        assert method_info["kernel_type"] is None
        assert not [w for w in batch.warnings if "covariance module" in w]


class TestCampaignToggleSpaceCompatibility:
    """M-class: ``allow_recommending_*=False`` needs a purely discrete space.

    BayBE raises ``IncompatibilityError`` at campaign build for these
    toggles on continuous/hybrid spaces; the capability layer must report
    the combination at intake instead of deferring the crash to
    suggestion time.
    """

    _HYBRID_PARAMS: ClassVar[list[ParameterSpec]] = [
        ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["x", "y"]),
    ]

    @pytest.mark.parametrize(
        "toggle",
        [
            "allow_recommending_already_measured",
            "allow_recommending_already_recommended",
            "allow_recommending_pending_experiments",
        ],
    )
    def test_false_toggle_with_continuous_space_is_reported(self, toggle: str) -> None:
        spec = _spec(self._HYBRID_PARAMS, {toggle: False})
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        assert any(r.key == f"backend_options.baybe.{toggle}" for r in result.unsupported)

    def test_true_toggle_with_continuous_space_is_accepted(self) -> None:
        spec = _spec(self._HYBRID_PARAMS, {"allow_recommending_already_measured": True})
        assert BayBEBackend().validate_capabilities(spec).is_compatible

    def test_true_pending_toggle_with_continuous_space_is_accepted(self) -> None:
        spec = _spec(self._HYBRID_PARAMS, {"allow_recommending_pending_experiments": True})
        assert BayBEBackend().validate_capabilities(spec).is_compatible

    def test_false_toggle_with_purely_discrete_space_is_accepted(self) -> None:
        """Path-pinning: the discrete strict default stays valid."""
        spec = _spec(_DISCRETE_PARAMS, {"allow_recommending_already_measured": False})
        assert BayBEBackend().validate_capabilities(spec).is_compatible

    def test_false_pending_toggle_with_purely_discrete_space_is_accepted(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"allow_recommending_pending_experiments": False})
        assert BayBEBackend().validate_capabilities(spec).is_compatible


class TestNGBoostAvailabilityProbe:
    """M-class: the NGBoost gate must fire at intake, not at recommend time.

    BayBE imports ``ngboost`` lazily at *fit*, so ``NGBoostSurrogate()``
    constructs fine without the package and a build-and-catch probe never
    catches anything — the module-availability check is the actual gate
    (the chem-availability precedent).
    """

    def test_missing_module_yields_unsupported_report(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import importlib.util

        import bo_engine_baybe.backend as backend_mod

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ngboost":
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(backend_mod.importlib.util, "find_spec", fake_find_spec)
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": "ngboost"}})
        result = BayBEBackend().validate_capabilities(spec)
        assert not result.is_compatible
        reports = [r for r in result.unsupported if r.key == "backend_options.baybe.surrogate"]
        assert reports
        assert "ngboost" in reports[0].reason

    def test_available_module_passes_the_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import importlib.util
        from types import SimpleNamespace

        import bo_engine_baybe.backend as backend_mod

        real_find_spec = importlib.util.find_spec

        def fake_find_spec(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "ngboost":
                return SimpleNamespace(name="ngboost")
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(backend_mod.importlib.util, "find_spec", fake_find_spec)
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": "ngboost"}})
        result = BayBEBackend().validate_capabilities(spec)
        assert result.is_compatible


class TestEnumPinGuards:
    """Pin-guards the option docstrings promise: a BayBE rename fails CI."""

    def test_hybrid_sampler_members_exist_in_installed_baybe(self) -> None:
        from baybe.utils.sampling_algorithms import DiscreteSamplingMethod

        from bo_engine_baybe.options import BayBEHybridSampler

        installed = {member.value for member in DiscreteSamplingMethod}
        for member in BayBEHybridSampler:
            assert member.value in installed, (
                f"BayBEHybridSampler.{member.name}={member.value!r} is not a member "
                "of baybe.utils.sampling_algorithms.DiscreteSamplingMethod"
            )

    def test_gp_preset_members_exist_in_installed_baybe(self) -> None:
        from baybe.surrogates.gaussian_process.presets import GaussianProcessPreset

        from bo_engine_baybe.options import BayBEGPPreset

        installed = {member.value for member in GaussianProcessPreset}
        for member in BayBEGPPreset:
            assert member.value in installed, (
                f"BayBEGPPreset.{member.name}={member.value!r} is not a member of "
                "baybe.surrogates.gaussian_process.presets.GaussianProcessPreset"
            )


# Seeds and dataset size for the linear-kernel stability sweep below.
_LINEAR_STABILITY_SEEDS = tuple(range(10))
_LINEAR_STABILITY_N_OBSERVATIONS = 6


@pytest.mark.nightly
class TestLinearKernelAcquisitionStability:
    """kind='linear' must survive fit + acquisition across seeds.

    Multi-seed invariant test (bounds and batch-size assertions only, per
    the stochastic-test strategy in TESTING.md): the homogeneous linear
    kernel is pinned to zero prior variance at the normalized origin —
    the dot-product kernel without its ``sigma_0^2`` offset term
    (Rasmussen & Williams 2006, GPML section 4.2.2,
    https://gaussianprocess.org/gpml/) — which produced seed-dependent
    NaN acquisition gradients (botorch ``OptimizationGradientError``) on
    continuous spaces. The observations are deliberately *nonlinear* so
    the linear surrogate is misspecified, the regime where the
    degeneracy surfaced.
    """

    _PARAMS: ClassVar[list[ParameterSpec]] = [
        ParameterSpec(name="a", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ParameterSpec(name="b", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
    ]

    @pytest.mark.parametrize("seed", _LINEAR_STABILITY_SEEDS)
    def test_linear_kernel_recommends_on_misspecified_data(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        x = rng.uniform(0.0, 1.0, size=(_LINEAR_STABILITY_N_OBSERVATIONS, 2))
        observations = [
            ObservationData(
                parameter_values={"a": float(x[i, 0]), "b": float(x[i, 1])},
                objective_values={"y": float(np.sin(6.0 * x[i, 0]) + (x[i, 1] - 0.5) ** 2)},
            )
            for i in range(_LINEAR_STABILITY_N_OBSERVATIONS)
        ]
        spec = _spec(
            self._PARAMS,
            {"surrogate": {"kind": "gp", "kernel": {"kind": "linear"}}},
            random_seed=seed,
        )
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert len(batch.suggestions) == 1
        values = batch.suggestions[0]["parameter_values"]
        assert 0.0 <= values["a"] <= 1.0
        assert 0.0 <= values["b"] <= 1.0
