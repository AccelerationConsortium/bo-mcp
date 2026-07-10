"""Recommender / surrogate / campaign-toggle options on the BayBE backend.

Covers the typed ``backend_options['baybe']`` growth: initial-recommender
selection (FPS / clustering), BotorchRecommender tuning knobs, the curated
surrogate/kernel/preset surface with its build-and-catch intake guard,
the campaign-level ``allow_recommending_*`` /
``measurements_must_be_within_tolerance`` toggles, and campaign
persistence round-trips per newly exposed recommender/surrogate.

References: BayBE recommenders userguide
(https://emdgroup.github.io/baybe/stable/userguide/recommenders.html) and
surrogates userguide
(https://emdgroup.github.io/baybe/stable/userguide/surrogates.html).
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from baybe import Campaign
from baybe.kernels import LinearKernel, PeriodicKernel, PolynomialKernel, RFFKernel, RQKernel
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
from bo_engine_baybe.options import (
    BayBEBackendOptions,
    BayBEInitialRecommender,
    BayBEKernelConfig,
    BayBEKernelKind,
    BayBESurrogateConfig,
    BayBESurrogateKind,
    extract_baybe_backend_options,
)
from bo_engine_baybe.state import _build_campaign
from bo_engine_baybe.surrogates import _build_kernel, build_baybe_surrogate


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

    def test_period_length_on_non_periodic_rejected(self) -> None:
        with pytest.raises(ValueError, match="period_length is only valid"):
            BayBEKernelConfig(kind=BayBEKernelKind.RBF, period_length=2.0)

    def test_power_on_non_polynomial_rejected(self) -> None:
        with pytest.raises(ValueError, match="power is only valid"):
            BayBEKernelConfig(kind=BayBEKernelKind.RBF, power=2)

    def test_power_required_for_polynomial(self) -> None:
        with pytest.raises(ValueError, match="power is required"):
            BayBEKernelConfig(kind=BayBEKernelKind.POLYNOMIAL)

    def test_num_samples_on_non_rff_rejected(self) -> None:
        with pytest.raises(ValueError, match="num_samples is only valid"):
            BayBEKernelConfig(kind=BayBEKernelKind.RBF, num_samples=100)

    def test_num_samples_required_for_rff(self) -> None:
        with pytest.raises(ValueError, match="num_samples is required"):
            BayBEKernelConfig(kind=BayBEKernelKind.RFF)


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

    def test_linear_kernel(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.LINEAR))
        assert isinstance(kernel.base_kernel, LinearKernel)

    def test_periodic_kernel_default(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.PERIODIC))
        assert isinstance(kernel.base_kernel, PeriodicKernel)
        assert kernel.base_kernel.period_length_initial_value is None

    def test_periodic_kernel_explicit_period_length(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.PERIODIC, period_length=3.5))
        assert isinstance(kernel.base_kernel, PeriodicKernel)
        assert kernel.base_kernel.period_length_initial_value == pytest.approx(3.5)

    def test_polynomial_kernel(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.POLYNOMIAL, power=3))
        assert isinstance(kernel.base_kernel, PolynomialKernel)
        assert kernel.base_kernel.power == 3

    def test_rq_kernel(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.RQ))
        assert isinstance(kernel.base_kernel, RQKernel)

    def test_rff_kernel(self) -> None:
        kernel = _build_kernel(BayBEKernelConfig(kind=BayBEKernelKind.RFF, num_samples=50))
        assert isinstance(kernel.base_kernel, RFFKernel)
        assert kernel.base_kernel.num_samples == 50

    @pytest.mark.parametrize(
        "kernel_payload",
        [
            {"kind": "linear"},
            {"kind": "periodic"},
            {"kind": "periodic", "period_length": 1.5},
            {"kind": "polynomial", "power": 2},
            {"kind": "rq"},
            {"kind": "rff", "num_samples": 32},
        ],
    )
    def test_new_kernel_kinds_build_a_campaign(self, kernel_payload: dict[str, Any]) -> None:
        spec = _spec(
            _CONTINUOUS_PARAMS,
            {"surrogate": {"kind": "gp", "kernel": kernel_payload}},
        )
        options = extract_baybe_backend_options(spec.backend_options)
        assert isinstance(build_baybe_surrogate(spec, options), GaussianProcessSurrogate)
        assert _build_campaign(spec) is not None

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

    def test_random_forest_generates_suggestion_end_to_end(self) -> None:
        spec = _spec(_DISCRETE_PARAMS, {"surrogate": {"kind": "random_forest"}}, random_seed=2)
        observations = [
            ObservationData(parameter_values={"a": v, "c": "x"}, objective_values={"y": v**2})
            for v in (0.0, 1.0, 2.0)
        ]
        batch = BayBEBackend().generate_suggestions(
            spec=spec, observations=observations, batch_size=1, iteration=1
        )
        assert len(batch.suggestions) == 1


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
