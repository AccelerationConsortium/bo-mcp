"""Pin the exact LOO downdate against brute-force leave-one-out.

``compute_exact_loo_moments`` implements the closed-form LOO predictive
moments of Rasmussen & Williams, "Gaussian Processes for Machine
Learning" (2006), §5.4.2, Eqs. 5.10-5.12:

    μ_{-i} = t_i - [K⁻¹(t - m)]_i / [K⁻¹]_{ii}
    σ²_{-i} = 1 / [K⁻¹]_{ii}

At *fixed* hyperparameters this identity is exact, so we can verify it to
numerical precision: refit-free downdate moments must equal the posterior
of a GP that is rebuilt on each n-1 subset with the hyperparameters of the
full-data model copied over. Because K includes the noise term, the
variance is the predictive variance of the held-out observation, which is
checked against ``posterior(..., observation_noise=True)``.

The CV layer on top of the downdate carries contracts of its own, pinned
here as well: scale-free metrics must be invariant to the units of y, the
result cache must never serve one model configuration's (or one CV
method's) metrics for another, K-fold splits must be paired across model
configurations and leave the global RNG untouched, and failed folds must
be excluded from the metrics instead of entering as zero predictions.

References:
    - Rasmussen & Williams "GPML" §5.4.2, Eqs. 5.10-5.12
    - Sundararajan & Keerthi, "Predictive Approaches for Choosing
      Hyperparameters in Gaussian Processes", Neural Computation 2001
"""

from __future__ import annotations

import math

import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_engine.cross_validation import (
    CVConfig,
    _compute_cache_key,
    _compute_kfold_cv,
    _fold_permutation,
    clear_cv_cache,
    compute_cv_for_model_list,
    compute_exact_loo_moments,
    compute_loo_cv_optimized,
)
from bo_engine.models import create_and_fit_model

BOUNDS_1D = torch.tensor([[0.0], [1.0]], dtype=torch.float64)


def _toy_data(n: int = 14, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    train_x = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.sin(3 * train_x) + 0.05 * torch.randn(n, 1, dtype=torch.float64)
    return train_x, train_y


def _fit_default_gp(train_x: torch.Tensor, train_y: torch.Tensor) -> SingleTaskGP:
    """The default CV model construction: Normalize + Standardize SingleTaskGP."""
    model = SingleTaskGP(
        train_x,
        train_y,
        input_transform=Normalize(d=train_x.shape[-1], bounds=BOUNDS_1D),
        outcome_transform=Standardize(m=1),
    )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    return model


class TestExactLOOMoments:
    """The downdate must reproduce brute-force LOO at fixed hyperparameters."""

    def test_matches_brute_force_loo_at_fixed_hyperparameters(self) -> None:
        """GPML Eq. 5.12 identity, checked to numerical precision.

        For every i, a GP rebuilt on the data without point i (with the
        full-data hyperparameters copied via ``load_state_dict``) must give
        the same held-out predictive mean and observation-noise-inclusive
        variance as the downdate. Transforms are disabled so the state dict
        carries no fold-dependent statistics.
        """
        train_x, train_y = _toy_data()
        n = train_x.shape[0]

        model = SingleTaskGP(train_x, train_y, outcome_transform=None)
        loo_mean, loo_var = compute_exact_loo_moments(model)

        assert loo_mean.shape == (n,)
        assert loo_var.shape == (n,)

        for i in range(n):
            mask = torch.ones(n, dtype=torch.bool)
            mask[i] = False
            fold_model = SingleTaskGP(train_x[mask], train_y[mask], outcome_transform=None)
            fold_model.load_state_dict(model.state_dict())
            fold_model.eval()
            with torch.no_grad():
                posterior = fold_model.posterior(train_x[i : i + 1], observation_noise=True)
                brute_mean = posterior.mean.reshape(-1)[0]
                brute_var = posterior.variance.reshape(-1)[0]

            assert loo_mean[i].item() == pytest.approx(brute_mean.item(), abs=1e-9)
            assert loo_var[i].item() == pytest.approx(brute_var.item(), abs=1e-9)

    def test_variance_includes_observation_noise(self) -> None:
        """LOO predictive variance must be at least the fitted noise level.

        Since K = K_f + σ_n²·I, the held-out predictive variance can never
        drop below the observation-noise variance — the property that makes
        the downdate suitable for coverage and residual checks against
        noisy measurements.
        """
        train_x, train_y = _toy_data()
        model = SingleTaskGP(train_x, train_y, outcome_transform=None)
        _, loo_var = compute_exact_loo_moments(model)

        noise_var = model.likelihood.noise.reshape(-1)[0].item()  # ty: ignore[call-non-callable]
        assert bool((loo_var >= noise_var - 1e-12).all())

    def test_works_in_models_transformed_target_space(self) -> None:
        """With Standardize, moments live in the standardized space.

        The downdate must operate on ``model.train_targets`` (already
        standardized at construction); rescaling y must therefore leave the
        downdate output unchanged up to floating-point noise.
        """
        train_x, train_y = _toy_data()

        def build(train_y_variant: torch.Tensor) -> SingleTaskGP:
            return SingleTaskGP(
                train_x,
                train_y_variant,
                input_transform=Normalize(d=1, bounds=BOUNDS_1D),
                outcome_transform=Standardize(m=1),
            )

        mean_a, var_a = compute_exact_loo_moments(build(train_y))
        mean_b, var_b = compute_exact_loo_moments(build(1000 * train_y))

        assert torch.allclose(mean_a, mean_b, atol=1e-8)
        assert torch.allclose(var_a, var_b, atol=1e-8)

    def test_rejects_batched_targets(self) -> None:
        """Batched/multi-output models have no single K⁻¹ diagonal to invert."""
        train_x, train_y = _toy_data()
        model = SingleTaskGP(train_x, torch.cat([train_y, train_y], dim=-1), outcome_transform=None)
        with pytest.raises(ValueError, match="single-output"):
            compute_exact_loo_moments(model)

    def test_restores_eval_mode(self) -> None:
        """The downdate flips to train mode internally and must restore eval."""
        train_x, train_y = _toy_data()
        model = SingleTaskGP(train_x, train_y, outcome_transform=None)
        model.eval()
        compute_exact_loo_moments(model)
        assert not model.training


class TestModelFactoryCacheIsolation:
    """CV results computed for one model configuration must never be served
    for another — the failure mode where every model-selection candidate
    silently received the first candidate's cached metrics.
    """

    def test_cache_key_distinguishes_factory_keys(self) -> None:
        train_x, train_y = _toy_data()
        config = CVConfig()

        key_a = _compute_cache_key(train_x, train_y, BOUNDS_1D, config, "candidate-a")
        key_b = _compute_cache_key(train_x, train_y, BOUNDS_1D, config, "candidate-b")
        key_none = _compute_cache_key(train_x, train_y, BOUNDS_1D, config, None)

        assert key_a != key_b
        assert key_a != key_none

    def test_cache_key_distinguishes_cv_methods(self) -> None:
        """Different ``CVConfig.method`` values must never share a key."""
        train_x, train_y = _toy_data()

        keys = {
            _compute_cache_key(train_x, train_y, BOUNDS_1D, CVConfig(method=method), "candidate")
            for method in ("auto", "batch_loo", "approximate_loo", "kfold")
        }

        assert len(keys) == 4

    def test_cached_result_respects_requested_method(self) -> None:
        """An explicit method request must not be answered from another
        method's cache entry: after caching a batch-LOO result, an
        approximate-LOO request on the same data must run (and report) the
        approximate path.
        """
        torch.manual_seed(3)
        n = 12
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x) + 0.01 * torch.randn(n, 1, dtype=torch.float64)

        clear_cv_cache()
        batch = compute_loo_cv_optimized(train_x, train_y, BOUNDS_1D, CVConfig(method="batch_loo"))
        approximate = compute_loo_cv_optimized(
            train_x, train_y, BOUNDS_1D, CVConfig(method="approximate_loo")
        )

        assert batch.method == "batch_loo"
        assert approximate.method == "approximate_loo", (
            "The approximate-LOO request was served the cached batch-LOO "
            "result — CVConfig.method is missing from the cache key."
        )

    def test_distinct_factories_yield_distinct_cached_metrics(self) -> None:
        """Same data, two factories with different keys → two CV results.

        A deliberately broken 'constant kernel' stand-in is simulated by
        fitting on shuffled targets, which must produce a clearly worse RMSE
        than the properly fitted factory; identical metrics would mean the
        cache ignored the factory key.
        """
        torch.manual_seed(3)
        n = 12
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x) + 0.01 * torch.randn(n, 1, dtype=torch.float64)
        config = CVConfig(k_folds=4)

        def fit_scrambled(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
            permutation = torch.randperm(y.shape[0])
            return _fit_default_gp(x, y[permutation])

        clear_cv_cache()
        torch.manual_seed(0)
        metrics_proper = compute_loo_cv_optimized(
            train_x,
            train_y,
            BOUNDS_1D,
            config,
            model_factory=_fit_default_gp,
            model_factory_key="proper",
        )
        torch.manual_seed(0)
        metrics_scrambled = compute_loo_cv_optimized(
            train_x,
            train_y,
            BOUNDS_1D,
            config,
            model_factory=fit_scrambled,
            model_factory_key="scrambled",
        )

        assert metrics_proper.rmse != metrics_scrambled.rmse, (
            "Two different model factories returned identical CV metrics — "
            "the cache key does not isolate model configurations."
        )
        assert metrics_proper.rmse < metrics_scrambled.rmse

    def test_anonymous_factory_bypasses_cache(self) -> None:
        """A factory without a key must not read or write the shared cache."""
        torch.manual_seed(3)
        n = 12
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x) + 0.01 * torch.randn(n, 1, dtype=torch.float64)
        config = CVConfig(k_folds=4)

        clear_cv_cache()
        torch.manual_seed(0)
        baseline = compute_loo_cv_optimized(train_x, train_y, BOUNDS_1D, config)

        calls = {"count": 0}

        def counting_factory(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
            calls["count"] += 1
            return _fit_default_gp(x, y)

        torch.manual_seed(0)
        compute_loo_cv_optimized(
            train_x, train_y, BOUNDS_1D, config, model_factory=counting_factory
        )

        assert not math.isnan(baseline.rmse)
        assert calls["count"] > 0, (
            "The anonymous factory was never invoked — its call was served "
            "from a cache entry it cannot legitimately share."
        )


class TestKFoldSplitDiscipline:
    """K-fold splits must be paired across model configurations and must not
    consume (or depend on) global torch RNG state.
    """

    def test_fold_permutation_is_deterministic_and_rng_neutral(self) -> None:
        train_x, train_y = _toy_data()

        state_before = torch.get_rng_state()
        permutation_a = _fold_permutation(train_x, train_y)
        permutation_b = _fold_permutation(train_x, train_y)
        state_after = torch.get_rng_state()

        assert torch.equal(permutation_a, permutation_b)
        assert torch.equal(state_before, state_after), "The K-fold split consumed global RNG state."
        assert sorted(permutation_a.tolist()) == list(range(train_x.shape[0]))

    def test_fold_permutation_depends_on_data(self) -> None:
        train_x, train_y = _toy_data()
        permutation_a = _fold_permutation(train_x, train_y)
        permutation_b = _fold_permutation(train_x, 2.0 * train_y)
        assert not torch.equal(permutation_a, permutation_b)

    def test_equivalent_factories_get_identical_folds_and_metrics(self) -> None:
        """Two separate CV calls with equivalent factories must agree exactly.

        This is the paired-comparison property model selection relies on:
        with shared splits and deterministic fitting, metric differences
        between candidates reflect the model configuration, not
        fold-sampling noise.
        """
        torch.manual_seed(3)
        n = 12
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x) + 0.01 * torch.randn(n, 1, dtype=torch.float64)

        metrics_a = _compute_kfold_cv(train_x, train_y, BOUNDS_1D, 4, _fit_default_gp)
        metrics_b = _compute_kfold_cv(train_x, train_y, BOUNDS_1D, 4, _fit_default_gp)

        assert metrics_a.rmse == metrics_b.rmse
        assert metrics_a.coverage_95 == metrics_b.coverage_95
        assert metrics_a.per_fold_errors == metrics_b.per_fold_errors


class TestKFoldFailedFolds:
    """Failed folds must be excluded, not scored as zero predictions."""

    def test_failed_fold_rows_do_not_enter_metrics(self) -> None:
        """One failing fold must leave metrics computed from the successful
        folds only. The targets sit near 100, so a zero-filled placeholder
        prediction for the failed fold's rows would inflate the RMSE to
        ~50 — orders of magnitude above the real generalization error.
        """
        torch.manual_seed(3)
        n = 12
        k = 4
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = 100.0 + train_x + 0.01 * torch.randn(n, 1, dtype=torch.float64)

        calls = {"count": 0}

        def flaky_factory(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
            calls["count"] += 1
            if calls["count"] == 1:
                message = "synthetic fold failure"
                raise RuntimeError(message)
            return _fit_default_gp(x, y)

        metrics = _compute_kfold_cv(train_x, train_y, BOUNDS_1D, k, flaky_factory)

        assert not math.isnan(metrics.rmse)
        assert len(metrics.per_fold_errors) == k - 1
        assert metrics.rmse < 5.0, (
            f"RMSE {metrics.rmse:.2f} is contaminated by zero-filled "
            "placeholder predictions from the failed fold."
        )

    def test_all_folds_failing_returns_failed_metrics(self) -> None:
        torch.manual_seed(3)
        train_x = torch.rand(12, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x)

        def always_failing(_x: torch.Tensor, _y: torch.Tensor) -> SingleTaskGP:
            message = "synthetic fold failure"
            raise RuntimeError(message)

        metrics = _compute_kfold_cv(train_x, train_y, BOUNDS_1D, 4, always_failing)

        assert math.isnan(metrics.rmse)
        assert metrics.method.endswith("fold_failed")


class TestFittedModelOverload:
    """The model-form call must validate the *passed* model, never silently
    score a freshly built default configuration in its place.
    """

    def test_different_fitted_models_get_different_metrics(self) -> None:
        """Two models with different kernels on identical data must produce
        different LOO metrics — identical results would mean the passed
        model was discarded in favor of a shared default (or a cache hit
        keyed only on the data).
        """
        torch.manual_seed(3)
        n = 16
        train_x = torch.rand(n, 1, dtype=torch.float64)
        train_y = torch.sin(6 * train_x) + 0.05 * torch.randn(n, 1, dtype=torch.float64)

        def fit_with_kernel(nu: float) -> SingleTaskGP:
            model = SingleTaskGP(
                train_x,
                train_y,
                covar_module=ScaleKernel(MaternKernel(nu=nu, ard_num_dims=1)),
                input_transform=Normalize(d=1, bounds=BOUNDS_1D),
                outcome_transform=Standardize(m=1),
            )
            fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
            return model

        clear_cv_cache()
        smooth = compute_loo_cv_optimized(fit_with_kernel(2.5), train_x, train_y, CVConfig())
        rough = compute_loo_cv_optimized(fit_with_kernel(0.5), train_x, train_y, CVConfig())

        assert smooth.method == "model_loo"
        assert rough.method == "model_loo"
        assert smooth.rmse != rough.rmse, (
            "Two differently configured fitted models received identical CV "
            "metrics — the model-form call is not validating the passed model."
        )

    def test_model_loo_matches_downdate_of_that_model(self) -> None:
        """Mechanism pin: the model-form metrics come from the passed
        model's own exact LOO downdate, untransformed to original units.
        """
        train_x, train_y = _toy_data()
        model = _fit_default_gp(train_x, train_y)

        metrics = compute_loo_cv_optimized(model, train_x, train_y, CVConfig())

        loo_mean, loo_var = compute_exact_loo_moments(model)
        loo_mean, _ = model.outcome_transform.untransform(  # ty: ignore[unresolved-attribute]
            loo_mean.unsqueeze(-1), loo_var.unsqueeze(-1)
        )
        expected_rmse = ((loo_mean.squeeze(-1) - train_y.squeeze(-1)) ** 2).mean().sqrt().item()

        assert metrics.rmse == pytest.approx(expected_rmse, rel=1e-9)

    def test_refit_methods_raise_without_factory(self) -> None:
        """batch_loo/kfold refit per fold and cannot rebuild an arbitrary
        fitted model — silently substituting the default construction would
        misreport the model's quality, so the call must fail loudly.
        """
        train_x, train_y = _toy_data()
        model = _fit_default_gp(train_x, train_y)

        for method in ("batch_loo", "kfold"):
            with pytest.raises(ValueError, match="refits models per fold"):
                compute_loo_cv_optimized(model, train_x, train_y, CVConfig(method=method))

    def test_k_folds_setting_raises_like_explicit_kfold(self) -> None:
        """``CVConfig(k_folds=...)`` keeps ``method="auto"`` but is this
        API's K-fold request (the tensor path resolves it to ``"kfold"``),
        so the model form must reject it identically — not silently
        downgrade it to the downdate. An explicit ``"approximate_loo"``
        still wins over ``k_folds``, mirroring the tensor-path precedence.
        """
        train_x, train_y = _toy_data()
        model = _fit_default_gp(train_x, train_y)

        with pytest.raises(ValueError, match="refits models per fold"):
            compute_loo_cv_optimized(model, train_x, train_y, CVConfig(k_folds=4))

        explicit = compute_loo_cv_optimized(
            model, train_x, train_y, CVConfig(method="approximate_loo", k_folds=4)
        )
        assert explicit.method == "model_loo"

    def test_model_list_forwarding_rejects_k_fold_requests(self) -> None:
        """``compute_cv_for_model_list`` validates the fitted sub-models, so
        a K-fold request must be rejected rather than silently downgraded;
        without a config it reports per-objective ``model_loo`` metrics.
        """
        torch.manual_seed(2)
        n = 14
        train_x = torch.rand(n, 2, dtype=torch.float64)
        train_y = torch.stack(
            [
                train_x[:, 0] + 0.05 * torch.randn(n, dtype=torch.float64),
                torch.sin(5 * train_x[:, 1]) + 0.05 * torch.randn(n, dtype=torch.float64),
            ],
            dim=-1,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model_list = create_and_fit_model(train_x, train_y, bounds)

        results = compute_cv_for_model_list(model_list, train_x, train_y, bounds)
        assert set(results) == {0, 1}
        for metrics in results.values():
            assert metrics.method == "model_loo"
            assert not math.isnan(metrics.rmse)

        with pytest.raises(ValueError, match="refits models per fold"):
            compute_cv_for_model_list(model_list, train_x, train_y, bounds, CVConfig(k_folds=4))

    def test_mismatched_training_data_raises(self) -> None:
        """The downdate describes the model's stored training set; passing
        other data must be rejected, not silently answered for the stored set.
        """
        train_x, train_y = _toy_data()
        model = _fit_default_gp(train_x, train_y)

        with pytest.raises(ValueError, match="training data"):
            compute_loo_cv_optimized(model, train_x, train_y + 1.0, CVConfig())

    def test_explicit_factory_takes_precedence_over_model(self) -> None:
        """With a factory supplied, the call follows tensor-form semantics."""
        train_x, train_y = _toy_data()
        model = _fit_default_gp(train_x, train_y)

        calls = {"count": 0}

        def counting_factory(x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
            calls["count"] += 1
            return _fit_default_gp(x, y)

        metrics = compute_loo_cv_optimized(
            model, train_x, train_y, CVConfig(k_folds=4), model_factory=counting_factory
        )

        assert calls["count"] > 0
        assert metrics.method == "4fold"
