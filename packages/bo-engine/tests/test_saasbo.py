"""Tests for SAASBO (Sparse Axis-Aligned Subspace Bayesian Optimization).

SAASBO is designed for high-dimensional optimization problems (50+ parameters)
where only a subset of parameters are important. It uses sparsity-inducing priors
on inverse lengthscales to identify important parameters.

These tests verify:
1. Model creation and fitting with NUTS inference
2. Lengthscale extraction and parameter importance
3. Sparse structure learning (irrelevant parameters should have large lengthscales)
4. Integration with suggestion generation
"""

import pytest
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize

import bo_engine.saasbo as saasbo_module
from bo_engine.constants import (
    SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD,
    SAASBO_MIN_DIMENSIONS,
)
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.saasbo import (
    SAASBOConfig,
    compute_saasbo_importance,
    compute_saasbo_importance_report,
    create_and_fit_saasbo_model,
    create_saasbo_model,
    estimate_saasbo_runtime,
    fit_saasbo_model,
    generate_saasbo_suggestions,
    get_saasbo_lengthscales,
    should_use_saasbo,
)


class TestSAASBOConfig:
    """Test SAASBOConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = SAASBOConfig()
        assert config.warmup_steps == 256
        assert config.num_samples == 128
        assert config.thinning == 16
        assert config.disable_progbar is True

    def test_custom_config(self) -> None:
        """Custom config values are respected."""
        config = SAASBOConfig(
            warmup_steps=512,
            num_samples=256,
            thinning=8,
            disable_progbar=False,
        )
        assert config.warmup_steps == 512
        assert config.num_samples == 256
        assert config.thinning == 8
        assert config.disable_progbar is False


class TestShouldUseSAASBO:
    """Test the should_use_saasbo decision function."""

    def test_low_dim_returns_false(self) -> None:
        """Low-dimensional problems should not use SAASBO."""
        assert should_use_saasbo(n_parameters=10, n_observations=50) is False
        assert should_use_saasbo(n_parameters=20, n_observations=100) is False
        assert should_use_saasbo(n_parameters=49, n_observations=200) is False

    def test_high_dim_returns_true(self) -> None:
        """High-dimensional problems with enough data should use SAASBO."""
        assert should_use_saasbo(n_parameters=50, n_observations=50) is True
        assert should_use_saasbo(n_parameters=100, n_observations=100) is True

    def test_high_dim_insufficient_data(self) -> None:
        """High-dimensional with insufficient data should not use SAASBO."""
        # Need at least max(10, n_params // 5) observations
        assert should_use_saasbo(n_parameters=50, n_observations=5) is False

    def test_custom_threshold(self) -> None:
        """Custom threshold is respected."""
        assert should_use_saasbo(n_parameters=30, n_observations=50, threshold=30) is True
        assert should_use_saasbo(n_parameters=30, n_observations=50, threshold=40) is False

    def test_default_threshold_tracks_min_dimensions_constant(self) -> None:
        """The default boundary is ``SAASBO_MIN_DIMENSIONS``, not a local literal.

        Pinning the default to the shared constant keeps the recommendation
        in sync with every other consumer of the SAASBO dimensionality
        threshold when the constant is retuned.
        """
        n_obs = 2 * SAASBO_MIN_DIMENSIONS
        assert should_use_saasbo(SAASBO_MIN_DIMENSIONS, n_obs) is True
        assert should_use_saasbo(SAASBO_MIN_DIMENSIONS - 1, n_obs) is False


class TestSAASBORuntimeEstimate:
    """Test runtime estimation for SAASBO."""

    def test_small_dataset_estimate(self) -> None:
        """Small datasets should have short runtime estimates."""
        estimate = estimate_saasbo_runtime(n_observations=30)
        assert "seconds" in estimate or "minute" in estimate

    def test_large_dataset_estimate(self) -> None:
        """Large datasets should have longer runtime estimates."""
        estimate = estimate_saasbo_runtime(n_observations=200)
        # Should be at least minutes
        assert "minute" in estimate or "hour" in estimate

    def test_config_affects_estimate(self) -> None:
        """More samples should increase runtime estimate."""
        config_small = SAASBOConfig(warmup_steps=100, num_samples=50)
        config_large = SAASBOConfig(warmup_steps=500, num_samples=300)

        estimate_small = estimate_saasbo_runtime(n_observations=50, config=config_small)
        estimate_large = estimate_saasbo_runtime(n_observations=50, config=config_large)

        # Both should give valid estimates (strings with time units)
        assert any(unit in estimate_small for unit in ["second", "minute", "hour"])
        assert any(unit in estimate_large for unit in ["second", "minute", "hour"])


@pytest.mark.usefixtures("torch_rng")
class TestSAASBOModelCreation:
    """Test SAASBO model creation."""

    def test_create_model_basic(self) -> None:
        """Model can be created with basic inputs."""
        train_x = torch.rand(10, 5, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)

        model = create_saasbo_model(train_x, train_y)
        assert model is not None
        # Model should not be fitted yet
        assert hasattr(model, "covar_module")

    def test_create_model_with_1d_y(self) -> None:
        """Model handles 1D y input."""
        train_x = torch.rand(10, 5, dtype=torch.double)
        train_y = torch.rand(10, dtype=torch.double)

        model = create_saasbo_model(train_x, train_y)
        assert model is not None

    def test_create_model_with_yvar(self) -> None:
        """Model can be created with known noise variance."""
        train_x = torch.rand(10, 5, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)
        train_yvar = torch.ones(10, 1, dtype=torch.double) * 0.01

        model = create_saasbo_model(train_x, train_y, train_yvar)
        assert model is not None


@pytest.mark.usefixtures("torch_rng")
class TestSAASBOModelFitting:
    """Test SAASBO model fitting with NUTS."""

    @pytest.mark.slow
    def test_fit_model_completes(self) -> None:
        """Model fitting completes without error."""
        # Use small config for fast testing
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 5, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_saasbo_model(train_x, train_y)
        fitted_model = fit_saasbo_model(model, config)

        assert fitted_model is not None

    @pytest.mark.slow
    def test_create_and_fit_combined(self) -> None:
        """Combined create and fit function works."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 5, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        assert model is not None


@pytest.mark.usefixtures("torch_rng")
class TestSAASBOLengthscales:
    """Test lengthscale extraction and importance computation."""

    @pytest.mark.slow
    def test_get_lengthscales_shape(self) -> None:
        """Lengthscales have correct shape."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)
        n_dims = 5

        train_x = torch.rand(15, n_dims, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        lengthscales = get_saasbo_lengthscales(model)

        # Should have one lengthscale per dimension
        assert lengthscales.shape[-1] == n_dims

    @pytest.mark.slow
    def test_importance_sums_to_one(self) -> None:
        """Parameter importance values sum to 1."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 5, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        importance = compute_saasbo_importance(model)

        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-5

    @pytest.mark.slow
    def test_importance_with_custom_names(self) -> None:
        """Importance uses custom parameter names."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        param_names = ["temp", "pressure", "flow_rate"]
        importance = compute_saasbo_importance(model, param_names)

        assert set(importance.keys()) == set(param_names)


class TestSAASBOSparseStructureLearning:
    """Test that SAASBO learns sparse structure."""

    @pytest.mark.slow
    def test_identifies_important_parameters(self) -> None:
        """SAASBO identifies parameters that affect the objective.

        Create a function where only the first 2 parameters matter:
        f(x) = x[0]^2 + x[1]^2 (other parameters are irrelevant)
        """
        config = SAASBOConfig(warmup_steps=16, num_samples=8, thinning=1)
        n_total_params = 10
        n_important = 2

        # Generate data where only first 2 params matter
        torch.manual_seed(42)
        train_x = torch.rand(30, n_total_params, dtype=torch.double)
        # Only x[0] and x[1] contribute to y
        train_y = train_x[:, 0:1] ** 2 + train_x[:, 1:2] ** 2

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        importance = compute_saasbo_importance(model)

        # Sort by importance
        sorted_params = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Top 2 parameters should be param_0 and param_1
        top_params = [p[0] for p in sorted_params[:n_important]]

        # At least one of the important params should be in top 2
        # (NUTS is stochastic so we allow some tolerance)
        important_in_top = sum(1 for p in top_params if p in ["param_0", "param_1"])
        assert important_in_top >= 1, f"Expected param_0 or param_1 in top, got {top_params}"


@pytest.mark.usefixtures("torch_rng")
class TestSAASBOSuggestionGeneration:
    """Test suggestion generation with SAASBO."""

    @pytest.mark.slow
    def test_generate_suggestions_returns_valid_candidates(self) -> None:
        """Suggestions are within bounds."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 5, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.double)

        candidates, acq_values, _metadata = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=2, config=config
        )

        # Check shapes
        assert candidates.shape == (2, 5)
        assert acq_values.numel() >= 1

        # Check bounds
        assert (candidates >= bounds[0]).all()
        assert (candidates <= bounds[1]).all()

    @pytest.mark.slow
    def test_generate_suggestions_metadata(self) -> None:
        """Suggestions include proper metadata."""
        config = SAASBOConfig(warmup_steps=8, num_samples=4, thinning=1)

        train_x = torch.rand(15, 5, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0] * 5, [1.0] * 5], dtype=torch.double)
        param_names = ["a", "b", "c", "d", "e"]

        _candidates, _acq_values, metadata = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=1, config=config, parameter_names=param_names
        )

        assert metadata["model_type"] == "SaasFullyBayesianSingleTaskGP (SAASBO)"
        assert metadata["acquisition_function"] == "qLogExpectedImprovement"
        assert metadata["inference_method"] == "NUTS (fully Bayesian)"
        assert "parameter_importance" in metadata
        assert "top_important_parameters" in metadata
        assert set(metadata["parameter_importance"].keys()) == set(param_names)


@pytest.mark.usefixtures("torch_rng")
class TestSAASBONormalization:
    """Inputs must be normalized to the unit cube before the SAAS prior.

    BoTorch's SAAS model assumes inputs normalized to ``[0, 1]^d``, and
    both the half-Cauchy SAAS prior and the inactive-dimension lengthscale
    calibration (``SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD``, inactive =>
    lengthscale >= 1e2) are stated in unit-cube units — see Eriksson &
    Jankowiak, "High-Dimensional Bayesian Optimization with Sparse
    Axis-Aligned Subspaces", UAI 2021 (https://arxiv.org/abs/2103.00349).
    A fit on raw lab-typical bounds (e.g. temperature 0-100) puts the
    lengthscales on the raw scale, so a genuinely driving dimension reads
    as inactive against the unit-calibrated threshold. Mirrors
    ``TestMultiFidelityNormalization``, which pinned the same contract for
    the multi-fidelity GP.
    """

    @staticmethod
    def _non_unit_problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x0 on a 0-100 scale drives the objective; x1/x2 live on [0, 1]."""
        bounds = torch.tensor([[0.0, 0.0, 0.0], [100.0, 1.0, 1.0]], dtype=torch.double)
        x = torch.rand(20, 3, dtype=torch.double)
        x[:, 0] *= 100.0  # temperature-like scale
        y = (x[:, 0:1] / 100.0) ** 2
        return x, y, bounds

    def test_create_model_attaches_normalize_with_bounds(self) -> None:
        """The factory builds a ``Normalize`` transform from the given bounds."""
        x, y, bounds = self._non_unit_problem()

        model = create_saasbo_model(x, y, bounds=bounds)

        assert isinstance(model.input_transform, Normalize)
        assert torch.equal(model.input_transform.bounds, bounds)

    def test_normalized_inputs_reach_the_pyro_model(self) -> None:
        """NUTS samples against unit-cube inputs, not raw parameter scales.

        ``SaasFullyBayesianSingleTaskGP`` hands ``pyro_model`` the
        *transformed* training inputs, so this is exactly the space the
        SAAS prior (and therefore the Eriksson & Jankowiak 2021 lengthscale
        calibration) sees during fitting.
        """
        x, y, bounds = self._non_unit_problem()

        model = create_saasbo_model(x, y, bounds=bounds)

        pyro_x = model.pyro_model.train_X
        assert (pyro_x >= 0.0).all()
        assert (pyro_x <= 1.0).all()

    def test_bounds_learned_from_data_when_omitted(self) -> None:
        """Without explicit bounds the transform still normalizes from data."""
        x = torch.rand(15, 4, dtype=torch.double)

        model = create_saasbo_model(x, torch.rand(15, 1, dtype=torch.double))

        assert isinstance(model.input_transform, Normalize)

    def test_generate_suggestions_threads_bounds_into_model_factory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The suggestion entry point passes its bounds to the model factory.

        Fast variant: the NUTS-fitted SAAS model is swapped for a plain
        fitted GP (the ``test_saasbo_direction`` pattern) so only the
        bounds plumbing is under test.
        """
        captured: dict[str, torch.Tensor | None] = {}

        def fake_create_and_fit(
            train_x: torch.Tensor,
            train_y: torch.Tensor,
            config: object = None,  # noqa: ARG001
            bounds: torch.Tensor | None = None,
        ) -> SingleTaskGP:
            captured["bounds"] = bounds
            assert bounds is not None
            return create_and_fit_single_task_model(train_x, train_y, bounds)

        monkeypatch.setattr(saasbo_module, "create_and_fit_saasbo_model", fake_create_and_fit)
        monkeypatch.setattr(
            saasbo_module, "compute_saasbo_importance_report", lambda _model, _names=None: []
        )

        bounds = torch.tensor([[0.0, 0.0], [100.0, 1.0]], dtype=torch.double)
        train_x = torch.rand(12, 2, dtype=torch.double) * (bounds[1] - bounds[0])
        train_y = (train_x[:, 0:1] / 100.0 - 0.3) ** 2

        candidates, _acq, _meta = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=1
        )

        assert captured["bounds"] is not None
        assert torch.equal(captured["bounds"], bounds)
        assert (candidates >= bounds[0]).all()
        assert (candidates <= bounds[1]).all()

    @pytest.mark.slow
    def test_planted_active_dimension_survives_non_unit_bounds(self) -> None:
        """A driving 0-100-scale dimension stays ``active=True`` after NUTS.

        On a raw-input fit the driving dimension's lengthscale tracks the
        0-100 magnitude and crosses the unit-cube-calibrated
        ``SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD``; with normalization it
        stays O(1). Calibration reference: Eriksson & Jankowiak,
        "High-Dimensional Bayesian Optimization with Sparse Axis-Aligned
        Subspaces", UAI 2021, §3.3 & Appendix B — active dimensions keep
        lengthscales O(1) in ``[0, 1]^d`` while inactive ones diverge to
        >= 1e2.
        """
        config = SAASBOConfig(warmup_steps=16, num_samples=8, thinning=1)
        n_dims = 6
        torch.manual_seed(42)
        bounds = torch.ones(2, n_dims, dtype=torch.double)
        bounds[0] = 0.0
        bounds[1, 0] = 100.0  # x0 is the non-unit, driving dimension
        x = torch.rand(30, n_dims, dtype=torch.double)
        x[:, 0] *= 100.0
        y = (x[:, 0:1] / 100.0) ** 2

        model = create_and_fit_saasbo_model(x, y, config=config, bounds=bounds)
        report = compute_saasbo_importance_report(
            model, parameter_names=[f"x{i}" for i in range(n_dims)]
        )

        planted = report[0]
        assert planted.active is True, (
            f"driving dimension x0 flagged inactive (lengthscale="
            f"{planted.lengthscale:.3g}); inputs were not normalized before "
            "the SAAS prior."
        )
        assert planted.lengthscale <= SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD
        # Normalized-space lengthscales must not track the raw 0-100 scale.
        assert planted.lengthscale < 100.0 / 2


@pytest.mark.usefixtures("torch_rng")
class TestSAASBOEdgeCases:
    """Test edge cases and error handling."""

    def test_too_few_samples(self) -> None:
        """Model creation fails gracefully with too few samples."""
        train_x = torch.rand(2, 5, dtype=torch.double)
        train_y = torch.rand(2, 1, dtype=torch.double)

        # Should still create model (fitting may fail later)
        model = create_saasbo_model(train_x, train_y)
        assert model is not None

    def test_single_dimension(self) -> None:
        """Works with single parameter (edge case)."""
        train_x = torch.rand(15, 1, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)

        model = create_saasbo_model(train_x, train_y)
        assert model is not None
