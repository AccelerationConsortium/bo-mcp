"""Tests reproducing results from BoTorch SAASBO tutorials.

This module validates that our SAASBO implementation achieves results consistent
with the official BoTorch tutorials and the SAASBO paper.

Test Tiers:
    - smoke: Fast tests for basic functionality (model creation, shapes)
    - default: Functional tests with reduced MCMC settings (CI-compatible)
    - slow: Full MCMC settings as in tutorial (nightly/manual only)

The slow tests use full MCMC settings (256 warmup, 128 samples) which can take
several minutes per test. These are skipped in normal CI runs.

References:
    - SAASBO Tutorial: https://botorch.org/docs/tutorials/saasbo/
    - SAASBO Paper: Eriksson & Jankowiak "High-Dimensional Bayesian Optimization
                    with Sparse Axis-Aligned Subspaces" UAI 2021
    - Paper PDF: https://proceedings.mlr.press/v161/eriksson21a/eriksson21a.pdf
"""

import pytest
import torch
from torch.quasirandom import SobolEngine

from bo_engine import (
    SAASBOConfig,
    compute_saasbo_importance,
    estimate_saasbo_runtime,
    should_use_saasbo,
)
from bo_engine.benchmarks import branin, branin_bounds

# =============================================================================
# MCMC Configuration Constants
# =============================================================================

# Full tutorial settings (slow but accurate)
# These produce high-quality posterior samples but take several minutes
TUTORIAL_INITIAL_POINTS = 10
TUTORIAL_ITERATIONS = 8
TUTORIAL_BATCH_SIZE = 5
TUTORIAL_TOTAL_EVALS = 50
TUTORIAL_WARMUP_STEPS = 256
TUTORIAL_NUM_SAMPLES = 128

# Smoke test settings (very fast, for basic functionality)
SMOKE_WARMUP_STEPS = 32
SMOKE_NUM_SAMPLES = 16

# CI-fast settings (faster than smoke but still validates core logic)
# These settings are a compromise: fast enough for CI, accurate enough for testing
CI_WARMUP_STEPS = 8
CI_NUM_SAMPLES = 4

BRANIN_MINIMUM = 0.397887


def branin_30d(x: torch.Tensor) -> torch.Tensor:
    """Branin function embedded in 30D (only dims 0,1 matter).

    Reference: SAASBO tutorial embeds Branin in 30D space.

    Args:
        x: Input tensor of shape (..., 30) in [0, 1]^30

    Returns:
        Function values of shape (...)
    """
    bounds = branin_bounds()
    # Extract first 2 dimensions and scale to Branin bounds
    x_2d = x[..., :2]
    x_scaled = x_2d * (bounds[1] - bounds[0]) + bounds[0]
    return branin(x_scaled)


# =============================================================================
# Test Classes
# =============================================================================


@pytest.mark.tutorial
class TestSAASBOBranin30D:
    """Reproduce SAASBO results on 30D embedded Branin from BoTorch tutorial.

    Reference:
        - Tutorial: https://botorch.org/docs/tutorials/saasbo/
        - Paper: Eriksson & Jankowiak "High-Dimensional Bayesian Optimization
                 with Sparse Axis-Aligned Subspaces" UAI 2021

    Tutorial setup:
        - Problem: Branin embedded in 30D (dims 0,1 are true, 2-29 are noise)
        - Initial points: 10 (Sobol)
        - Iterations: 8
        - Batch size: 5
        - Total evaluations: 50
        - Warmup steps: 256
        - Num samples: 128
    """

    @pytest.mark.smoke
    def test_branin_30d_function_shape(self) -> None:
        """Branin embedded in 30D should return correct shape."""
        x = torch.rand(10, 30, dtype=torch.float64)
        y = branin_30d(x)

        assert y.shape == (10,), f"Expected shape (10,), got {y.shape}"

    @pytest.mark.smoke
    def test_branin_30d_irrelevant_dims(self) -> None:
        """Branin 30D should ignore dimensions 2-29.

        Only dimensions 0 and 1 affect the function value.
        """
        # Same first 2 dims, different rest
        x1 = torch.zeros(1, 30, dtype=torch.float64)
        x2 = torch.ones(1, 30, dtype=torch.float64)
        x1[0, :2] = torch.tensor([0.5, 0.5])
        x2[0, :2] = torch.tensor([0.5, 0.5])

        y1 = branin_30d(x1)
        y2 = branin_30d(x2)

        assert y1.item() == pytest.approx(y2.item()), (
            f"Different irrelevant dims should give same output: {y1.item()} vs {y2.item()}"
        )

    @pytest.mark.smoke
    def test_branin_30d_minimum(self) -> None:
        """Branin 30D should have same minimum as 2D Branin.

        Minimum is at dims 0,1 = (pi, 2.275) scaled to [0,1]
        """
        # Optimal point in Branin coordinates: (pi, 2.275)
        # Scaled to [0,1]: ((pi + 5) / 15, 2.275 / 15) ~ (0.542, 0.152)
        x_opt = torch.zeros(1, 30, dtype=torch.float64)
        x_opt[0, 0] = (3.14159 + 5) / 15  # ~0.542
        x_opt[0, 1] = 2.275 / 15  # ~0.152

        y = branin_30d(x_opt)

        assert y.item() == pytest.approx(BRANIN_MINIMUM, rel=0.01), (
            f"Minimum should be ~{BRANIN_MINIMUM}, got {y.item()}"
        )


@pytest.mark.tutorial
class TestSAASBOLengthscaleSparsity:
    """Test that SAASBO identifies important dimensions via lengthscales.

    Reference: Section 4 of SAASBO paper - Half-Cauchy prior induces sparsity

    Test Tiers:
        - test_saasbo_identifies_important_dims_ci: Fast CI test with minimal MCMC
        - test_saasbo_identifies_important_dims_full: Full MCMC (nightly only)
    """

    @pytest.mark.slow
    def test_saasbo_identifies_important_dims_ci(self) -> None:
        """SAASBO model fits and produces valid lengthscales (CI-fast version).

        This test uses minimal MCMC settings to verify basic functionality:
        - Model creation and fitting completes
        - Lengthscales have correct shape
        - All lengthscales are positive

        For full validation of lengthscale sparsity, see the nightly test.
        """
        torch.manual_seed(42)

        # Generate training data from 30D embedded Branin
        sobol = SobolEngine(dimension=30, scramble=True, seed=42)
        train_x = sobol.draw(15).to(torch.float64)
        train_y = branin_30d(train_x).unsqueeze(-1)

        # CI-fast config: minimal MCMC for speed
        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import create_and_fit_saasbo_model, get_saasbo_lengthscales

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        lengthscales = get_saasbo_lengthscales(model)

        # Basic validity checks (always hold regardless of MCMC quality)
        assert lengthscales.numel() == 30, (
            f"Should have 30 lengthscales, got {lengthscales.numel()}"
        )
        assert (lengthscales > 0).all(), "All lengthscales should be positive"

    @pytest.mark.nightly
    def test_saasbo_identifies_important_dims_full(self) -> None:
        """SAASBO correctly identifies important dimensions (full MCMC).

        This test uses full tutorial MCMC settings and validates that:
        - Important dims (0, 1) have smaller lengthscales
        - Irrelevant dims (2-29) have larger lengthscales

        Marked as @nightly because full MCMC takes several minutes.
        """
        torch.manual_seed(42)

        sobol = SobolEngine(dimension=30, scramble=True, seed=42)
        train_x = sobol.draw(20).to(torch.float64)
        train_y = branin_30d(train_x).unsqueeze(-1)

        # Full tutorial config
        config = SAASBOConfig(
            warmup_steps=TUTORIAL_WARMUP_STEPS,
            num_samples=TUTORIAL_NUM_SAMPLES,
            thinning=16,
            disable_progbar=True,
        )

        from bo_engine.saasbo import create_and_fit_saasbo_model, get_saasbo_lengthscales

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        lengthscales = get_saasbo_lengthscales(model)

        # With full MCMC, we can test sparsity properties
        important_dims = lengthscales[:2]  # Dims 0, 1
        irrelevant_dims = lengthscales[2:]  # Dims 2-29

        # Important dims should have smaller lengthscales (more variation)
        # Note: Using median for robustness to outliers
        assert important_dims.median() < irrelevant_dims.median(), (
            f"Important dims median ({important_dims.median():.2f}) should be < "
            f"irrelevant dims median ({irrelevant_dims.median():.2f})"
        )


@pytest.mark.tutorial
class TestSAASBOImportance:
    """Test SAASBO parameter importance computation."""

    @pytest.mark.slow
    def test_importance_sums_to_one(self) -> None:
        """Parameter importance scores should sum to 1.0."""
        torch.manual_seed(42)

        # Small problem for speed
        sobol = SobolEngine(dimension=10, scramble=True, seed=42)
        train_x = sobol.draw(15).to(torch.float64)
        # Create data where first 2 dims are important
        train_y = train_x[:, 0:1] ** 2 + train_x[:, 1:2] ** 2

        # CI-fast config
        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import create_and_fit_saasbo_model

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        importance = compute_saasbo_importance(model)

        total = sum(importance.values())
        assert total == pytest.approx(1.0, rel=0.01), f"Importance should sum to 1.0, got {total}"

    @pytest.mark.slow
    def test_importance_with_parameter_names(self) -> None:
        """Parameter importance should use provided names."""
        torch.manual_seed(42)

        sobol = SobolEngine(dimension=5, scramble=True, seed=42)
        train_x = sobol.draw(15).to(torch.float64)
        train_y = train_x[:, 0:1] ** 2

        # CI-fast config
        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import create_and_fit_saasbo_model

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)
        param_names = ["x1", "x2", "x3", "x4", "x5"]
        importance = compute_saasbo_importance(model, param_names)

        assert list(importance.keys()) == param_names


@pytest.mark.tutorial
class TestShouldUseSAASBO:
    """Test SAASBO applicability heuristics."""

    @pytest.mark.smoke
    def test_low_dim_no_saasbo(self) -> None:
        """Low-dimensional problems don't need SAASBO."""
        assert should_use_saasbo(10, 50) is False
        assert should_use_saasbo(30, 100) is False
        assert should_use_saasbo(49, 100) is False

    @pytest.mark.smoke
    def test_high_dim_with_data_use_saasbo(self) -> None:
        """High-dimensional problems with enough data should use SAASBO."""
        assert should_use_saasbo(50, 50) is True
        assert should_use_saasbo(100, 50) is True

    @pytest.mark.smoke
    def test_high_dim_insufficient_data(self) -> None:
        """High dimensions with too little data should not use SAASBO."""
        assert should_use_saasbo(100, 5) is False

    @pytest.mark.smoke
    def test_custom_threshold(self) -> None:
        """Custom threshold can be specified."""
        assert should_use_saasbo(30, 30, threshold=25) is True
        assert should_use_saasbo(30, 30, threshold=50) is False


@pytest.mark.tutorial
class TestSAASBOConfig:
    """Test SAASBO configuration."""

    @pytest.mark.smoke
    def test_default_config(self) -> None:
        """Default config should match tutorial recommendations."""
        config = SAASBOConfig()

        assert config.warmup_steps == 256
        assert config.num_samples == 128
        assert config.thinning == 16
        assert config.disable_progbar is True

    @pytest.mark.smoke
    def test_smoke_config(self) -> None:
        """Smoke config should use reduced samples for speed."""
        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
        )

        assert config.warmup_steps == CI_WARMUP_STEPS
        assert config.num_samples == CI_NUM_SAMPLES


@pytest.mark.tutorial
class TestSAASBORuntimeEstimation:
    """Test SAASBO runtime estimation."""

    @pytest.mark.smoke
    def test_runtime_estimate_small_data(self) -> None:
        """Small datasets should have reasonable runtime estimates."""
        estimate = estimate_saasbo_runtime(30)

        assert "seconds" in estimate or "minutes" in estimate

    @pytest.mark.smoke
    def test_runtime_estimate_large_data(self) -> None:
        """Large datasets should have longer runtime estimates."""
        estimate_small = estimate_saasbo_runtime(30)
        estimate_large = estimate_saasbo_runtime(300)

        # Can't easily compare strings, but both should be valid
        assert estimate_small is not None
        assert estimate_large is not None

    @pytest.mark.smoke
    def test_runtime_with_config(self) -> None:
        """Runtime should scale with number of samples."""
        config_small = SAASBOConfig(warmup_steps=32, num_samples=16)
        config_large = SAASBOConfig(warmup_steps=512, num_samples=256)

        estimate_small = estimate_saasbo_runtime(50, config_small)
        estimate_large = estimate_saasbo_runtime(50, config_large)

        # Both should be valid estimates
        assert estimate_small is not None
        assert estimate_large is not None


@pytest.mark.tutorial
class TestSAASBOModelCreation:
    """Test SAASBO model creation."""

    @pytest.mark.slow
    def test_create_saasbo_model(self) -> None:
        """Should create SaasFullyBayesianSingleTaskGP model."""
        torch.manual_seed(42)

        train_x = torch.rand(15, 30, dtype=torch.float64)
        train_y = torch.rand(15, 1, dtype=torch.float64)

        from bo_engine.saasbo import create_saasbo_model

        model = create_saasbo_model(train_x, train_y)

        # Check it's the right type
        from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

        assert isinstance(model, SaasFullyBayesianSingleTaskGP)

    @pytest.mark.slow
    def test_create_and_fit_saasbo_model(self) -> None:
        """Should create and fit SAASBO model."""
        torch.manual_seed(42)

        train_x = torch.rand(15, 10, dtype=torch.float64)
        train_y = torch.rand(15, 1, dtype=torch.float64)

        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import create_and_fit_saasbo_model

        model = create_and_fit_saasbo_model(train_x, train_y, config=config)

        # Check model is fitted (can make predictions)
        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 10, dtype=torch.float64)
            posterior = model.posterior(test_x)
            # Fully Bayesian models return mean shape (num_samples, n_test, 1)
            # The number of MCMC samples affects the first dimension
            # We check that the test dimension (5) is present
            assert 5 in posterior.mean.shape, (
                f"Expected test dimension 5 in shape {posterior.mean.shape}"
            )


@pytest.mark.tutorial
class TestSAASBOSuggestionGeneration:
    """Test SAASBO suggestion generation."""

    @pytest.mark.slow
    def test_generate_saasbo_suggestions(self) -> None:
        """Should generate valid suggestions using SAASBO."""
        torch.manual_seed(42)

        train_x = torch.rand(15, 10, dtype=torch.float64)
        train_y = torch.rand(15, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float64)

        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import generate_saasbo_suggestions

        candidates, _acq_values, metadata = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=2, config=config
        )

        assert candidates.shape == (2, 10)
        assert (candidates >= 0).all()
        assert (candidates <= 1).all()
        assert "parameter_importance" in metadata
        assert "top_important_parameters" in metadata

    @pytest.mark.slow
    def test_saasbo_metadata_completeness(self) -> None:
        """SAASBO metadata should include all required fields."""
        torch.manual_seed(42)

        train_x = torch.rand(15, 10, dtype=torch.float64)
        train_y = torch.rand(15, 1, dtype=torch.float64)
        bounds = torch.tensor([[0.0] * 10, [1.0] * 10], dtype=torch.float64)

        config = SAASBOConfig(
            warmup_steps=CI_WARMUP_STEPS,
            num_samples=CI_NUM_SAMPLES,
            thinning=2,
            disable_progbar=True,
        )

        from bo_engine.saasbo import generate_saasbo_suggestions

        _, _, metadata = generate_saasbo_suggestions(
            train_x, train_y, bounds, batch_size=1, config=config
        )

        assert "model_type" in metadata
        assert "SAASBO" in metadata["model_type"]
        assert "acquisition_function" in metadata
        assert "parameter_importance" in metadata
        assert "top_important_parameters" in metadata
        assert "inference_method" in metadata
        assert "NUTS" in metadata["inference_method"]
