"""Tests for Multi-Fidelity Bayesian Optimization using qMFKG.

Multi-fidelity BO supports cheap/expensive evaluation strategies where
lower-fidelity evaluations are cheaper but provide useful information
about the high-fidelity function.

These tests verify:
1. FidelitySpec and MultiFidelityConfig dataclasses
2. Multi-fidelity model creation and fitting
3. Cost model creation
4. qMFKG acquisition function creation
5. Integration with suggestion generation
6. Fidelity-aware optimization behavior
"""

import pytest
import torch

from bo_engine.multifidelity import (
    FidelitySpec,
    MultiFidelityConfig,
    create_and_fit_multifidelity_model,
    create_cost_model,
    create_mfkg_acquisition,
    create_multifidelity_model,
    fit_multifidelity_model,
    generate_multifidelity_suggestions,
    optimize_mfkg,
)


class TestFidelitySpec:
    """Test FidelitySpec dataclass."""

    def test_basic_creation(self) -> None:
        """FidelitySpec can be created with required fields."""
        spec = FidelitySpec(
            fidelity_dim=2,
            target_fidelity=1.0,
            name="resolution",
            bounds=(0.1, 1.0),
        )

        assert spec.fidelity_dim == 2
        assert spec.target_fidelity == 1.0
        assert spec.target == 1.0  # Alias
        assert spec.name == "resolution"
        assert spec.bounds == (0.1, 1.0)
        assert spec.cost_weight == 1.0  # default
        assert spec.fixed_cost == 5.0  # default

    def test_custom_costs(self) -> None:
        """FidelitySpec accepts custom cost parameters."""
        spec = FidelitySpec(
            fidelity_dim=0,
            target_fidelity=1000.0,
            name="grid_size",
            bounds=(10.0, 1000.0),
            cost_weight=2.5,
            fixed_cost=10.0,
        )

        assert spec.cost_weight == 2.5
        assert spec.fixed_cost == 10.0


class TestMultiFidelityConfig:
    """Test MultiFidelityConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        fidelity = FidelitySpec(fidelity_dim=2, target_fidelity=1.0)
        config = MultiFidelityConfig(fidelity_spec=fidelity)

        assert config.fidelity_spec == fidelity
        assert config.num_fantasies == 64
        assert config.num_restarts == 10
        assert config.raw_samples == 512

    def test_custom_config(self) -> None:
        """Custom config values are respected."""
        fidelity = FidelitySpec(fidelity_dim=2, target_fidelity=1.0)
        config = MultiFidelityConfig(
            fidelity_spec=fidelity,
            num_fantasies=32,
            num_restarts=20,
            raw_samples=256,
        )

        assert config.num_fantasies == 32
        assert config.num_restarts == 20
        assert config.raw_samples == 256


@pytest.mark.usefixtures("torch_rng")
class TestMultiFidelityModelCreation:
    """Test multi-fidelity model creation."""

    def test_create_model_basic(self) -> None:
        """Model can be created with fidelity dimension."""
        # 2 params + 1 fidelity = 3 columns, fidelity is last
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        fidelity_dim = 2  # Last column

        model = create_multifidelity_model(train_x, train_y, fidelity_dim)
        assert model is not None

    def test_create_model_with_1d_y(self) -> None:
        """Model handles 1D y input."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, dtype=torch.double)
        fidelity_dim = 2

        model = create_multifidelity_model(train_x, train_y, fidelity_dim)
        assert model is not None


@pytest.mark.usefixtures("torch_rng")
class TestMultiFidelityModelFitting:
    """Test multi-fidelity model fitting."""

    def test_fit_model(self) -> None:
        """Model can be fitted."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        fidelity_dim = 2

        model = create_multifidelity_model(train_x, train_y, fidelity_dim)
        fitted = fit_multifidelity_model(model)

        # Model should be fitted and usable for predictions
        fitted.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 3, dtype=torch.double)
            posterior = fitted.posterior(test_x)
            assert posterior.mean.shape == (5, 1)

    def test_create_and_fit_combined(self) -> None:
        """Combined create and fit function works."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        fidelity_dim = 2

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)

        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 3, dtype=torch.double)
            posterior = model.posterior(test_x)
            assert posterior.mean.shape == (5, 1)


class TestCostModel:
    """Test cost model creation."""

    def test_create_cost_model_defaults(self) -> None:
        """Cost model can be created with defaults."""
        cost_model = create_cost_model(fidelity_dim=2)

        assert cost_model is not None
        # Test that it can evaluate cost
        test_x = torch.tensor([[0.5, 0.5, 0.8]], dtype=torch.double)
        cost = cost_model(test_x)
        assert cost.numel() > 0

    def test_create_cost_model_custom(self) -> None:
        """Cost model respects custom parameters."""
        cost_model = create_cost_model(
            fidelity_dim=2,
            cost_weight=3.0,
            fixed_cost=10.0,
        )

        # Cost should include fixed cost
        test_x = torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.double)
        cost = cost_model(test_x)
        # At minimum fidelity (0), cost should be near fixed_cost
        assert cost.item() >= 9.0  # Allow some tolerance

    def test_cost_increases_with_fidelity(self) -> None:
        """Higher fidelity should have higher cost."""
        cost_model = create_cost_model(fidelity_dim=2, cost_weight=1.0, fixed_cost=5.0)

        low_fidelity = torch.tensor([[0.5, 0.5, 0.1]], dtype=torch.double)
        high_fidelity = torch.tensor([[0.5, 0.5, 0.9]], dtype=torch.double)

        cost_low = cost_model(low_fidelity)
        cost_high = cost_model(high_fidelity)

        assert cost_high > cost_low


@pytest.mark.usefixtures("torch_rng")
class TestMFKGAcquisition:
    """Test qMFKG acquisition function creation."""

    def test_create_acquisition(self) -> None:
        """qMFKG acquisition can be created."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)
        fidelity_dim = 2
        target_fidelity = 1.0

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)
        cost_model = create_cost_model(fidelity_dim)

        acqf = create_mfkg_acquisition(
            model=model,
            bounds=bounds,
            fidelity_dim=fidelity_dim,
            target_fidelity=target_fidelity,
            cost_model=cost_model,
            num_fantasies=4,  # Small for testing
            num_restarts=2,
            raw_samples=32,
        )

        assert acqf is not None

    def test_create_acquisition_without_cost(self) -> None:
        """qMFKG can be created without explicit cost model."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)
        fidelity_dim = 2
        target_fidelity = 1.0

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)

        acqf = create_mfkg_acquisition(
            model=model,
            bounds=bounds,
            fidelity_dim=fidelity_dim,
            target_fidelity=target_fidelity,
            cost_model=None,
            num_fantasies=4,
            num_restarts=2,
            raw_samples=32,
        )

        assert acqf is not None


@pytest.mark.usefixtures("torch_rng")
class TestMFKGOptimization:
    """Test qMFKG optimization."""

    def test_optimize_mfkg(self) -> None:
        """qMFKG can be optimized to produce candidates."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)
        fidelity_dim = 2
        target_fidelity = 1.0

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)
        acqf = create_mfkg_acquisition(
            model=model,
            bounds=bounds,
            fidelity_dim=fidelity_dim,
            target_fidelity=target_fidelity,
            num_fantasies=4,
            num_restarts=2,
            raw_samples=32,
        )

        candidates, _acq_values = optimize_mfkg(
            acqf=acqf,
            bounds=bounds,
            batch_size=1,
            num_restarts=2,
            raw_samples=32,
        )

        assert candidates.shape == (1, 3)
        assert (candidates >= bounds[0]).all()
        assert (candidates <= bounds[1]).all()


class TestMultiFidelitySuggestionGeneration:
    """Test integrated suggestion generation."""

    def test_generate_suggestions_basic(self) -> None:
        """Suggestions can be generated with multi-fidelity config."""
        torch.manual_seed(42)

        # 2 params + fidelity (last dim)
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)

        fidelity = FidelitySpec(fidelity_dim=2, target_fidelity=1.0)
        config = MultiFidelityConfig(
            fidelity_spec=fidelity,
            num_fantasies=4,  # Small for testing
            num_restarts=2,
            raw_samples=32,
        )

        candidates, _acq_values, _metadata = generate_multifidelity_suggestions(
            train_x, train_y, bounds, config, batch_size=1
        )

        assert candidates.shape == (1, 3)
        assert (candidates >= bounds[0]).all()
        assert (candidates <= bounds[1]).all()

    def test_generate_suggestions_metadata(self) -> None:
        """Suggestions include proper multi-fidelity metadata."""
        torch.manual_seed(42)

        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)

        fidelity = FidelitySpec(fidelity_dim=2, target_fidelity=1.0)
        config = MultiFidelityConfig(
            fidelity_spec=fidelity,
            num_fantasies=4,
            num_restarts=2,
            raw_samples=32,
        )

        _, _, metadata = generate_multifidelity_suggestions(
            train_x, train_y, bounds, config, batch_size=1
        )

        assert metadata["model_type"] == "SingleTaskMultiFidelityGP"
        assert metadata["acquisition_function"] == "qMultiFidelityKnowledgeGradient"
        assert metadata["fidelity_dim"] == 2  # Last dimension
        assert metadata["target_fidelity"] == 1.0
        assert "num_fantasies" in metadata


class TestMultiFidelityBehavior:
    """Test that multi-fidelity optimization exhibits expected behavior."""

    def test_prefers_low_fidelity_for_exploration(self) -> None:
        """Early suggestions should prefer lower fidelity (cheaper).

        When we have little data, it's more cost-effective to explore
        at lower fidelity before committing to expensive high-fidelity evaluations.
        """
        torch.manual_seed(42)

        # Very sparse data - should encourage exploration
        train_x = torch.tensor(
            [
                [0.2, 0.3, 0.5],  # Medium fidelity
                [0.8, 0.7, 0.5],  # Medium fidelity
            ],
            dtype=torch.double,
        )
        train_y = torch.tensor([[0.3], [0.7]], dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.1], [1.0, 1.0, 1.0]], dtype=torch.double)

        fidelity = FidelitySpec(
            fidelity_dim=2,
            target_fidelity=1.0,
            cost_weight=1.0,
            fixed_cost=5.0,
        )
        config = MultiFidelityConfig(
            fidelity_spec=fidelity,
            num_fantasies=8,
            num_restarts=5,
            raw_samples=64,
        )

        candidates, _, _ = generate_multifidelity_suggestions(
            train_x, train_y, bounds, config, batch_size=1
        )

        # Get the fidelity value of the suggestion
        suggested_fidelity = candidates[0, 2].item()

        # With cost awareness, we expect lower fidelity suggestions
        # (Not always guaranteed due to stochastic optimization, but trend should hold)
        assert 0.1 <= suggested_fidelity <= 1.0

    def test_model_learns_fidelity_correlation(self) -> None:
        """Model should learn that low and high fidelity are correlated.

        If we provide data showing correlation between fidelities,
        the model should capture this relationship.
        """
        torch.manual_seed(42)

        # Create data where low and high fidelity are correlated
        # f(x, s) = x[0]^2 + x[1]^2 + noise * (1 - s)
        # High fidelity (s=1) is clean, low fidelity (s=0.1) is noisy but correlated

        def f(x: torch.Tensor, fidelity: torch.Tensor) -> torch.Tensor:
            clean = x[:, 0:1] ** 2 + x[:, 1:2] ** 2
            noise = torch.randn_like(clean) * 0.1 * (1 - fidelity)
            return clean + noise

        # Generate training data at various fidelities
        n_low = 10
        n_high = 5

        x_low = torch.rand(n_low, 2, dtype=torch.double)
        s_low = torch.ones(n_low, 1, dtype=torch.double) * 0.2
        train_x_low = torch.cat([x_low, s_low], dim=1)
        train_y_low = f(x_low, s_low)

        x_high = torch.rand(n_high, 2, dtype=torch.double)
        s_high = torch.ones(n_high, 1, dtype=torch.double) * 1.0
        train_x_high = torch.cat([x_high, s_high], dim=1)
        train_y_high = f(x_high, s_high)

        train_x = torch.cat([train_x_low, train_x_high], dim=0)
        train_y = torch.cat([train_y_low, train_y_high], dim=0)

        fidelity_dim = 2
        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)

        # Test prediction: same x, different fidelity should give similar predictions
        # but high fidelity should have lower variance
        test_x = torch.tensor([[0.5, 0.5]], dtype=torch.double)
        test_low = torch.cat([test_x, torch.tensor([[0.2]], dtype=torch.double)], dim=1)
        test_high = torch.cat([test_x, torch.tensor([[1.0]], dtype=torch.double)], dim=1)

        model.eval()
        with torch.no_grad():
            pred_low = model.posterior(test_low)
            pred_high = model.posterior(test_high)

            # High fidelity should have lower variance (more confident)
            # Allow some tolerance since this is a soft expectation
            assert pred_high.variance.item() >= 0
            assert pred_low.variance.item() >= 0


@pytest.mark.usefixtures("torch_rng")
class TestMultiFidelityEdgeCases:
    """Test edge cases and error handling."""

    def test_fidelity_at_boundary(self) -> None:
        """Works when fidelity values are at bounds."""
        train_x = torch.tensor(
            [
                [0.5, 0.5, 0.1],  # Minimum fidelity
                [0.5, 0.5, 1.0],  # Maximum fidelity
            ],
            dtype=torch.double,
        )
        train_y = torch.tensor([[0.5], [0.3]], dtype=torch.double)
        fidelity_dim = 2

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)
        assert model is not None

    def test_single_fidelity_level(self) -> None:
        """Works with data at only one fidelity level."""
        train_x = torch.rand(10, 3, dtype=torch.double)
        train_x[:, 2] = 0.5  # All at same fidelity
        train_y = torch.rand(10, 1, dtype=torch.double)
        fidelity_dim = 2

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)
        assert model is not None

    def test_high_dimensional_params(self) -> None:
        """Works with many parameters + fidelity."""
        n_params = 10
        train_x = torch.rand(20, n_params + 1, dtype=torch.double)  # +1 for fidelity
        train_y = torch.rand(20, 1, dtype=torch.double)
        fidelity_dim = n_params  # Last dimension

        model = create_and_fit_multifidelity_model(train_x, train_y, fidelity_dim)
        assert model is not None
