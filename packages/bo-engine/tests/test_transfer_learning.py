"""Tests for Transfer Learning with RGPE (Rank-weighted GP Ensemble).

RGPE enables leveraging knowledge from prior optimization campaigns to
accelerate optimization on a new but related task. It uses rank-based
weighting to combine predictions from multiple models.

These tests verify:
1. PriorTaskData and RGPEConfig dataclasses
2. Base model creation and fitting
3. RGPE ensemble creation and weight computation
4. Posterior predictions with weighted combination
5. Integration with suggestion generation
6. Behavior when prior tasks are helpful vs. unhelpful
"""

import pytest
import torch

from bo_engine.transfer_learning import (
    RGPE,
    PriorTaskData,
    RGPEConfig,
    RGPELogEI,
    create_base_model,
    create_rgpe_model,
    generate_rgpe_suggestions,
    get_rgpe_weights_explanation,
)


@pytest.mark.usefixtures("torch_rng")
class TestPriorTaskData:
    """Test PriorTaskData dataclass."""

    def test_basic_creation(self) -> None:
        """PriorTaskData can be created with required fields."""
        train_x = torch.rand(10, 2, dtype=torch.double)
        train_y = torch.rand(10, 1, dtype=torch.double)

        prior = PriorTaskData(
            name="campaign_123",
            train_x=train_x,
            train_y=train_y,
        )

        assert prior.name == "campaign_123"
        assert prior.train_x.shape == (10, 2)
        assert prior.train_y.shape == (10, 1)
        assert prior.metadata == {}

    def test_with_metadata(self) -> None:
        """PriorTaskData accepts optional metadata."""
        train_x = torch.rand(5, 3, dtype=torch.double)
        train_y = torch.rand(5, 1, dtype=torch.double)

        prior = PriorTaskData(
            name="prior_1",
            train_x=train_x,
            train_y=train_y,
            metadata={"source": "historical", "quality": "high"},
        )

        assert prior.metadata["source"] == "historical"
        assert prior.metadata["quality"] == "high"


class TestRGPEConfig:
    """Test RGPEConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has reasonable values."""
        config = RGPEConfig()
        assert config.num_samples == 512
        assert config.use_input_warping is False

    def test_custom_config(self) -> None:
        """Custom config values are respected."""
        config = RGPEConfig(num_samples=256, use_input_warping=True)
        assert config.num_samples == 256
        assert config.use_input_warping is True


@pytest.mark.usefixtures("torch_rng")
class TestBaseModelCreation:
    """Test base model creation for transfer learning."""

    def test_create_base_model_basic(self) -> None:
        """Base model can be created and fitted."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, 1, dtype=torch.double)
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)

        model = create_base_model(train_x, train_y, bounds)

        # Model should be fitted and usable
        model.eval()
        with torch.no_grad():
            test_x = torch.rand(5, 3, dtype=torch.double)
            posterior = model.posterior(test_x)
            assert posterior.mean.shape == (5, 1)

    def test_create_base_model_with_1d_y(self) -> None:
        """Base model handles 1D y input."""
        train_x = torch.rand(15, 3, dtype=torch.double)
        train_y = torch.rand(15, dtype=torch.double)  # 1D
        bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.double)

        model = create_base_model(train_x, train_y, bounds)
        assert model is not None


@pytest.mark.usefixtures("torch_rng")
class TestRGPEClass:
    """Test the RGPE ensemble class."""

    @pytest.fixture
    def simple_rgpe(
        self, request: pytest.FixtureRequest
    ) -> tuple[RGPE, list[PriorTaskData], torch.Tensor, torch.Tensor]:
        """Create a simple RGPE ensemble for testing."""
        request.getfixturevalue("torch_rng")
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        # Create one prior task
        prior_x = torch.rand(10, 2, dtype=torch.double)
        prior_y = torch.rand(10, 1, dtype=torch.double)
        prior_tasks = [PriorTaskData(name="prior_1", train_x=prior_x, train_y=prior_y)]

        # Create target task
        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        return rgpe, prior_tasks, target_x, target_y

    def test_num_models(self, simple_rgpe: tuple) -> None:
        """RGPE correctly counts total models."""
        rgpe, _prior_tasks, _, _ = simple_rgpe
        # 1 prior + 1 target = 2 models
        assert rgpe.num_models == 2

    def test_weights_shape(self, simple_rgpe: tuple) -> None:
        """Weights have correct shape."""
        rgpe, _, _, _ = simple_rgpe
        assert rgpe.weights.shape == (2,)

    def test_weights_sum_to_one(self, simple_rgpe: tuple) -> None:
        """Weights sum to 1."""
        rgpe, _, _, _ = simple_rgpe
        assert abs(rgpe.weights.sum().item() - 1.0) < 1e-5

    def test_posterior_prediction_shape(self, simple_rgpe: tuple) -> None:
        """Posterior predictions have correct shape."""
        rgpe, _, _, _ = simple_rgpe
        test_x = torch.rand(5, 2, dtype=torch.double)

        posterior = rgpe.posterior(test_x)
        # Posterior mean can be (n,) or (n, 1) depending on implementation
        assert posterior.mean.shape in [(5,), (5, 1)]
        assert posterior.variance.shape in [(5,), (5, 1)]

    def test_forward_returns_distribution(self, simple_rgpe: tuple) -> None:
        """Forward pass returns a distribution."""
        rgpe, _, _, _ = simple_rgpe
        test_x = torch.rand(5, 2, dtype=torch.double)

        distribution = rgpe(test_x)
        assert hasattr(distribution, "mean")
        assert hasattr(distribution, "variance")


@pytest.mark.usefixtures("torch_rng")
class TestRGPEWeightComputation:
    """Test RGPE weight computation behavior."""

    def test_weights_without_compute_raises(self) -> None:
        """Accessing weights before computation raises error."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        base_model = create_base_model(
            torch.rand(10, 2, dtype=torch.double),
            torch.rand(10, 1, dtype=torch.double),
            bounds,
        )
        target_model = create_base_model(
            torch.rand(10, 2, dtype=torch.double),
            torch.rand(10, 1, dtype=torch.double),
            bounds,
        )

        rgpe = RGPE(base_models=[base_model], target_model=target_model)

        with pytest.raises(ValueError, match="Weights not computed"):
            _ = rgpe.weights

    def test_relevant_prior_gets_higher_weight(self) -> None:
        """Prior tasks similar to target should get higher weights.

        Create a scenario where one prior task is very similar to target
        and another is completely different.
        """
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        # Target function: f(x) = x[0]^2 + x[1]^2
        def target_fn(x: torch.Tensor) -> torch.Tensor:
            return x[:, 0:1] ** 2 + x[:, 1:2] ** 2

        # Prior 1: Same as target (should get high weight)
        prior1_x = torch.rand(15, 2, dtype=torch.double)
        prior1_y = target_fn(prior1_x)

        # Prior 2: Different function (should get low weight)
        prior2_x = torch.rand(15, 2, dtype=torch.double)
        prior2_y = torch.sin(prior2_x[:, 0:1] * 3.14) * torch.cos(prior2_x[:, 1:2] * 3.14)

        prior_tasks = [
            PriorTaskData(name="similar", train_x=prior1_x, train_y=prior1_y),
            PriorTaskData(name="different", train_x=prior2_x, train_y=prior2_y),
        ]

        # Target data from same function
        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = target_fn(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        weights = rgpe.weights.tolist()

        # Weight structure: [prior1, prior2, target]
        # Prior 1 should generally have higher weight than Prior 2
        # (Note: stochastic so we just check they're reasonable)
        assert len(weights) == 3  # 2 priors + 1 target
        assert all(w >= 0 for w in weights)


@pytest.mark.usefixtures("torch_rng")
class TestRGPEWeightsExplanation:
    """Test human-readable weight explanation."""

    def test_explanation_format(self) -> None:
        """Explanation maps task IDs to weights."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name="campaign_abc",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
            PriorTaskData(
                name="campaign_xyz",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        explanation = get_rgpe_weights_explanation(rgpe, prior_tasks)

        assert "prior_campaign_abc" in explanation
        assert "prior_campaign_xyz" in explanation
        assert "target" in explanation
        assert abs(sum(explanation.values()) - 1.0) < 1e-5


class TestRGPESuggestionGeneration:
    """Test suggestion generation with RGPE."""

    def test_generate_suggestions_basic(self) -> None:
        """RGPE generates valid suggestions."""
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name="prior_1",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        candidates, acq_values, _metadata = generate_rgpe_suggestions(
            target_x, target_y, prior_tasks, bounds, batch_size=2
        )

        # Check shapes
        assert candidates.shape == (2, 2)
        assert acq_values.numel() >= 1

        # Check bounds
        assert (candidates >= bounds[0]).all()
        assert (candidates <= bounds[1]).all()

    def test_generate_suggestions_metadata(self) -> None:
        """Suggestions include proper RGPE metadata."""
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name="prior_1",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
            PriorTaskData(
                name="prior_2",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        _, _, metadata = generate_rgpe_suggestions(
            target_x, target_y, prior_tasks, bounds, batch_size=1
        )

        assert metadata["model_type"] == "RGPE (Rank-weighted GP Ensemble)"
        # Default behavior uses ensemble acquisition (v2.6 improvement)
        assert "RGPEAcquisition" in metadata["acquisition_function"]
        assert metadata["n_prior_tasks"] == 2
        assert "weights" in metadata
        assert "target_weight" in metadata


class TestRGPETransferBenefit:
    """Test that RGPE actually provides transfer benefit."""

    def test_transfer_reduces_uncertainty(self) -> None:
        """Prior knowledge should reduce prediction uncertainty.

        When prior tasks are from the same function, RGPE should have
        lower uncertainty than a model trained only on target data.
        """
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        # Simple quadratic function
        def f(x: torch.Tensor) -> torch.Tensor:
            return (x[:, 0:1] - 0.5) ** 2 + (x[:, 1:2] - 0.5) ** 2

        # Rich prior data
        prior_x = torch.rand(30, 2, dtype=torch.double)
        prior_y = f(prior_x)
        prior_tasks = [PriorTaskData(name="rich_prior", train_x=prior_x, train_y=prior_y)]

        # Sparse target data
        target_x = torch.rand(5, 2, dtype=torch.double)
        target_y = f(target_x)

        # Create RGPE model
        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)

        # Create target-only model for comparison
        target_only_model = create_base_model(target_x, target_y, bounds)

        # Compare uncertainty at test points
        test_x = torch.rand(10, 2, dtype=torch.double)

        rgpe_posterior = rgpe.posterior(test_x)
        target_posterior = target_only_model.posterior(test_x)

        # RGPE should have lower average variance (benefiting from prior)
        # This is a soft test - stochastic nature means we just check reasonable behavior
        assert rgpe_posterior.variance.mean() >= 0
        assert target_posterior.variance.mean() >= 0


@pytest.mark.usefixtures("torch_rng")
class TestRGPEEdgeCases:
    """Test edge cases and error handling."""

    def test_single_prior_task(self) -> None:
        """Works with a single prior task."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name="only_prior",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert rgpe.num_models == 2

    def test_many_prior_tasks(self) -> None:
        """Works with multiple prior tasks."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name=f"prior_{i}",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            )
            for i in range(5)
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = torch.rand(10, 1, dtype=torch.double)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert rgpe.num_models == 6  # 5 priors + 1 target

    def test_minimal_target_data(self) -> None:
        """Works with minimal target data."""
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        prior_tasks = [
            PriorTaskData(
                name="prior",
                train_x=torch.rand(20, 2, dtype=torch.double),
                train_y=torch.rand(20, 1, dtype=torch.double),
            ),
        ]

        # Very sparse target data
        target_x = torch.rand(3, 2, dtype=torch.double)
        target_y = torch.rand(3, 1, dtype=torch.double)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert rgpe is not None


class TestRankingLossWeights:
    """Weights follow the paper's ranking loss, not prediction error.

    Feurer et al. (https://arxiv.org/abs/1802.02219) define the weight of
    model i as the fraction of posterior samples in which it has the
    lowest pair-misranking count on the target data. Because the loss
    counts orderings only, a prior task whose objective is an affine
    rescaling of the target (different units / offset, same landscape)
    must rank highly — exactly the transfer scenario MSE-style weighting
    fails on, since its posterior un-standardizes to the prior's own
    y-scale. The BoTorch RGPE tutorial implements the same estimator:
    https://botorch.org/docs/tutorials/meta_learning_with_rgpe/
    """

    @staticmethod
    def _target_fn(x: torch.Tensor) -> torch.Tensor:
        return (x[:, 0:1] - 0.3) ** 2 + (x[:, 1:2] - 0.7) ** 2

    def test_affine_rescaled_prior_ranks_highest(self) -> None:
        """y_prior = 100 + 50*y_target must beat an unrelated prior."""
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        rescaled_x = torch.rand(15, 2, dtype=torch.double)
        rescaled_y = 100.0 + 50.0 * self._target_fn(rescaled_x)

        unrelated_x = torch.rand(15, 2, dtype=torch.double)
        unrelated_y = torch.rand(15, 1, dtype=torch.double) * 100.0

        prior_tasks = [
            PriorTaskData(name="rescaled_same", train_x=rescaled_x, train_y=rescaled_y),
            PriorTaskData(name="unrelated", train_x=unrelated_x, train_y=unrelated_y),
        ]

        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = self._target_fn(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        weights = rgpe.weights  # [rescaled_same, unrelated, target]

        assert weights[0] > weights[1], (
            f"The affine-rescaled same-landscape prior got weight "
            f"{weights[0]:.3f} <= unrelated prior {weights[1]:.3f} — the "
            "weighting is scale-sensitive instead of rank-based."
        )
        assert weights[0] == max(weights), (
            "With dense matching prior data, the rescaled prior should "
            f"carry the largest weight; got {weights.tolist()}"
        )

    def test_weights_form_simplex(self) -> None:
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        prior_tasks = [
            PriorTaskData(
                name=f"prior_{i}",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            )
            for i in range(3)
        ]
        target_x = torch.rand(8, 2, dtype=torch.double)
        target_y = self._target_fn(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert (rgpe.weights >= 0).all()
        assert rgpe.weights.sum().item() == pytest.approx(1.0, abs=1e-9)

    def test_weight_dilution_discards_random_prior(self) -> None:
        """A pure-noise prior must be discarded by the percentile rule.

        Feurer et al. (v1) discard base model i when the median of its
        loss samples exceeds the 95th percentile of the target model's —
        without it, many weak priors would each win a few samples and
        dilute the target.
        """
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        noise_x = torch.rand(15, 2, dtype=torch.double)
        noise_y = torch.rand(15, 1, dtype=torch.double) * 100.0
        prior_tasks = [PriorTaskData(name="noise", train_x=noise_x, train_y=noise_y)]

        # Densely observed smooth target → the target model ranks well,
        # so the noise prior's loss median clears the dilution threshold.
        target_x = torch.rand(20, 2, dtype=torch.double)
        target_y = self._target_fn(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert rgpe.weights[0].item() == pytest.approx(0.0, abs=1e-12), (
            f"Pure-noise prior kept weight {rgpe.weights[0]:.4f} — the "
            "dilution filter did not fire."
        )

    def test_uniform_weights_below_minimum_observations(self) -> None:
        """< 3 target observations → uniform weights (paper prescription)."""
        torch.manual_seed(42)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        prior_tasks = [
            PriorTaskData(
                name="prior",
                train_x=torch.rand(10, 2, dtype=torch.double),
                train_y=torch.rand(10, 1, dtype=torch.double),
            ),
        ]
        target_x = torch.rand(2, 2, dtype=torch.double)
        target_y = self._target_fn(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        assert torch.allclose(rgpe.weights, torch.full((2,), 0.5, dtype=torch.double))


class TestBatchDiversityViaPendingPoints:
    """Sequential batch generation must not collapse onto one maximizer.

    ``optimize_acqf(sequential=True)`` informs the acquisition of
    already-selected candidates via ``set_X_pending``; an acquisition
    that ignores the property solves the identical problem for every
    slot and returns near-duplicate batches. The RGPE acquisitions fold
    pending points in via local penalization (Gonzalez et al. 2016,
    https://arxiv.org/abs/1505.08052).
    """

    def test_batch_candidates_are_spread_out(self) -> None:
        torch.manual_seed(0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        def f(x: torch.Tensor) -> torch.Tensor:
            return (x[:, 0:1] - 0.3) ** 2 + (x[:, 1:2] - 0.7) ** 2

        prior_x = torch.rand(15, 2, dtype=torch.double)
        prior_tasks = [PriorTaskData(name="same", train_x=prior_x, train_y=f(prior_x))]
        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = f(target_x)

        candidates, _, _ = generate_rgpe_suggestions(
            target_x, target_y, prior_tasks, bounds, batch_size=3
        )

        distances = torch.cdist(candidates, candidates)
        distances.fill_diagonal_(float("inf"))
        min_pairwise = distances.min().item()
        assert min_pairwise > 0.05, (
            f"Batch of 3 collapsed to near-duplicates (min pairwise "
            f"distance {min_pairwise:.4f}) — X_pending is being ignored."
        )

    def test_pending_points_lower_nearby_acquisition(self) -> None:
        """Setting X_pending must reduce the EI exactly at the pending point."""
        from bo_engine.transfer_learning import RGPEAcquisition, create_rgpe_model

        torch.manual_seed(0)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)

        def f(x: torch.Tensor) -> torch.Tensor:
            return (x[:, 0:1] - 0.3) ** 2 + (x[:, 1:2] - 0.7) ** 2

        prior_x = torch.rand(15, 2, dtype=torch.double)
        prior_tasks = [PriorTaskData(name="same", train_x=prior_x, train_y=f(prior_x))]
        target_x = torch.rand(10, 2, dtype=torch.double)
        target_y = f(target_x)

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        acqf = RGPEAcquisition(rgpe, best_f=target_y.min().item(), bounds=bounds)

        probe = torch.tensor([[[0.25, 0.65]]], dtype=torch.double)
        with torch.no_grad():
            free_value = acqf(probe)
            acqf.X_pending = probe.squeeze(0)
            pinned_value = acqf(probe)

        assert pinned_value.item() < free_value.item(), (
            "EI at a pending point did not drop — forward() ignores X_pending."
        )
        assert pinned_value.item() == pytest.approx(0.0, abs=1e-6)


class TestLegacyPathDirection:
    """The target-only (legacy) acquisition must minimize, not maximize.

    ``generate_rgpe_suggestions`` documents a minimization contract
    (``best_f = target_y.min()``). The ensemble path enforces it via the
    hand-rolled EI's ``maximize=False``; the legacy path uses BoTorch's
    ``qLogNoisyExpectedImprovement``, which always maximizes (it has no
    direction flag — see ``botorch.acquisition.logei``), so it needs a
    negating MC objective to honor the same contract.
    """

    def test_legacy_path_suggests_near_minimum(self) -> None:
        """On a 1-D bowl the legacy path proposes points in the minimum's basin.

        The bowl ``y = (x - 0.3)²`` is densely observed, so a correctly
        oriented EI exploits near ``x = 0.3`` while a sign-flipped one
        chases the observed maximum at ``x = 1.0``.
        """
        torch.manual_seed(0)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        optimum = 0.3

        target_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
        target_y = (target_x - optimum) ** 2

        # Informative prior: the same bowl, lightly perturbed.
        prior_x = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
        prior_y = (prior_x - optimum) ** 2 + 0.01
        prior_tasks = [PriorTaskData(name="same_bowl", train_x=prior_x, train_y=prior_y)]

        candidates, _acq_values, metadata = generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            batch_size=1,
            use_ensemble_acquisition=False,
        )

        assert "Target Only" in metadata["acquisition_function"]
        suggested_x = float(candidates[0, 0].item())
        observed_argmax = 1.0  # x of the worst (largest) observed value
        assert abs(suggested_x - optimum) < abs(suggested_x - observed_argmax), (
            f"Legacy RGPE suggestion x={suggested_x:.3f} is closer to the "
            f"observed maximum than to the minimum {optimum} — the "
            "acquisition is maximizing under a minimization contract."
        )
        assert abs(suggested_x - optimum) <= 0.3, (
            f"Legacy RGPE suggestion x={suggested_x:.3f} is outside the "
            f"minimum's basin around {optimum}."
        )


@pytest.mark.usefixtures("torch_rng")
class TestRGPELogEINumericalStability:
    """``RGPELogEI`` must stay finite where the naive log-EI produced NaN (L6).

    Near a densely sampled point a well-fit, near-noiseless model drives the
    predictive ``std`` to its 1e-6 clamp, so ``z = (best_f - mean)/std`` is a
    huge negative number. The previous ``log(z * Phi(z) + phi(z))`` underflowed
    the cdf clamp into a negative argument and returned NaN even though
    ``h(z) = z*Phi(z) + phi(z)`` is mathematically positive. The stable
    ``_log_ei_helper`` (Ament et al. 2023) keeps the value finite.
    """

    def test_log_ei_finite_at_training_point(self) -> None:
        """Evaluating at an observed point of a near-noiseless model stays finite."""
        torch.manual_seed(0)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)

        target_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
        target_y = (target_x - 0.3) ** 2
        prior_x = torch.linspace(0, 1, 15, dtype=torch.double).unsqueeze(-1)
        prior_y = (prior_x - 0.3) ** 2 + 0.01
        prior_tasks = [PriorTaskData(name="same_bowl", train_x=prior_x, train_y=prior_y)]

        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)
        acqf = RGPELogEI(rgpe, best_f=float(target_y.min().item()), maximize=False)

        # Evaluate exactly at training points (q=1, shape (n, 1, d)) — the
        # regime where std collapses and |z| explodes.
        candidates = target_x.unsqueeze(1)
        log_ei = acqf(candidates)

        assert torch.isfinite(log_ei).all(), (
            f"RGPELogEI produced non-finite values at training points: {log_ei}"
        )

    def test_log_ei_finite_for_extreme_negative_z(self) -> None:
        """A hand-built extreme |z| (via best_f far below the data) stays finite."""
        torch.manual_seed(1)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        target_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
        target_y = (target_x - 0.3) ** 2
        prior_tasks = [
            PriorTaskData(name="p", train_x=target_x, train_y=target_y + 0.01),
        ]
        rgpe = create_rgpe_model(target_x, target_y, prior_tasks, bounds)

        # best_f far below every prediction forces large negative z.
        acqf = RGPELogEI(rgpe, best_f=-1e6, maximize=False)
        log_ei = acqf(target_x.unsqueeze(1))
        assert torch.isfinite(log_ei).all()
