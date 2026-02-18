"""Unit tests for batch diversity metrics with categorical parameters.

Tests the compute_batch_diversity function from batch_diversity.py with
both categorical and continuous parameter scenarios.

References:
    - Local Penalization in BO: https://arxiv.org/abs/1505.08052
    - BoTorch batch optimization: https://botorch.org/docs/batched_bayesian_optimization/
"""

import torch

from bo_engine.batch_diversity import compute_batch_diversity


class TestBatchDiversityCategorical:
    """Test batch diversity metrics for categorical (one-hot) parameters."""

    def test_identical_points_have_zero_distance(self) -> None:
        """Identical one-hot encoded points should have zero pairwise distance."""
        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.min_pairwise_distance == 0.0
        assert not metrics.is_diverse

    def test_different_categories_have_positive_distance(self) -> None:
        """Different one-hot encoded points should have positive distance."""
        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.min_pairwise_distance > 0.0

    def test_diversity_score_zero_for_duplicates(self) -> None:
        """Duplicate categorical points should have zero diversity score."""
        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.diversity_score == 0.0
        assert metrics.pairs_below_threshold > 0

    def test_diversity_score_positive_for_unique_categories(self) -> None:
        """Unique categorical points should have positive diversity score."""
        candidates = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.diversity_score > 0.0
        assert metrics.min_pairwise_distance > 0.0


class TestBatchDiversityContinuous:
    """Test batch diversity metrics for continuous parameters."""

    def test_well_spread_points_high_diversity(self) -> None:
        """Well-spread points should have high diversity."""
        candidates = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.min_pairwise_distance > 1.0  # sqrt(2) ~= 1.414
        assert metrics.is_diverse

    def test_clustered_points_low_diversity(self) -> None:
        """Clustered points should have low diversity."""
        candidates = torch.tensor(
            [
                [0.5, 0.5],
                [0.501, 0.501],
            ],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.min_pairwise_distance < 0.01

    def test_single_point_returns_trivial_metrics(self) -> None:
        """A single point should return trivial diversity metrics."""
        candidates = torch.tensor(
            [[0.5, 0.5]],
            dtype=torch.double,
        )
        bounds = torch.tensor(
            [[0.0, 0.0], [1.0, 1.0]],
            dtype=torch.double,
        )

        metrics = compute_batch_diversity(candidates, bounds)
        assert metrics.min_pairwise_distance == float("inf")
        assert metrics.is_diverse
        assert metrics.diversity_score == 1.0
