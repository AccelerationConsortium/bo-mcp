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


class TestLocalPenalizationSignSafety:
    """Penalization must reduce attractiveness regardless of value sign.

    The multiplicative penalizer of Gonzalez et al.
    (https://arxiv.org/abs/1505.08052) assumes strictly positive
    acquisition values. The log-EI family used elsewhere in the engine is
    frequently negative, where a bare ``acq * (1 - p)`` makes values near
    selected points LESS negative — i.e. more attractive. The helper now
    penalizes the height above the batch minimum, which is sign-safe.
    """

    @staticmethod
    def _setup() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
        selected = torch.tensor([[0.5]], dtype=torch.double)
        bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
        return candidates, selected, bounds

    def test_negative_values_never_become_more_attractive(self) -> None:
        from bo_engine.batch_diversity import apply_local_penalization

        candidates, selected, bounds = self._setup()
        acq_values = torch.tensor([-5.0, -1.0, -4.0], dtype=torch.double)

        penalized = apply_local_penalization(acq_values, candidates, selected, bounds)

        assert (penalized <= acq_values + 1e-12).all(), (
            f"Penalization increased an acquisition value: {acq_values.tolist()} "
            f"-> {penalized.tolist()}"
        )
        # The candidate at the selected point must drop to the least
        # attractive level of the batch, not float toward zero.
        assert penalized[1].item() < acq_values[1].item()
        assert penalized[1].item() >= acq_values.min().item() - 1e-12

    def test_positive_values_keep_classic_behavior_shape(self) -> None:
        from bo_engine.batch_diversity import apply_local_penalization

        candidates, selected, bounds = self._setup()
        acq_values = torch.tensor([0.0, 3.0, 1.0], dtype=torch.double)

        penalized = apply_local_penalization(acq_values, candidates, selected, bounds)

        assert (penalized <= acq_values + 1e-12).all()
        # Far-away candidates are essentially untouched.
        assert penalized[0].item() == acq_values[0].item()
        assert abs(penalized[2].item() - acq_values[2].item()) < 1e-3
        # The selected point's value collapses toward the batch floor.
        assert penalized[1].item() < 0.1

    def test_no_selected_points_is_identity(self) -> None:
        from bo_engine.batch_diversity import apply_local_penalization

        candidates, _, bounds = self._setup()
        acq_values = torch.tensor([-2.0, 1.0, 0.5], dtype=torch.double)
        empty = torch.empty(0, 1, dtype=torch.double)

        penalized = apply_local_penalization(acq_values, candidates, empty, bounds)
        assert torch.equal(penalized, acq_values)
