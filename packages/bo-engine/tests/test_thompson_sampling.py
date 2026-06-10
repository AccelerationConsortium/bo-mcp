"""Tests for Thompson Sampling RNG discipline and batch diversity.

``generate_thompson_samples`` isolates the global torch RNG in a
``fork_rng`` block, which RESTORES the pre-call state on exit. Without a
per-call seed, every unseeded call therefore replayed the identical
posterior draw, and ``generate_diverse_thompson_batch``'s retry loop —
which calls the sampler repeatedly expecting fresh draws — degenerated
into n copies of one point. These tests pin the repaired contract:

* consecutive unseeded calls draw fresh randomness,
* seeded calls stay exactly reproducible (mirroring the discipline in
  ``test_rng_isolation.py`` / ``test_fixture_seed_discipline.py``),
* diverse batches actually satisfy the min-distance property.

References:
    - Russo et al. "A Tutorial on Thompson Sampling" (2018),
      https://arxiv.org/abs/1707.02038 — repeated posterior draws must be
      independent for TS to explore.
    - torch.random.fork_rng semantics:
      https://pytorch.org/docs/stable/random.html#torch.random.fork_rng
"""

from __future__ import annotations

import pytest
import torch

from bo_engine.batch_diversity import compute_batch_diversity
from bo_engine.models import create_and_fit_single_task_model
from bo_engine.thompson_sampling import (
    ThompsonConfig,
    generate_diverse_thompson_batch,
    generate_thompson_samples,
)


@pytest.fixture(scope="module")
def fitted_model_and_bounds() -> tuple[object, torch.Tensor]:
    """A 1-D GP on a noisy bowl — enough posterior spread for distinct draws."""
    torch.manual_seed(1234)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_x = torch.linspace(0, 1, 12, dtype=torch.double).unsqueeze(-1)
    train_y = (train_x - 0.3) ** 2 + 0.05 * torch.randn_like(train_x)
    model = create_and_fit_single_task_model(train_x, train_y, bounds)
    return model, bounds


class TestPerCallRandomness:
    """Unseeded calls must not replay the identical draw."""

    def test_unseeded_calls_differ(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        first = generate_thompson_samples(model, bounds, n_samples=1)
        second = generate_thompson_samples(model, bounds, n_samples=1)
        assert not torch.equal(first.parameters_tensor, second.parameters_tensor), (
            "Two consecutive unseeded Thompson calls returned the identical "
            "sample — fork_rng restored the global state and no per-call "
            "seed was installed."
        )

    def test_unseeded_manual_calls_differ(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        config = ThompsonConfig(use_max_posterior_sampling=False)
        first = generate_thompson_samples(model, bounds, n_samples=1, config=config)
        second = generate_thompson_samples(model, bounds, n_samples=1, config=config)
        assert not torch.equal(first.parameters_tensor, second.parameters_tensor)


class TestSeededReproducibility:
    """A configured seed keeps the documented reproducibility contract."""

    def test_seeded_calls_identical(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        first = generate_thompson_samples(model, bounds, n_samples=3, config=ThompsonConfig(seed=7))
        second = generate_thompson_samples(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=7)
        )
        assert torch.equal(first.parameters_tensor, second.parameters_tensor)

    def test_different_seeds_differ(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        first = generate_thompson_samples(model, bounds, n_samples=3, config=ThompsonConfig(seed=7))
        second = generate_thompson_samples(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=8)
        )
        assert not torch.equal(first.parameters_tensor, second.parameters_tensor)

    def test_seeded_call_leaves_global_rng_untouched(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        torch.manual_seed(99)
        before = torch.random.get_rng_state()
        generate_thompson_samples(model, bounds, n_samples=2, config=ThompsonConfig(seed=7))
        after = torch.random.get_rng_state()
        assert torch.equal(before, after), "fork_rng isolation was lost"


class TestDiverseBatch:
    """The diversity retry loop must produce spread-out batches."""

    def test_diverse_batch_min_distance(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        min_distance = 0.02
        batch = generate_diverse_thompson_batch(
            model,
            bounds,
            n_samples=4,
            min_distance=min_distance,
            config=ThompsonConfig(seed=7),
        )
        metrics = compute_batch_diversity(batch.parameters_tensor, bounds, min_distance)
        assert metrics.min_pairwise_distance >= min_distance, (
            f"Diverse batch has min pairwise distance "
            f"{metrics.min_pairwise_distance:.4f} < {min_distance} — the "
            "retry loop is replaying identical draws (diversity no-op)."
        )
        assert batch.diversity_score > 0.0

    def test_diverse_batch_has_no_duplicates(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        batch = generate_diverse_thompson_batch(
            model, bounds, n_samples=4, config=ThompsonConfig(seed=11)
        )
        params = batch.parameters_tensor
        for i in range(params.shape[0]):
            for j in range(i + 1, params.shape[0]):
                assert not torch.allclose(params[i], params[j]), (
                    f"Batch slots {i} and {j} are identical points"
                )

    def test_seeded_diverse_batch_reproducible(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        first = generate_diverse_thompson_batch(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=21)
        )
        second = generate_diverse_thompson_batch(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=21)
        )
        assert torch.equal(first.parameters_tensor, second.parameters_tensor)
