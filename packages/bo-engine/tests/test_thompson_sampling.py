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
from bo_engine.constants import (
    THOMPSON_CANDIDATES_PER_DIM,
    THOMPSON_MAX_CANDIDATES,
    THOMPSON_MIN_CANDIDATES,
)
from bo_engine.models import create_and_fit_model, create_and_fit_single_task_model
from bo_engine.thompson_sampling import (
    ThompsonConfig,
    generate_diverse_thompson_batch,
    generate_thompson_samples,
    generate_thompson_samples_multi_objective,
    resolve_thompson_num_candidates,
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


class TestDimensionAdaptiveCandidateCount:
    """Discrete-TS candidate sizing follows the TuRBO tutorial formula.

    A fixed 1000-point cloud in >=10-d is so sparse the posterior-sample
    argmax is nearly model-independent; the TuRBO tutorial sizes its
    discrete TS candidate set ``min(5000, max(2000, 200 * d))``
    (https://botorch.org/tutorials/turbo_1/), which the resolver mirrors
    via the ``THOMPSON_*`` constants.
    """

    def test_count_scales_with_dimension(self) -> None:
        counts = [resolve_thompson_num_candidates(d) for d in (1, 10, 20, 50)]
        assert counts == sorted(counts)
        assert counts[0] == THOMPSON_MIN_CANDIDATES
        # 20-d sits in the linear regime; 50-d saturates at the cap.
        assert counts[2] == THOMPSON_CANDIDATES_PER_DIM * 20
        assert counts[3] == THOMPSON_MAX_CANDIDATES

    def test_explicit_request_overrides_the_formula(self) -> None:
        assert resolve_thompson_num_candidates(50, requested=64) == 64

    def test_method_info_reports_the_resolved_count(self, fitted_model_and_bounds: tuple) -> None:
        model, bounds = fitted_model_and_bounds
        batch = generate_thompson_samples(model, bounds, n_samples=1, config=ThompsonConfig(seed=5))
        expected = resolve_thompson_num_candidates(bounds.shape[1])
        assert f"n_candidates={expected}" in batch.method_info


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


class TestSampledValueIsWinningDraw:
    """``ThompsonSample.sampled_value`` must be the winning sample, not a re-draw.

    The reported value should be the value of the posterior sample path whose
    optimum selected the point — recomputing a *fresh* independent draw at the
    point (the previous behaviour) returns a number uncorrelated with why the
    point won, and is statistically far from the selecting minimum.

    Reference: Russo et al. "A Tutorial on Thompson Sampling" (2018) — the
    chosen action is the argmax/argmin of a single sampled reward function;
    the reported reward is that sample's value, not a new sample.
    """

    def test_sampled_value_matches_recorded_winning_draw(
        self, fitted_model_and_bounds: tuple
    ) -> None:
        from bo_engine.thompson_sampling import _generate_sobol_candidates

        model, bounds = fitted_model_and_bounds
        seed = 4242
        num_candidates = 64

        batch = generate_thompson_samples(
            model,
            bounds,
            n_samples=1,
            config=ThompsonConfig(
                seed=seed, num_candidates=num_candidates, use_max_posterior_sampling=True
            ),
            minimize=True,
        )
        sample = batch.samples[0]

        # Replay the exact internal selection: same seed, same candidate set,
        # same single posterior draw. fork_rng keeps the global state clean.
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            candidates = _generate_sobol_candidates(bounds, num_candidates)
            with torch.no_grad():
                sampled_values = model.posterior(candidates).rsample().reshape(-1)
            best_idx = int(sampled_values.argmin().item())
            expected_value = float(sampled_values[best_idx].item())
            expected_point = candidates[best_idx]

        assert sample.sampled_value == pytest.approx(expected_value), (
            "sampled_value is not the winning sample-path value — it looks "
            "like an independent fresh draw at the selected point."
        )
        assert torch.allclose(sample.parameters, expected_point)

    def test_sampled_value_not_a_fresh_independent_draw(
        self, fitted_model_and_bounds: tuple
    ) -> None:
        """The winning (minimum) draw should sit at/below the posterior mean.

        A fresh draw at the point is symmetric about the mean, so it lands
        above the mean ~half the time; the selecting minimum over many
        candidates is essentially never above it. This distinguishes the two
        behaviours without replaying the RNG.
        """
        model, bounds = fitted_model_and_bounds
        sample = generate_thompson_samples(
            model,
            bounds,
            n_samples=1,
            config=ThompsonConfig(seed=7, num_candidates=256),
            minimize=True,
        ).samples[0]

        assert sample.sampled_value <= sample.posterior_mean + sample.posterior_std, (
            "The winning minimum draw is implausibly high — sampled_value is "
            "not the selecting sample path."
        )


class TestDirectBatchNoDuplicates:
    """A direct multi-sample batch over a shared candidate set must not repeat.

    The shared-candidate path restores ``MaxPosteriorSampling(replacement=
    False)``: selecting from one candidate set without removing chosen points
    let independent argmin/argmax draws collapse onto the same candidate
    (observed: ``seed=1, num_candidates=16, n_samples=5`` → 4 repeats). The
    diverse-batch API was protected, but the direct API was not.
    """

    @pytest.mark.parametrize("minimize", [True, False])
    def test_no_duplicate_rows_in_direct_batch(
        self, fitted_model_and_bounds: tuple, minimize: bool
    ) -> None:
        model, bounds = fitted_model_and_bounds
        batch = generate_thompson_samples(
            model,
            bounds,
            n_samples=5,
            config=ThompsonConfig(seed=1, num_candidates=16),
            minimize=minimize,
        )
        params = batch.parameters_tensor
        for i in range(params.shape[0]):
            for j in range(i + 1, params.shape[0]):
                assert not torch.allclose(params[i], params[j]), (
                    f"Direct Thompson batch slots {i} and {j} are identical — "
                    "without-replacement selection regressed."
                )


class TestSingleCandidateConfig:
    """``num_candidates=1`` is a valid public config and must not crash."""

    @pytest.mark.parametrize("minimize", [True, False])
    def test_single_candidate_returns_finite_sample(
        self, fitted_model_and_bounds: tuple, minimize: bool
    ) -> None:
        """A single-candidate draw must not collapse to a 0-D scalar.

        ``squeeze()`` on a 1-candidate posterior draw produced a 0-D tensor
        that ``sampled_values[best_idx]`` could not index (``IndexError``);
        ``reshape(-1)`` keeps it 1-D.
        """
        import math

        model, bounds = fitted_model_and_bounds
        batch = generate_thompson_samples(
            model,
            bounds,
            n_samples=1,
            config=ThompsonConfig(seed=3, num_candidates=1),
            minimize=minimize,
        )
        assert len(batch.samples) == 1
        assert math.isfinite(batch.samples[0].sampled_value)
        assert batch.parameters_tensor.shape == (1, 1)


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


@pytest.fixture(scope="module")
def conflicting_scaled_model() -> tuple[object, torch.Tensor]:
    """A 1-D, two-objective problem with a 1000x scale gap and a real trade-off.

    ``obj0(x) = x`` (scale ~[0, 1], minimized at x=0) conflicts with
    ``obj1(x) = 1000 * (1 - x)`` (scale ~[0, 1000], minimized at x=1). A plain
    weighted sum is dominated by ``obj1``'s magnitude, so even a tiny weight on
    ``obj1`` pulls every pick to x≈1; ParEGO normalizes both objectives onto
    [0, 1] first, so the weights actually steer the trade-off.
    """
    torch.manual_seed(2024)
    bounds = torch.tensor([[0.0], [1.0]], dtype=torch.double)
    train_x = torch.linspace(0, 1, 14, dtype=torch.double).unsqueeze(-1)
    obj0 = train_x
    obj1 = 1000.0 * (1.0 - train_x)
    train_y = torch.cat([obj0, obj1], dim=-1)
    model = create_and_fit_model(train_x, train_y, bounds)
    return model, bounds


class TestMultiObjectiveParEGO:
    """Augmented Tchebycheff scalarization on normalized objectives (M25)."""

    def test_weight_steers_trade_off_despite_scale_gap(
        self, conflicting_scaled_model: tuple
    ) -> None:
        """Heavy weight on the small-scale objective must select near its optimum.

        With weights ``[0.99, 0.01]`` and minimization, ParEGO normalizes both
        objectives to [0, 1], so the pick lands near ``obj0``'s optimum (x≈0).
        The old linear scalarization ``0.99*x + 0.01*1000*(1-x)`` is minimized
        at x=1 (``obj1``'s optimum), so this assertion fails on the pre-fix code.
        """
        model, bounds = conflicting_scaled_model
        batch = generate_thompson_samples_multi_objective(
            model,
            bounds,
            n_samples=1,
            weights=[0.99, 0.01],
            minimize=True,
            config=ThompsonConfig(seed=3),
        )
        x = float(batch.parameters_tensor[0, 0].item())
        assert x < 0.3, (
            f"ParEGO selected x={x:.3f}; a 0.99 weight on the small-scale "
            "objective should pick near its optimum x=0, not be dragged to "
            "obj1's optimum at x=1 by raw scale."
        )

    def test_random_weights_do_not_collapse_to_large_scale_optimum(
        self, conflicting_scaled_model: tuple
    ) -> None:
        """Random scalarizations spread along the front, not all at obj1's optimum."""
        model, bounds = conflicting_scaled_model
        batch = generate_thompson_samples_multi_objective(
            model,
            bounds,
            n_samples=8,
            minimize=True,
            config=ThompsonConfig(seed=5),
        )
        xs = batch.parameters_tensor[:, 0]
        # At least one pick must fall in obj0's half — a scale-dominated linear
        # sum would push every random-weighted pick toward x≈1.
        assert float(xs.min().item()) < 0.5, (
            f"All ParEGO picks clustered at x≥0.5 ({xs.tolist()}); the "
            "large-scale objective is still dominating the scalarization."
        )

    def test_per_sample_stats_aggregate_all_objectives(
        self, conflicting_scaled_model: tuple
    ) -> None:
        """Reported posterior mean reflects every objective, not just objective 0.

        ``obj0`` lives in [0, 1] while ``obj1`` runs to ~1000, so the
        across-objective mean is in the hundreds. The old code reported
        objective 0 only (mean ≤ 1), so a mean ≫ 1 proves the aggregation.
        """
        model, bounds = conflicting_scaled_model
        batch = generate_thompson_samples_multi_objective(
            model,
            bounds,
            n_samples=2,
            minimize=True,
            config=ThompsonConfig(seed=9),
        )
        for sample in batch.samples:
            assert sample.posterior_mean > 10.0, (
                f"posterior_mean={sample.posterior_mean:.3f} looks like obj0 "
                "alone (∈ [0, 1]); the stats must aggregate obj1's ~1000 scale."
            )
            assert sample.posterior_std > 0.0

    def test_seeded_calls_reproducible(self, conflicting_scaled_model: tuple) -> None:
        model, bounds = conflicting_scaled_model
        first = generate_thompson_samples_multi_objective(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=42)
        )
        second = generate_thompson_samples_multi_objective(
            model, bounds, n_samples=3, config=ThompsonConfig(seed=42)
        )
        assert torch.equal(first.parameters_tensor, second.parameters_tensor)
