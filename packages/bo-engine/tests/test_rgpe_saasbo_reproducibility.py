"""Seed plumbing and RNG isolation for the RGPE / SAASBO / what-if entry points.

Mirrors ``TestMultiFidelityReproducibility`` (test_multifidelity.py) and the
``test_rng_isolation.py`` global-state checks: a configured master seed is
routed through ``derive_seed`` and installed inside a
``GLOBAL_RNG_LOCK`` + ``fork_rng`` block, so

* two calls with the same seed reproduce identical candidates,
* unseeded calls draw fresh entropy (no replay), and
* the process-global torch RNG stream is untouched by a call — a
  concurrent RGPE/SAASBO invocation must not silently perturb another
  seeded campaign's stream.

References:
    - PyTorch reproducibility notes:
      https://pytorch.org/docs/stable/notes/randomness.html
    - Feurer et al., "Scalable Meta-Learning for Bayesian Optimization"
      (RGPE) — the ranking-loss weights are Monte-Carlo estimates, hence
      seed-dependent by construction.
"""

from __future__ import annotations

import torch

from bo_engine.saasbo import SAASBOConfig, generate_saasbo_suggestions
from bo_engine.transfer_learning import (
    PriorTaskData,
    RGPEConfig,
    generate_rgpe_suggestions,
)
from bo_engine.types import AcquisitionOptimizationConfig
from bo_engine.whatif import find_most_informative_point


def _rgpe_problem() -> tuple[torch.Tensor, torch.Tensor, list[PriorTaskData], torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    target_x = torch.rand(8, 2, dtype=torch.double, generator=generator)
    target_y = ((target_x - 0.3) ** 2).sum(dim=-1, keepdim=True)
    prior_x = torch.rand(10, 2, dtype=torch.double, generator=generator)
    prior_y = ((prior_x - 0.35) ** 2).sum(dim=-1, keepdim=True)
    prior_tasks = [PriorTaskData(prior_x, prior_y, name="prior")]
    bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    return target_x, target_y, prior_tasks, bounds


_FAST_ACQ = AcquisitionOptimizationConfig(num_restarts=2, raw_samples=32)


class TestRGPEReproducibility:
    def test_same_seed_reproduces_candidates(self) -> None:
        target_x, target_y, prior_tasks, bounds = _rgpe_problem()
        config = RGPEConfig(num_samples=32, random_seed=2024)

        cand_a, _, meta_a = generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            config=config,
            acquisition_optimization=_FAST_ACQ,
        )
        cand_b, _, meta_b = generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            config=config,
            acquisition_optimization=_FAST_ACQ,
        )

        assert torch.allclose(cand_a, cand_b), (
            "Identical master seeds must reproduce identical RGPE candidates."
        )
        assert meta_a["random_seed"] == meta_b["random_seed"]
        assert meta_a["random_seed"] is not None

    def test_unseeded_calls_draw_fresh_entropy(self) -> None:
        """fork_rng restores global state, so unseeded calls need fresh entropy.

        The observable contract is the per-call seed: without the fallback
        draw, fork_rng's restore would replay the identical stream on every
        unseeded call. The *candidates* may still legitimately coincide —
        different seeds can converge to the same acquisition argmax on a
        smooth surface — so only the seeds are asserted here.
        """
        target_x, target_y, prior_tasks, bounds = _rgpe_problem()
        config = RGPEConfig(num_samples=32)

        _, _, meta_a = generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            config=config,
            acquisition_optimization=_FAST_ACQ,
        )
        _, _, meta_b = generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            config=config,
            acquisition_optimization=_FAST_ACQ,
        )

        assert meta_a["random_seed"] != meta_b["random_seed"]

    def test_global_torch_stream_untouched(self) -> None:
        """A call must not consume from (or reseed) the process-global stream."""
        target_x, target_y, prior_tasks, bounds = _rgpe_problem()

        torch.manual_seed(123)
        expected = torch.rand(4)

        torch.manual_seed(123)
        generate_rgpe_suggestions(
            target_x,
            target_y,
            prior_tasks,
            bounds,
            config=RGPEConfig(num_samples=32, random_seed=1),
            acquisition_optimization=_FAST_ACQ,
        )
        observed = torch.rand(4)

        assert torch.equal(expected, observed), (
            "generate_rgpe_suggestions must snapshot/restore the global torch RNG "
            "so concurrent seeded campaigns keep their streams."
        )


class TestSAASBOReproducibility:
    """SAASBO (NUTS) seed plumbing — kept fast via a tiny MCMC budget."""

    @staticmethod
    def _problem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(5)
        train_x = torch.rand(10, 3, dtype=torch.double, generator=generator)
        train_y = ((train_x - 0.5) ** 2).sum(dim=-1, keepdim=True)
        bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
        return train_x, train_y, bounds

    @staticmethod
    def _tiny_config(random_seed: int) -> SAASBOConfig:
        """Minimal MCMC budget so the NUTS fit stays in the fast suite."""
        return SAASBOConfig(warmup_steps=32, num_samples=16, thinning=8, random_seed=random_seed)

    def test_same_seed_reproduces_candidates(self) -> None:
        train_x, train_y, bounds = self._problem()
        config = self._tiny_config(random_seed=2024)

        cand_a, _, meta_a = generate_saasbo_suggestions(
            train_x, train_y, bounds, config=config, acquisition_optimization=_FAST_ACQ
        )
        cand_b, _, meta_b = generate_saasbo_suggestions(
            train_x, train_y, bounds, config=config, acquisition_optimization=_FAST_ACQ
        )

        assert torch.allclose(cand_a, cand_b)
        assert meta_a["random_seed"] == meta_b["random_seed"]

    def test_global_torch_stream_untouched(self) -> None:
        train_x, train_y, bounds = self._problem()

        torch.manual_seed(321)
        expected = torch.rand(4)

        torch.manual_seed(321)
        generate_saasbo_suggestions(
            train_x,
            train_y,
            bounds,
            config=self._tiny_config(random_seed=1),
            acquisition_optimization=_FAST_ACQ,
        )
        observed = torch.rand(4)

        assert torch.equal(expected, observed)


class TestFindMostInformativePointSeeding:
    """The Sobol candidate cloud must be reproducible per seed."""

    @staticmethod
    def _model():
        from bo_engine.models import create_and_fit_single_task_model

        generator = torch.Generator().manual_seed(3)
        train_x = torch.rand(6, 2, dtype=torch.double, generator=generator)
        train_y = train_x.sum(dim=-1)
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
        return create_and_fit_single_task_model(train_x, train_y, bounds), bounds

    def test_same_seed_same_answer(self) -> None:
        model, bounds = self._model()
        params_a, value_a = find_most_informative_point(model, bounds, seed=42)
        params_b, value_b = find_most_informative_point(model, bounds, seed=42)
        assert params_a == params_b
        assert value_a == value_b

    def test_different_seeds_generally_differ(self) -> None:
        model, bounds = self._model()
        params_a, _ = find_most_informative_point(model, bounds, seed=1)
        params_b, _ = find_most_informative_point(model, bounds, seed=2)
        assert params_a != params_b
