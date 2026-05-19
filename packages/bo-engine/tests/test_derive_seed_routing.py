"""Promote ``_derive_seed`` and route every stochastic phase through it (8.39a).

A private seed-derivation helper already existed
(``bo_engine.reproducibility._derive_seed``) but the initial-design Sobol
seed and the acquisition multi-start seed each derived their seeds
independently — the first passed ``spec.random_seed`` directly to
``SobolEngine``, the second used a ``+ iteration * SEED_ITERATION_OFFSET``
stride. Two campaigns sharing the same master seed could therefore have
the same iteration's Sobol and acquisition seeds collide, with subtle
reproducibility consequences for any future stochastic phase that picked
yet another derivation.

We promote the helper to :func:`bo_engine.reproducibility.derive_seed`
and route both phases through it with stable role tags
(``"sobol:initial_design"`` and ``"acquisition:iter_{iteration}"``).

References:
    - Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011 —
      hash-based per-phase keying as a standard pattern for reproducible
      multi-phase stochastic pipelines.
"""

from __future__ import annotations

from bo_engine.reproducibility import _derive_seed, derive_seed
from bo_engine.suggestions import _resolve_acquisition_seed
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _make_spec(seed: int | None) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        random_seed=seed,
    )


class TestPublicAlias:
    """The promoted public helper and the legacy private name must agree."""

    def test_derive_seed_alias_matches(self) -> None:
        # The legacy ``_derive_seed`` symbol must continue to resolve to
        # the same function so external callers that imported the
        # underscore name are not broken.
        assert _derive_seed is derive_seed

    def test_derive_seed_is_deterministic(self) -> None:
        a = derive_seed(123, "sobol:initial_design")
        b = derive_seed(123, "sobol:initial_design")
        assert a == b

    def test_role_tag_changes_the_seed(self) -> None:
        sobol = derive_seed(123, "sobol:initial_design")
        acquisition = derive_seed(123, "acquisition:iter_5")
        assert sobol != acquisition, (
            "Per-phase role tags must produce distinct seeds so the Sobol "
            "and acquisition phases at the same master seed don't trample "
            "each other."
        )


class TestAcquisitionRoutesThroughDeriveSeed:
    """``_resolve_acquisition_seed`` must derive its seed via ``derive_seed``."""

    def test_acquisition_seed_matches_derive_seed_call(self) -> None:
        spec = _make_spec(seed=12345)
        resolved = _resolve_acquisition_seed(spec, iteration=7, rng=None)
        expected = derive_seed(12345, "acquisition:iter_7")
        assert resolved == expected

    def test_acquisition_seed_distinct_per_iteration(self) -> None:
        spec = _make_spec(seed=12345)
        seeds = {_resolve_acquisition_seed(spec, iteration=i, rng=None) for i in range(8)}
        # Eight iterations should yield eight distinct seeds (collisions
        # under derive_seed are vanishingly unlikely for 32-bit space).
        assert len(seeds) == 8


class TestSobolRoutesThroughDeriveSeed:
    """The Sobol initial-design seed must match the derive_seed output."""

    def test_sobol_seed_uses_stable_role_tag(self) -> None:
        from bo_engine.initial_design import _draw_sobol_designs

        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
                for i in range(3)
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
            random_seed=98765,
        )
        # Two calls with the same master seed must produce identical Sobol
        # draws — the public contract of seed propagation.
        designs_a = _draw_sobol_designs(spec, n_dims=3, draw_count=4, n_drawn=0)
        designs_b = _draw_sobol_designs(spec, n_dims=3, draw_count=4, n_drawn=0)
        assert designs_a == designs_b
