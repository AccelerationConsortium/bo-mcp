"""Tests that pin the seed-discipline contract of shared training-data fixtures.

The ``branin_training_data`` and ``hartmann6_training_data`` fixtures in
``conftest.py`` are expected to:

1. Depend on the ``seed`` fixture rather than hard-coding ``torch.manual_seed(42)``
   inside the fixture body, so that callers that parameterize the seed (or that
   override the ``seed`` fixture) get distinct training sets.
2. Route seeding through ``set_all_seeds`` so torch / numpy / random are all
   advanced together — partial seeding leaves callers exposed to flakiness from
   whichever generator was missed.

These properties are easy to regress (a future fixture refactor can quietly
reintroduce ``torch.manual_seed(42)`` and the call site will still "work"). The
tests below therefore drive the *actual fixtures* (not a re-implementation of
the same recipe) under multiple seed parametrizations and assert that the
fixture's outputs differ across seeds. Outputs are collected into a
class-scoped fixture-output collector and compared pairwise once all
parametrize values have been observed; a regression that ignored ``seed``
would surface as identical tensors across the parametrize values and fail
the cross-seed assertion.

References:
    - pytest fixture parametrization:
      https://docs.pytest.org/en/stable/how-to/fixtures.html#fixture-parametrize
    - pytest fixture overrides at class/test scope:
      https://docs.pytest.org/en/stable/how-to/fixtures.html#override-a-fixture-on-a-test-or-class-level
    - PyTorch reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
"""

from __future__ import annotations

import pytest
import torch

# Seeds used to drive the parametrize-based fixture-output collection. Three
# distinct values are enough to catch any "fixture ignores seed" regression
# without inflating runtime.
SEED_SAMPLES = (1, 2, 3)


class TestBraninFixtureRespectsSeed:
    """Branin training-data fixture must respond to the seed fixture.

    The parametrized test invokes the *actual* ``branin_training_data``
    fixture with three different seed values, collects each fixture output,
    and on the final parametrize value asserts that the three outputs are
    pairwise distinct. If a future refactor reintroduced hard-coded
    ``torch.manual_seed(42)`` inside the fixture body, the parametrize values
    would yield identical tensors and the pairwise assertion would fail.
    """

    # Class-level shared collector so parametrize values can compare against
    # one another. pytest creates a fresh class instance per test method but
    # the class object itself is shared, so the dict accumulates outputs
    # across the parametrize axis.
    _outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def setup_class(cls) -> None:
        """Reset the per-class collector before the parametrize sweep."""
        cls._outputs = {}

    @pytest.mark.parametrize("seed", list(SEED_SAMPLES))
    def test_fixture_output_changes_with_seed_fixture(
        self,
        seed: int,
        branin_training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """The Branin fixture must yield distinct outputs across seeds.

        Each parametrize value drives the real ``branin_training_data``
        fixture (not a re-implementation) through the ``seed`` fixture
        override and stashes the result in a class-scoped collector. Once
        every parametrize value has been observed the test performs a
        pairwise distinctness assertion. A fixture that ignored its
        ``seed`` argument would store the same tensor under every key and
        fail this assertion on the last parametrize invocation.
        """
        train_x, train_y = branin_training_data
        assert train_x.shape == (10, 2)
        assert torch.isfinite(train_x).all()
        assert torch.isfinite(train_y).all()

        # Record this seed's actual fixture output.
        self.__class__._outputs[seed] = (train_x.clone(), train_y.clone())

        # When all parametrize values are present, run the pairwise
        # distinctness check that proves the fixture honours ``seed``.
        if set(self.__class__._outputs) >= set(SEED_SAMPLES):
            _assert_pairwise_distinct(self.__class__._outputs, fixture_name="branin")

    def test_same_seed_is_reproducible(
        self, branin_training_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Two invocations of the fixture under the default ``seed`` agree exactly.

        Drives the same ``branin_training_data`` fixture twice (the second
        time via a manual replay of its body under the same ``seed=42``
        default) and asserts the tensors match. This protects against
        a regression that partially seeds the RNG and leaks state across
        otherwise-identical seeded invocations.
        """
        from bo_engine.benchmarks import branin, branin_bounds
        from tests.conftest import set_all_seeds

        first_x, first_y = branin_training_data

        # Replay the fixture body deterministically under the default seed.
        set_all_seeds(42)
        bounds = branin_bounds()
        replay_unit = torch.rand(10, 2, dtype=first_x.dtype)
        replay_x = replay_unit * (bounds[1] - bounds[0]) + bounds[0]
        replay_y = branin(replay_x).unsqueeze(-1)

        assert torch.allclose(first_x, replay_x)
        assert torch.allclose(first_y, replay_y)


class TestHartmann6FixtureRespectsSeed:
    """Hartmann6 training-data fixture must respond to the seed fixture."""

    _outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    @classmethod
    def setup_class(cls) -> None:
        """Reset the per-class collector before the parametrize sweep."""
        cls._outputs = {}

    @pytest.mark.parametrize("seed", list(SEED_SAMPLES))
    def test_fixture_output_changes_with_seed_fixture(
        self,
        seed: int,
        hartmann6_training_data: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """The Hartmann6 fixture must yield distinct outputs across seeds.

        Same contract as the Branin variant — see that test's docstring for
        the full rationale.
        """
        train_x, train_y = hartmann6_training_data
        assert train_x.shape == (15, 6)
        assert torch.isfinite(train_x).all()
        assert torch.isfinite(train_y).all()

        self.__class__._outputs[seed] = (train_x.clone(), train_y.clone())

        if set(self.__class__._outputs) >= set(SEED_SAMPLES):
            _assert_pairwise_distinct(self.__class__._outputs, fixture_name="hartmann6")

    def test_same_seed_is_reproducible(
        self, hartmann6_training_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Two invocations of the fixture under the default ``seed`` agree exactly."""
        from bo_engine.benchmarks import hartmann6
        from tests.conftest import set_all_seeds

        first_x, first_y = hartmann6_training_data

        set_all_seeds(42)
        replay_x = torch.rand(15, 6, dtype=first_x.dtype)
        replay_y = hartmann6(replay_x).unsqueeze(-1)

        assert torch.allclose(first_x, replay_x)
        assert torch.allclose(first_y, replay_y)


class TestFixturesUseSetAllSeeds:
    """The fixtures must route seeding through ``set_all_seeds``.

    ``set_all_seeds`` advances torch + numpy + Python ``random`` together. If a
    fixture used ``torch.manual_seed`` alone, then any downstream consumer that
    drew from numpy (e.g. ``np.random.default_rng(None)`` or a ``baybe``
    backend that consults numpy) would diverge between runs even when the test
    pinned ``seed`` for reproducibility.
    """

    def test_set_all_seeds_advances_numpy_and_random(self) -> None:
        """``set_all_seeds`` advances numpy + python random in addition to torch."""
        import random

        import numpy as np

        from tests.conftest import set_all_seeds

        set_all_seeds(123)
        torch_a = torch.rand(3)
        np_a = np.random.rand(3)
        py_a = [random.random() for _ in range(3)]  # noqa: S311 - test fixture

        set_all_seeds(123)
        torch_b = torch.rand(3)
        np_b = np.random.rand(3)
        py_b = [random.random() for _ in range(3)]  # noqa: S311 - test fixture

        assert torch.allclose(torch_a, torch_b)
        assert np.allclose(np_a, np_b)
        assert py_a == py_b

    def test_branin_fixture_pins_numpy_and_python_random_too(
        self, branin_training_data: tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """A seeded fixture call leaves numpy + python ``random`` reproducible too.

        After ``branin_training_data`` has been built under its default
        ``seed=42``, both ``np.random.rand`` and ``random.random`` must
        return the same value as they would under a manual ``set_all_seeds(42)``
        replay. A fixture that only called ``torch.manual_seed`` would leave
        numpy and python's ``random`` un-seeded, so the post-fixture draws
        below would diverge from the manual replay.
        """
        import random

        import numpy as np

        from tests.conftest import set_all_seeds

        # After the fixture (default seed=42) ran, draw from numpy + random.
        fixture_numpy_draw = np.random.rand(3)
        fixture_python_draw = [random.random() for _ in range(3)]  # noqa: S311 - test fixture

        # Replay the same seed manually and run an equivalent draw sequence
        # so the comparison only diverges if the fixture failed to seed numpy
        # or python's random module.
        set_all_seeds(42)
        # The fixture itself consumed: 10x2 torch.rand and the branin eval
        # internally (which does not touch numpy or python random). So a
        # matching post-state replay just re-seeds and immediately draws.
        replay_numpy_draw = np.random.rand(3)
        replay_python_draw = [random.random() for _ in range(3)]  # noqa: S311 - test fixture

        assert np.allclose(fixture_numpy_draw, replay_numpy_draw)
        assert fixture_python_draw == replay_python_draw
        # Sanity check that the fixture actually produced data.
        train_x, _ = branin_training_data
        assert train_x.shape == (10, 2)


def _assert_pairwise_distinct(
    outputs: dict[int, tuple[torch.Tensor, torch.Tensor]],
    fixture_name: str,
) -> None:
    """Helper: every (seed_i, seed_j) output pair must differ for both x and y."""
    seeds = sorted(outputs)
    for i, seed_i in enumerate(seeds):
        x_i, y_i = outputs[seed_i]
        for seed_j in seeds[i + 1 :]:
            x_j, y_j = outputs[seed_j]
            assert not torch.allclose(x_i, x_j), (
                f"{fixture_name} fixture must produce different train_x for "
                f"seeds {seed_i} and {seed_j}; got identical tensors — has the "
                "fixture stopped depending on its `seed` argument?"
            )
            assert not torch.allclose(y_i, y_j), (
                f"{fixture_name} fixture must produce different train_y for "
                f"seeds {seed_i} and {seed_j}; got identical tensors"
            )
