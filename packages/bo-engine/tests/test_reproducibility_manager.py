"""``ReproducibilityManager`` must not mutate process-global state implicitly.

Two invariants are pinned here:

1. **Construction is side-effect free.** Building a manager derives seeds
   purely; it must not flip ``torch``'s deterministic-algorithm mode or set
   ``CUBLAS_WORKSPACE_CONFIG``. Those are process-wide settings — coupling
   them to object construction lets one campaign silently change numerical
   behavior (and possibly raise on nondeterministic ops) for unrelated
   concurrent campaigns in the same server process. Determinism is opt-in
   through the explicit :meth:`ReproducibilityManager.apply_global_settings`.
2. **Verification is a pure query.** ``verify_iteration`` re-derives the
   iteration seeds without appending to the audit ``seed_log``, so checking
   reproducibility cannot leave phantom entries behind.

Reference: PyTorch reproducibility notes
(https://pytorch.org/docs/stable/notes/randomness.html) document
``use_deterministic_algorithms`` / ``CUBLAS_WORKSPACE_CONFIG`` as
process-global switches, which is why they must be applied deliberately.
"""

from __future__ import annotations

import os

import torch

from bo_engine.reproducibility import (
    IterationSeeds,
    ReproducibilityConfig,
    ReproducibilityManager,
    derive_seed,
)


def _expected_iteration_seeds(master_seed: int, iteration: int) -> IterationSeeds:
    """Replicate the documented per-iteration derivation contract."""
    return IterationSeeds(
        iteration=iteration,
        initial_design_seed=derive_seed(master_seed, f"initial_{iteration}"),
        model_fit_seed=derive_seed(master_seed, f"model_{iteration}"),
        acquisition_opt_seed=derive_seed(master_seed, f"acq_{iteration}"),
        batch_seed=derive_seed(master_seed, f"batch_{iteration}"),
    )


class TestConstructionHasNoGlobalSideEffects:
    """Constructing a manager must leave global determinism state untouched."""

    def test_construction_does_not_change_deterministic_mode(self) -> None:
        before = torch.are_deterministic_algorithms_enabled()

        # ``deterministic_algorithms=True`` is the config default — the very
        # setting that used to be applied eagerly from ``__init__``.
        ReproducibilityManager(
            ReproducibilityConfig(master_seed=12345, deterministic_algorithms=True)
        )

        assert torch.are_deterministic_algorithms_enabled() == before


class TestApplyGlobalSettings:
    """The explicit opt-in actually applies the process-wide settings."""

    def test_apply_enables_deterministic_mode(self) -> None:
        was_enabled = torch.are_deterministic_algorithms_enabled()
        warn_only_query = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
        was_warn_only = warn_only_query() if warn_only_query is not None else False
        prev_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

        try:
            manager = ReproducibilityManager(
                ReproducibilityConfig(
                    master_seed=999,
                    deterministic_algorithms=True,
                    strict_mode=False,
                )
            )
            manager.apply_global_settings()

            assert torch.are_deterministic_algorithms_enabled() is True
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        finally:
            # Restore the prior process-global state so this test does not
            # leak determinism settings into the rest of the suite.
            torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
            if prev_cublas is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = prev_cublas


class TestVerifyIterationDoesNotPolluteLog:
    """``verify_iteration`` is a query: it must not touch the seed log."""

    def test_matching_seeds_return_true_without_logging(self) -> None:
        manager = ReproducibilityManager(ReproducibilityConfig(master_seed=7, log_seeds=True))
        expected = _expected_iteration_seeds(7, 3)

        assert len(manager.get_reproducibility_report().seed_log) == 0
        assert manager.verify_iteration(3, expected) is True
        # No phantom entry recorded by the verification query.
        assert len(manager.get_reproducibility_report().seed_log) == 0

    def test_mismatched_seeds_return_false_without_logging(self) -> None:
        manager = ReproducibilityManager(ReproducibilityConfig(master_seed=7, log_seeds=True))
        expected = _expected_iteration_seeds(7, 3)
        wrong = IterationSeeds(
            iteration=3,
            initial_design_seed=expected.initial_design_seed + 1,
            model_fit_seed=expected.model_fit_seed,
            acquisition_opt_seed=expected.acquisition_opt_seed,
            batch_seed=expected.batch_seed,
        )

        assert manager.verify_iteration(3, wrong) is False
        assert len(manager.get_reproducibility_report().seed_log) == 0

    def test_get_iteration_seeds_still_logs(self) -> None:
        """The logging mutator path is unchanged: it still records seeds."""
        manager = ReproducibilityManager(ReproducibilityConfig(master_seed=7, log_seeds=True))

        manager.get_iteration_seeds(3)

        assert len(manager.get_reproducibility_report().seed_log) == 1
