"""Pending-point fantasy helper removal contract (8.38).

The ``get_pending_as_fantasy_model_input`` helper in
``bo_engine.pending_points`` was exported but never called — the active
suggestion pipeline routes pending points through BoTorch's ``X_pending``
conditioning instead. The audit asked us to pick one path; we kept
``X_pending`` and removed the fantasy helper to avoid the documentation
hazard of pointing future maintainers at dead code.

This test pins the contract so the helper doesn't accidentally return.

References:
    - BoTorch ``X_pending`` conditioning:
      https://botorch.org/docs/batched_bayesian_optimization/
"""

from __future__ import annotations

from bo_engine import pending_points


def test_fantasy_helper_removed() -> None:
    assert not hasattr(pending_points, "get_pending_as_fantasy_model_input"), (
        "The Kriging-Believer fantasy helper was removed in favor of "
        "BoTorch's X_pending conditioning. Re-adding it requires wiring "
        "fantasy generation into the suggestion pipeline or documenting "
        "the dual-path strategy."
    )
