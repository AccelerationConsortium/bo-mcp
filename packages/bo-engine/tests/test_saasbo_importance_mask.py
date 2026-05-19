"""Tests for the SAASBO active-dimension mask (TODO 1.6).

SAASBO's sparsity-inducing prior drives inactive dimensions to large
lengthscales. The normalized inverse-lengthscale importance metric then
collapses those large lengthscales to small-but-nonzero scores, so a
consumer cannot tell "slightly important" from "provably inactive". The
``compute_saasbo_importance_report`` helper restores that distinction by
returning both the raw lengthscale and a boolean ``active`` flag derived
from :data:`bo_engine.constants.SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD`.

Reference: Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization
with Sparse Axis-Aligned Subspaces", UAI 2021
(https://arxiv.org/abs/2103.00349), §3.3 & Appendix B -- truly inactive
SAAS dimensions converge to lengthscales >> 1, so a threshold ~10 is a
safe binary cut-off.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
import torch
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP

from bo_engine.constants import SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD
from bo_engine.saasbo import (
    SAASBOImportance,
    compute_saasbo_importance,
    compute_saasbo_importance_report,
)


def _patched_lengthscales(values: list[float]):
    """Patch the lengthscale extractor with deterministic inputs."""
    return patch(
        "bo_engine.saasbo.get_saasbo_lengthscales",
        return_value=torch.tensor(values, dtype=torch.float64),
    )


def _fake_model() -> SaasFullyBayesianSingleTaskGP:
    """Return a stand-in for the patched-extractor branch.

    ``compute_saasbo_importance_report`` calls
    ``get_saasbo_lengthscales(model)`` — and *only that* — so any object whose
    structural type the type checker accepts as ``SaasFullyBayesianSingleTaskGP``
    is sufficient. We use ``cast`` rather than instantiating a real (and
    expensive-to-fit) SAASBO model.
    """
    return cast(SaasFullyBayesianSingleTaskGP, object())


class TestActiveMaskFromLengthscales:
    """The boolean ``active`` flag is purely a function of the raw lengthscales."""

    def test_short_lengthscales_are_active(self) -> None:
        """Lengthscales well below the threshold are flagged active."""
        with _patched_lengthscales([0.1, 0.5, 1.0]):
            report = compute_saasbo_importance_report(
                model=_fake_model(),  # patched extractor ignores it
                parameter_names=["a", "b", "c"],
            )
        assert [entry.active for entry in report] == [True, True, True]

    def test_long_lengthscales_are_inactive(self) -> None:
        """Lengthscales above the threshold are flagged inactive."""
        large = SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD * 10
        with _patched_lengthscales([large, large, large]):
            report = compute_saasbo_importance_report(
                model=_fake_model(),
                parameter_names=["a", "b", "c"],
            )
        assert [entry.active for entry in report] == [False, False, False]

    def test_threshold_is_inclusive_at_boundary(self) -> None:
        """Exact threshold counts as active (<=), the next epsilon is inactive."""
        eps = 1e-9
        with _patched_lengthscales(
            [
                SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD,
                SAASBO_INACTIVE_LENGTHSCALE_THRESHOLD + eps,
            ]
        ):
            report = compute_saasbo_importance_report(
                model=_fake_model(),
                parameter_names=["edge", "above"],
            )
        assert report[0].active is True
        assert report[1].active is False

    def test_mixed_sparse_pattern_matches_published_behavior(self) -> None:
        """Two-active-of-six pattern matches the SAASBO synthetic-example shape.

        Eriksson & Jankowiak 2021 §4 reports that on the 6D Branin embedding
        the two informative dimensions have lengthscales O(1) while the four
        embedded dummies converge to O(1e2)+. The mask must reflect that
        partition.
        """
        lengthscales = [0.7, 1.4, 200.0, 300.0, 150.0, 500.0]
        with _patched_lengthscales(lengthscales):
            report = compute_saasbo_importance_report(
                model=_fake_model(),
                parameter_names=[f"x{i}" for i in range(6)],
            )
        active = [entry.name for entry in report if entry.active]
        inactive = [entry.name for entry in report if not entry.active]
        assert active == ["x0", "x1"]
        assert inactive == ["x2", "x3", "x4", "x5"]


class TestImportanceReportShape:
    """The report carries the raw lengthscale, importance and active mask."""

    def test_entries_align_with_parameter_names(self) -> None:
        """Returned list is ordered by input-parameter index."""
        with _patched_lengthscales([1.0, 100.0]):
            report = compute_saasbo_importance_report(
                model=_fake_model(),
                parameter_names=["alpha", "beta"],
            )
        assert [entry.name for entry in report] == ["alpha", "beta"]

    def test_default_parameter_names(self) -> None:
        """Without explicit names the entries use ``param_{i}``."""
        with _patched_lengthscales([0.5, 1.0, 1000.0]):
            report = compute_saasbo_importance_report(model=_fake_model())
        assert [entry.name for entry in report] == ["param_0", "param_1", "param_2"]

    def test_importance_still_sums_to_one(self) -> None:
        """Sum of the normalized scores stays at 1.0 -- API contract preserved."""
        with _patched_lengthscales([0.1, 0.5, 1.0, 200.0]):
            report = compute_saasbo_importance_report(model=_fake_model())
        total = sum(entry.importance for entry in report)
        assert abs(total - 1.0) < 1e-6

    def test_lengthscale_value_is_raw_median(self) -> None:
        """``lengthscale`` echoes the per-dimension input value exactly."""
        values = [0.13, 27.0, 9.5]
        with _patched_lengthscales(values):
            report = compute_saasbo_importance_report(model=_fake_model())
        for entry, expected in zip(report, values, strict=True):
            assert abs(entry.lengthscale - expected) < 1e-9
            assert isinstance(entry, SAASBOImportance)


class TestBackwardCompatibility:
    """Legacy ``compute_saasbo_importance`` shape and totals are preserved."""

    def test_legacy_function_returns_dict_with_same_totals(self) -> None:
        """The plain dict API stays at sum==1 and identical keys."""
        with _patched_lengthscales([0.5, 5.0, 50.0]):
            result = compute_saasbo_importance(
                model=_fake_model(),
                parameter_names=["alpha", "beta", "gamma"],
            )
        assert set(result.keys()) == {"alpha", "beta", "gamma"}
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_legacy_function_ranks_match_report(self) -> None:
        """``compute_saasbo_importance`` is the projection of the report."""
        with _patched_lengthscales([0.5, 5.0, 50.0]):
            dict_form = compute_saasbo_importance(model=_fake_model())
            report = compute_saasbo_importance_report(model=_fake_model())
        for entry in report:
            assert abs(dict_form[entry.name] - entry.importance) < 1e-9


class TestShapeRobustness:
    """Single-parameter and scalar extractor outputs must not crash.

    ``get_saasbo_lengthscales`` returns ``lengthscales.median(...).values`` —
    which for a 1-D search space collapsed to a 0-D tensor under the old
    ``.squeeze()`` and then crashed ``shape[-1]`` indexing. The extractor now
    flattens to a strict 1-D tensor, so the report API stays well-defined
    irrespective of model dimensionality.
    """

    def test_single_dimension_lengthscale_does_not_crash(self) -> None:
        """A scalar extractor output is treated as a 1-D vector of length 1."""
        with patch(
            "bo_engine.saasbo.get_saasbo_lengthscales",
            return_value=torch.tensor(0.7, dtype=torch.float64),
        ):
            report = compute_saasbo_importance_report(model=_fake_model())
        assert len(report) == 1
        assert report[0].name == "param_0"
        assert report[0].active is True
        # Single-dimension importance is trivially 1.0 after normalization.
        assert abs(report[0].importance - 1.0) < 1e-6

    def test_one_parameter_name_with_scalar_extractor(self) -> None:
        """Passing the right ``parameter_names`` length still works."""
        with patch(
            "bo_engine.saasbo.get_saasbo_lengthscales",
            return_value=torch.tensor(2.0, dtype=torch.float64),
        ):
            report = compute_saasbo_importance_report(
                model=_fake_model(), parameter_names=["alpha"]
            )
        assert [entry.name for entry in report] == ["alpha"]


class TestParameterNameValidation:
    """Mismatched ``parameter_names`` length must be rejected explicitly."""

    def test_mismatched_names_raise(self) -> None:
        """Three lengthscales but two names -> ValueError, not silent drop."""
        with (
            _patched_lengthscales([0.5, 1.0, 50.0]),
            pytest.raises(ValueError, match="parameter_names length"),
        ):
            compute_saasbo_importance_report(model=_fake_model(), parameter_names=["a", "b"])

    def test_legacy_function_validates_names_too(self) -> None:
        """The dict-shape API delegates and therefore inherits the check."""
        with (
            _patched_lengthscales([0.5, 1.0]),
            pytest.raises(ValueError, match="parameter_names length"),
        ):
            compute_saasbo_importance(model=_fake_model(), parameter_names=["only_one"])
