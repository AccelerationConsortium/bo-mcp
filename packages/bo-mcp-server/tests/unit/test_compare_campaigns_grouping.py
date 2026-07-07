"""Sample-efficiency comparability groups in ``compare_campaigns``.

The single-objective metric (relative improvement per result,
dimensionless) and the multi-objective metric (hypervolume per result, in
objective units) are incommensurable: a raw ``max()`` across the mix let
multi-objective campaigns "win" on magnitude alone (an HV rate of 5400
trivially beats a 0.05 improvement ratio) and fed the transfer
recommendation with a meaningless ranking. Winners are therefore computed
per comparability group, and the flat ``best_sample_efficiency`` is only
populated when every compared campaign shares one group.
"""

from __future__ import annotations

from typing import Any

from bo_mcp_server.operations.compare_campaigns import _compare_metrics


def _single(sample_efficiency: float, best_value: float) -> dict[str, Any]:
    return {
        "n_results": 10,
        "is_multi_objective": False,
        "objective_signature": ["yield:maximize"],
        "sample_efficiency_basis": "relative_improvement_per_result",
        "sample_efficiency": sample_efficiency,
        "best_value": best_value,
        "hypervolume": None,
    }


def _multi(
    sample_efficiency: float,
    hypervolume: float,
    signature: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "n_results": 10,
        "is_multi_objective": True,
        "objective_signature": signature or ["cost:minimize", "yield:maximize"],
        "sample_efficiency_basis": "hypervolume_per_result",
        "sample_efficiency": sample_efficiency,
        "best_value": None,
        "hypervolume": hypervolume,
    }


class TestMixedComparisonDoesNotRankAcrossGroups:
    def test_mixed_single_multi_yields_no_flat_winner(self) -> None:
        metrics = [_single(0.05, 0.9), _multi(5400.0, 54000.0)]
        comparison = _compare_metrics(metrics, ["Single", "Multi"])

        assert comparison["best_sample_efficiency"] is None, (
            "A raw-magnitude HV rate must not beat a dimensionless improvement "
            "ratio in one flat ranking."
        )
        by_group = comparison["best_sample_efficiency_by_group"]
        assert by_group["single_objective"] == "Single"
        assert any(key.startswith("multi_objective[") for key in by_group)
        assert "not" in comparison["recommendation"].lower()

    def test_single_only_comparison_keeps_flat_winner(self) -> None:
        metrics = [_single(0.05, 0.9), _single(0.10, 0.7)]
        comparison = _compare_metrics(metrics, ["A", "B"])

        assert comparison["best_sample_efficiency"] == "B"
        assert comparison["best_sample_efficiency_by_group"] == {"single_objective": "B"}

    def test_multi_same_signature_ranked_within_group(self) -> None:
        metrics = [_multi(10.0, 100.0), _multi(20.0, 200.0)]
        comparison = _compare_metrics(metrics, ["A", "B"])

        assert comparison["best_sample_efficiency"] == "B"
        assert comparison["best_multi_objective"] == "B"

    def test_multi_different_signatures_not_ranked_on_raw_hypervolume(self) -> None:
        metrics = [
            _multi(10.0, 100.0, signature=["cost:minimize", "yield:maximize"]),
            _multi(99999.0, 999999.0, signature=["impurity:minimize", "rate:maximize"]),
        ]
        comparison = _compare_metrics(metrics, ["A", "B"])

        assert comparison["best_sample_efficiency"] is None
        assert comparison["best_multi_objective"] is None, (
            "Raw hypervolumes of different objective structures carry different "
            "units and must not be ranked against each other."
        )

    def test_envelope_labels_the_metric_basis(self) -> None:
        metrics = [_single(0.05, 0.9), _multi(5400.0, 54000.0)]
        comparison = _compare_metrics(metrics, ["Single", "Multi"])

        note = comparison["sample_efficiency_note"]
        assert "relative_improvement_per_result" in note
        assert "hypervolume_per_result" in note
