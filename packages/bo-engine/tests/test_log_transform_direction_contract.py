"""Contract tests: capability surface vetoes maximize + ``log_transform``.

``ObjectiveSpec.log_transform`` is contractually restricted to
``minimize=True`` objectives: the suggestion pipeline rejects the
maximize combination with a ``ValueError`` in both the single-objective
path (``bo_engine.suggestions_single_objective``) and the
multi-objective path (``bo_engine.suggestions_multi_objective``), and
the model factory's ``Log`` outcome stage assumes the maximization-form
negation handled by ``Negate`` (see :mod:`bo_engine.models`).

``BoTorchBackend.validate_capabilities`` must mirror that runtime guard:
a compatible verdict for a spec the pipeline is guaranteed to reject
would let a pinned ``backend="botorch"`` (or auto-routed) campaign pass
intake and then fail on its first suggestion batch. This mirrors the
SAASBO / multi-fidelity advertisement-contract precedent and the
equivalent BayBE-side guard
(``bo_engine_baybe/tests/test_backend_capabilities.py``).

References:
    - BoTorch ``Log`` outcome transform (requires positive raw targets,
      applied before ``Standardize``):
      https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log
"""

from __future__ import annotations

from bo_engine.backend_base import CapabilityStatus
from bo_engine.botorch_backend import BoTorchBackend
from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _spec(objectives: list[ObjectiveSpec]) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
        objectives=objectives,
    )


class TestLogTransformDirectionCapability:
    def test_maximize_with_log_transform_reports_unsupported(self) -> None:
        spec = _spec([ObjectiveSpec(name="rate", minimize=False, log_transform=True)])
        result = BoTorchBackend().validate_capabilities(spec)
        reports = [r for r in result.option_reports if r.key == "objectives[0].log_transform"]
        assert reports
        assert reports[0].status == CapabilityStatus.UNSUPPORTED
        assert "minimize=True" in reports[0].reason
        assert not result.is_compatible

    def test_minimize_with_log_transform_stays_supported(self) -> None:
        """The supported combination must not regress to a veto."""
        spec = _spec([ObjectiveSpec(name="rate", minimize=True, log_transform=True)])
        result = BoTorchBackend().validate_capabilities(spec)
        assert not any("log_transform" in r.key for r in result.option_reports)
        assert result.is_compatible

    def test_multi_objective_flags_only_the_maximize_objective(self) -> None:
        spec = _spec(
            [
                ObjectiveSpec(name="impurity", minimize=True, log_transform=True),
                ObjectiveSpec(name="rate", minimize=False, log_transform=True),
            ]
        )
        result = BoTorchBackend().validate_capabilities(spec)
        keys = [r.key for r in result.option_reports if "log_transform" in r.key]
        assert keys == ["objectives[1].log_transform"]
        assert not result.is_compatible
