"""Cross-backend contract test suite.

Exercises the JSON-safety, capability-report, initial-design,
suggestion-batch, and state-envelope assertions in
:mod:`bo_engine.backend_testing` against:

* The default :class:`BoTorchBackend` implementation.
* A minimal :class:`_FakeBackend` that subclasses :class:`BaseBackend`
  without overriding the optional defaults. Pinning the defaults this
  way is the regression net the protocol's per-method contract relies
  on — accidental drift in any default implementation will fail here.
* A pure-:class:`BOBackend` plugin (``_ProtocolOnlyBackend``) that does
  not inherit from :class:`BaseBackend`. This is the entry-point
  smoke test promised by TODO 1.69: third-party backends must remain
  usable without :class:`BaseBackend`.

Reference: TODO.md item 1.69 ("Backend plugin contract is too heavyweight").
"""

from __future__ import annotations

from typing import Any

from bo_engine import (
    CURRENT_STATE_ENVELOPE_VERSION,
    BackendStateEnvelope,
    BackendValidationResult,
    BaseBackend,
    BoTorchBackend,
    CapabilityStatus,
    NormalizedProblem,
    build_normalized_problem,
    is_state_envelope,
    required_features,
    unwrap_state,
    wrap_state,
)
from bo_engine.backend import BOBackend, Feature, SuggestionBatch
from bo_engine.backend_testing import (
    assert_initial_design_contract,
    assert_json_serializable,
    assert_state_envelope_contract,
    assert_suggestion_batch_contract,
    assert_validate_capabilities,
)
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)


def _make_simple_spec() -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        random_seed=42,
    )


class _FakeBackend(BaseBackend):
    """Bare-minimum backend that overrides only the abstract members.

    Sole purpose: exercise the :class:`BaseBackend` defaults end-to-end so
    accidental drift (an "override" hidden by a typo, a missing fallback)
    fails this test instead of users' campaigns.
    """

    @property
    def name(self) -> str:
        return "fake"

    @property
    def supported_features(self) -> frozenset[Feature]:
        return frozenset({Feature.MULTI_OBJECTIVE})

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
        pending_points: list[dict[str, Any]] | None = None,
        progress_callback=None,
    ) -> SuggestionBatch:
        _ = observations, backend_state, pending_points, progress_callback
        return SuggestionBatch(
            suggestions=[
                {
                    "parameter_values": {p.name: 0.5 for p in spec.parameters},
                    "provenance": {"iteration": iteration, "batch_index": i},
                }
                for i in range(batch_size)
            ],
            method_info=self.select_methods(spec, len(observations)),
            backend_state=self.wrap_state({"iteration": iteration}),
        )


class _ProtocolOnlyBackend:
    """Pure ``BOBackend`` implementation, not derived from BaseBackend.

    Validates the TODO 1.69 promise that third-party plugins can satisfy
    the protocol without inheriting from :class:`BaseBackend`. The class
    is intentionally minimal: only the members the protocol marks as
    required are filled in.
    """

    name = "protocol-only"
    supported_features = frozenset({Feature.MULTI_OBJECTIVE})

    def validate_spec(self, spec: OptimizationSpec) -> list[str]:
        _ = spec
        return []

    def validate_capabilities(self, spec: OptimizationSpec) -> BackendValidationResult:
        _ = spec
        return BackendValidationResult(backend=self.name)

    def generate_initial_design(
        self,
        spec: OptimizationSpec,
        n_points: int,
    ) -> list[dict[str, Any]]:
        return [{p.name: 0.5 for p in spec.parameters} for _ in range(n_points)]

    def generate_suggestions(
        self,
        spec: OptimizationSpec,
        observations: list[ObservationData],
        batch_size: int,
        iteration: int,
        backend_state: dict[str, Any] | None = None,
        pending_points: list[dict[str, Any]] | None = None,
        progress_callback=None,
    ) -> SuggestionBatch:
        _ = observations, backend_state, pending_points, progress_callback
        return SuggestionBatch(
            suggestions=[
                {
                    "parameter_values": {p.name: 0.5 for p in spec.parameters},
                    "provenance": {"iteration": iteration, "batch_index": i},
                }
                for i in range(batch_size)
            ],
            method_info={"backend": "protocol-only"},
        )

    def compute_hypervolume(self, spec, observations):  # type: ignore[no-untyped-def]
        _ = spec, observations
        return None

    def detect_duplicates(self, new_params, existing_params, tolerance):  # type: ignore[no-untyped-def]
        _ = new_params, existing_params, tolerance
        return []

    def update_state_after_results(self, spec, new_observations, backend_state):  # type: ignore[no-untyped-def]
        _ = spec, new_observations
        return backend_state

    def compute_batch_diversity(self, spec, candidates):  # type: ignore[no-untyped-def]
        _ = spec, candidates
        return None

    def select_methods(self, spec, n_observations):  # type: ignore[no-untyped-def]
        _ = spec, n_observations
        return {"backend": "protocol-only"}

    def compute_diagnostics(self, spec, observations, sections=None, progress_callback=None):  # type: ignore[no-untyped-def]
        _ = spec, observations, sections, progress_callback
        return {}


class TestBackendValidationResult:
    def test_is_compatible_true_when_no_unsupported(self) -> None:
        result = BackendValidationResult(backend="x")
        assert result.is_compatible

    def test_is_compatible_false_with_unsupported_feature(self) -> None:
        from bo_engine.backend_base import CapabilityReport

        result = BackendValidationResult(
            backend="x",
            feature_reports=(
                CapabilityReport(
                    key="multi_objective",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason="r",
                ),
            ),
        )
        assert not result.is_compatible

    def test_warnings_collect_degraded_and_ignored(self) -> None:
        from bo_engine.backend_base import CapabilityReport

        result = BackendValidationResult(
            backend="x",
            option_reports=(
                CapabilityReport(
                    key="turbo_config",
                    status=CapabilityStatus.IGNORED,
                    reason="not supported",
                ),
                CapabilityReport(
                    key="saasbo_config",
                    status=CapabilityStatus.DEGRADED,
                    reason="slow",
                ),
            ),
        )
        assert any("ignored" in w for w in result.warnings)
        assert any("degraded" in w for w in result.warnings)

    def test_supported_features_extracts_supported_only(self) -> None:
        from bo_engine.backend_base import CapabilityReport

        result = BackendValidationResult(
            backend="x",
            feature_reports=(
                CapabilityReport(key="multi_objective", status=CapabilityStatus.SUPPORTED),
                CapabilityReport(
                    key="categorical",
                    status=CapabilityStatus.UNSUPPORTED,
                    reason="r",
                ),
            ),
        )
        assert Feature.MULTI_OBJECTIVE in result.supported_features
        assert Feature.CATEGORICAL not in result.supported_features


class TestRequiredFeaturesMapping:
    def test_basic_spec_has_no_required_features(self) -> None:
        spec = _make_simple_spec()
        assert required_features(spec) == frozenset()

    def test_multi_objective_required(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )
        assert Feature.MULTI_OBJECTIVE in required_features(spec)

    def test_categorical_and_mixed(self) -> None:
        spec = OptimizationSpec(
            parameters=[
                ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
                ParameterSpec(name="c", type=ParameterType.CATEGORICAL, categories=["A", "B"]),
            ],
            objectives=[ObjectiveSpec(name="y", minimize=True)],
        )
        feats = required_features(spec)
        assert Feature.CATEGORICAL in feats
        assert Feature.MIXED_SEARCH_SPACE in feats


class TestNormalizedProblem:
    def test_preserves_parameter_ordering(self) -> None:
        spec = _make_simple_spec()
        problem = build_normalized_problem(spec)
        assert problem.parameter_names == ["x1", "x2"]
        assert problem.parameter_index == {"x1": 0, "x2": 1}

    def test_preserves_objective_direction(self) -> None:
        spec = OptimizationSpec(
            parameters=[ParameterSpec(name="x", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))],
            objectives=[
                ObjectiveSpec(name="y1", minimize=True),
                ObjectiveSpec(name="y2", minimize=False),
            ],
        )
        problem = build_normalized_problem(spec)
        mask = problem.minimize_mask()
        assert bool(mask[0].item()) is True
        assert bool(mask[1].item()) is False

    def test_round_trip_returns_normalized_problem(self) -> None:
        spec = _make_simple_spec()
        problem = build_normalized_problem(spec)
        assert isinstance(problem, NormalizedProblem)
        assert problem.spec is spec


class TestStateEnvelope:
    def test_wrap_unwrap_round_trip(self) -> None:
        payload = {"key": "value", "n": 1}
        wrapped = wrap_state("botorch", payload)
        assert wrapped is not None
        assert wrapped["backend"] == "botorch"
        assert wrapped["schema_version"] == CURRENT_STATE_ENVELOPE_VERSION
        assert wrapped["payload"] == payload
        inner = unwrap_state("botorch", wrapped)
        assert inner == payload

    def test_wrap_none_returns_none(self) -> None:
        assert wrap_state("botorch", None) is None

    def test_unwrap_accepts_legacy_bare_payload(self) -> None:
        legacy = {"campaign_json": "{}"}
        assert unwrap_state("baybe", legacy) == legacy

    def test_unwrap_rejects_foreign_envelope(self) -> None:
        import pytest

        wrapped = wrap_state("baybe", {"k": 1})
        with pytest.raises(ValueError, match="cannot be restored"):
            unwrap_state("botorch", wrapped)

    def test_from_dict_validates_payload_type(self) -> None:
        import pytest

        with pytest.raises(TypeError):
            BackendStateEnvelope.from_dict(
                {"backend": "x", "schema_version": 1, "payload": "not-a-dict"}
            )

    def test_is_state_envelope_detection(self) -> None:
        assert is_state_envelope({"backend": "x", "schema_version": 1, "payload": {}})
        assert not is_state_envelope({"campaign_json": "{}"})
        assert not is_state_envelope(None)


class TestBoTorchBackendContract:
    """Run the reusable contract assertions against the BoTorch backend."""

    def test_validate_capabilities_returns_typed_result(self) -> None:
        spec = _make_simple_spec()
        result = assert_validate_capabilities(BoTorchBackend(), spec)
        assert result.is_compatible

    def test_initial_design_contract(self) -> None:
        spec = _make_simple_spec()
        assert_initial_design_contract(BoTorchBackend(), spec, n_points=3)

    def test_suggestion_batch_contract(self) -> None:
        spec = _make_simple_spec()
        observations = [
            ObservationData(parameter_values={"x1": 0.2, "x2": 0.3}, objective_values={"y": 1.5}),
            ObservationData(parameter_values={"x1": 0.8, "x2": 0.7}, objective_values={"y": 0.3}),
            ObservationData(parameter_values={"x1": 0.5, "x2": 0.5}, objective_values={"y": 0.8}),
        ]
        batch = assert_suggestion_batch_contract(
            BoTorchBackend(),
            spec,
            observations=observations,
            batch_size=1,
        )
        assert_state_envelope_contract(batch.backend_state, backend_name="botorch")


class TestFakeBackendDefaults:
    """Pin the BaseBackend defaults via the bare-minimum FakeBackend."""

    def test_initial_design_default_uses_sobol(self) -> None:
        spec = _make_simple_spec()
        designs = _FakeBackend().generate_initial_design(spec, n_points=4)
        assert len(designs) == 4
        for d in designs:
            assert set(d.keys()) == {"x1", "x2"}

    def test_default_state_envelope_is_wrapped(self) -> None:
        spec = _make_simple_spec()
        backend = _FakeBackend()
        batch = backend.generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=3,
        )
        state = batch.backend_state
        assert state is not None
        assert is_state_envelope(state)
        envelope = BackendStateEnvelope.from_dict(state)
        assert envelope.backend == "fake"

    def test_default_validate_capabilities_reports_only_required(self) -> None:
        # _FakeBackend.supported_features = {MULTI_OBJECTIVE}; a single-objective
        # spec requires no features, so the report is empty.
        spec = _make_simple_spec()
        result = _FakeBackend().validate_capabilities(spec)
        assert result.is_compatible

    def test_default_detect_duplicates(self) -> None:
        backend = _FakeBackend()
        dups = backend.detect_duplicates(
            new_params={"x1": 0.5, "x2": 0.3},
            existing_params=[{"x1": 0.5, "x2": 0.3}],
            tolerance=1e-6,
        )
        assert len(dups) == 1
        assert dups[0].is_exact

    def test_default_validate_spec_delegates_to_capabilities(self) -> None:
        # No degraded/ignored => empty warning list.
        warnings = _FakeBackend().validate_spec(_make_simple_spec())
        assert warnings == []

    def test_default_update_state_passes_through(self) -> None:
        backend = _FakeBackend()
        state = backend.wrap_state({"iteration": 1})
        assert backend.update_state_after_results(_make_simple_spec(), [], state) == state


class TestProtocolOnlyBackend:
    """Pure-protocol implementations must still satisfy the contract."""

    def test_implements_protocol(self) -> None:
        assert isinstance(_ProtocolOnlyBackend(), BOBackend)

    def test_initial_design_contract(self) -> None:
        spec = _make_simple_spec()
        assert_initial_design_contract(_ProtocolOnlyBackend(), spec, n_points=2)

    def test_validate_capabilities_returns_typed_result(self) -> None:
        spec = _make_simple_spec()
        result = assert_validate_capabilities(_ProtocolOnlyBackend(), spec)
        assert result.backend == "protocol-only"

    def test_json_safety(self) -> None:
        spec = _make_simple_spec()
        batch = _ProtocolOnlyBackend().generate_suggestions(
            spec=spec,
            observations=[],
            batch_size=1,
            iteration=1,
        )
        assert_json_serializable(batch.suggestions, context="suggestions")
