"""Tests for TuRBO (Trust Region BO) implementation."""

import math
import warnings

import pytest
import torch

from bo_engine import (
    TurboState,
    create_turbo_state,
    should_use_turbo,
    update_turbo_state,
)
from bo_engine.constants import (
    TURBO_INITIAL_LENGTH,
    TURBO_LENGTH_MAX,
    TURBO_LENGTH_MIN,
    TURBO_SUCCESS_TOLERANCE,
    TURBO_UNIT_SCALE_MEAN_ABS_MAX,
    TURBO_UNIT_SCALE_STD_MAX,
)
from bo_engine.turbo import assert_unit_scale_targets


class TestTurboState:
    """Test TurboState dataclass."""

    def test_create_turbo_state(self) -> None:
        """create_turbo_state creates a valid state."""
        state = create_turbo_state(dim=10, batch_size=4)
        assert state.dim == 10
        assert state.batch_size == 4
        assert state.length == 0.8
        assert state.restart_triggered is False

    def test_failure_tolerance_computed(self) -> None:
        """Failure tolerance is computed from dim and batch_size."""
        state = create_turbo_state(dim=20, batch_size=4)
        # Default: ceil(max(4/batch, dim/batch))
        expected = math.ceil(max(4.0 / 4, 20 / 4))  # ceil(max(1, 5)) = 5
        assert state.failure_tolerance == expected

    def test_initial_best_value(self) -> None:
        """Initial best value can be set."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=5.0)
        assert state.best_value == 5.0

    def test_length_bounds(self) -> None:
        """Length min and max are set correctly."""
        state = create_turbo_state(dim=10, batch_size=2)
        assert state.length_min == pytest.approx(0.5**7)
        assert state.length_max == 1.6


class TestUpdateTurboState:
    """Test update_turbo_state function."""

    def test_success_increments_counter(self) -> None:
        """Success increments success counter and resets failure counter."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=0.0)
        # Provide improvement
        state = update_turbo_state(state, torch.tensor([1.0]), minimize=False)

        assert state.success_counter == 1
        assert state.failure_counter == 0

    def test_failure_increments_counter(self) -> None:
        """Failure increments failure counter and resets success counter."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=10.0)
        state = update_turbo_state(state, torch.tensor([5.0]), minimize=False)  # Not improving

        assert state.failure_counter == 1
        assert state.success_counter == 0

    def test_expansion_on_sustained_success(self) -> None:
        """Trust region expands after success_tolerance consecutive improvements."""
        state = TurboState(
            dim=10,
            batch_size=2,
            success_counter=9,  # One more success triggers expansion
            success_tolerance=10,
            failure_tolerance=5,
            length=0.4,
            length_max=1.6,
            best_value=0.0,
        )
        initial_length = state.length

        # Trigger expansion with improvement
        state = update_turbo_state(
            state, torch.tensor([1.0]), minimize=False
        )  # Better than best_value=0

        assert state.length == min(2.0 * initial_length, state.length_max)
        assert state.success_counter == 0  # Reset after expansion

    def test_contraction_on_sustained_failure(self) -> None:
        """Trust region contracts after failure_tolerance consecutive failures."""
        state = TurboState(
            dim=10,
            batch_size=2,
            failure_counter=4,  # One more failure triggers contraction
            failure_tolerance=5,
            length=0.8,
            best_value=10.0,
        )
        initial_length = state.length

        # Trigger contraction
        state = update_turbo_state(state, torch.tensor([5.0]), minimize=False)  # Not improving

        assert state.length == initial_length / 2.0
        assert state.failure_counter == 0  # Reset after contraction

    def test_best_value_updated(self) -> None:
        """Best value is updated when improvement found."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=5.0)
        state = update_turbo_state(state, torch.tensor([10.0]), minimize=False)

        assert state.best_value == 10.0

    def test_best_value_not_decreased(self) -> None:
        """Best value is not decreased on failure."""
        state = create_turbo_state(dim=10, batch_size=2, initial_best_value=10.0)
        state = update_turbo_state(state, torch.tensor([5.0]), minimize=False)

        assert state.best_value == 10.0

    def test_restart_triggered_when_length_too_small(self) -> None:
        """Restart is triggered when length falls below minimum."""
        # Start with a length that after one contraction will be below minimum
        state = TurboState(
            dim=10,
            batch_size=2,
            length=0.01,  # After contraction: 0.005 < 0.0078
            length_min=0.5**7,  # ~0.0078
            failure_counter=4,  # One more failure triggers contraction
            failure_tolerance=5,
            best_value=10.0,
        )

        # Trigger contraction with non-improving value
        state = update_turbo_state(state, torch.tensor([5.0]), minimize=False)

        # Length should now be 0.005, which is less than 0.0078
        assert state.restart_triggered

    def test_handles_batch_tensor(self) -> None:
        """Handles batch of objective values."""
        state = create_turbo_state(dim=10, batch_size=4, initial_best_value=0.0)
        batch_values = torch.tensor([1.0, 2.0, 3.0, 5.0])  # Best is 5.0

        state = update_turbo_state(state, batch_values, minimize=False)

        assert state.best_value == 5.0

    def test_handles_2d_tensor(self) -> None:
        """Handles 2D tensor [batch_size, 1]."""
        state = create_turbo_state(dim=10, batch_size=4, initial_best_value=0.0)
        batch_values = torch.tensor([[1.0], [2.0], [3.0], [5.0]])

        state = update_turbo_state(state, batch_values, minimize=False)

        assert state.best_value == 5.0


class TestShouldUseTurbo:
    """Test should_use_turbo helper."""

    def test_low_dim_returns_false(self) -> None:
        """Low dimensional problems don't need TuRBO."""
        assert should_use_turbo(5) is False
        assert should_use_turbo(10) is False
        assert should_use_turbo(19) is False

    def test_high_dim_returns_true(self) -> None:
        """High dimensional problems benefit from TuRBO."""
        assert should_use_turbo(20) is True
        assert should_use_turbo(50) is True
        assert should_use_turbo(100) is True

    def test_custom_threshold(self) -> None:
        """Custom threshold can be specified."""
        assert should_use_turbo(10, threshold=5) is True
        assert should_use_turbo(10, threshold=15) is False


class TestTurboStateImmutability:
    """Test that TurboState updates return new instances."""

    def test_update_returns_new_instance(self) -> None:
        """update_turbo_state returns a new state instance."""
        state1 = create_turbo_state(dim=10, batch_size=2, initial_best_value=0.0)
        state2 = update_turbo_state(state1, torch.tensor([5.0]), minimize=False)

        # Should be different objects
        assert state1 is not state2
        # Original should be unchanged
        assert state1.best_value == 0.0
        assert state2.best_value == 5.0


class TestTurboConfigDefaultsExposed:
    """``create_turbo_state`` accepts the TuRBO paper tolerances as kwargs.

    Reference: Eriksson et al., NeurIPS 2019, Algorithm 1 — the success /
    failure tolerances and the trust-region length bounds are first-class
    configuration knobs; previously they could only be edited by
    constructing :class:`TurboState` by hand and were silently swallowed
    when supplied via ``TurboConfig``.
    """

    def test_defaults_match_paper(self) -> None:
        """No-override construction reproduces the paper defaults."""
        state = create_turbo_state(dim=10, batch_size=2)
        assert state.length == TURBO_INITIAL_LENGTH
        assert state.length_min == TURBO_LENGTH_MIN
        assert state.length_max == TURBO_LENGTH_MAX
        assert state.success_tolerance == TURBO_SUCCESS_TOLERANCE

    def test_overrides_propagate(self) -> None:
        """Each keyword argument is forwarded onto the dataclass."""
        state = create_turbo_state(
            dim=10,
            batch_size=2,
            initial_length=0.5,
            length_min=0.001,
            length_max=2.0,
            success_tolerance=4,
            failure_tolerance=7,
        )
        assert state.length == pytest.approx(0.5)
        assert state.length_min == pytest.approx(0.001)
        assert state.length_max == pytest.approx(2.0)
        assert state.success_tolerance == 4
        assert state.failure_tolerance == 7


class TestAssertUnitScaleTargets:
    """``assert_unit_scale_targets`` warns when raw train_y is far from unit scale.

    The default expand/contract tolerances inside ``update_turbo_state`` use
    ``IMPROVEMENT_TOLERANCE_RELATIVE * abs(best_value)`` to decide whether a
    batch improved. With raw targets several orders of magnitude off unit
    scale that threshold lands either far below sensor noise or above the
    realistic improvement step — either way the trust-region dynamics
    mis-fire (Eriksson et al., 2019, §3.2). The assertion mirrors the
    BoTorch single-task model's ``InputDataWarning`` so users see a similar
    nudge for the TuRBO path.
    """

    def test_unit_scale_targets_emit_no_warning(self) -> None:
        """Standardized-style targets pass silently."""
        train_y = torch.tensor([-1.2, -0.4, 0.1, 0.7, 1.5], dtype=torch.float64)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert_unit_scale_targets(train_y)
        assert all(w.category is not UserWarning for w in captured)

    def test_large_magnitude_emits_warning(self) -> None:
        """Targets with |mean| > threshold raise a UserWarning."""
        train_y = torch.tensor([100.0, 110.0, 120.0, 130.0], dtype=torch.float64)
        assert abs(train_y.mean().item()) > TURBO_UNIT_SCALE_MEAN_ABS_MAX
        with pytest.warns(UserWarning, match="unit-standardized targets"):
            assert_unit_scale_targets(train_y)

    def test_extreme_spread_emits_warning(self) -> None:
        """Targets with std outside the accepted band raise a UserWarning."""
        train_y = torch.tensor([-100.0, -50.0, 0.0, 50.0, 100.0], dtype=torch.float64)
        assert train_y.std(unbiased=True).item() > TURBO_UNIT_SCALE_STD_MAX
        with pytest.warns(UserWarning, match="unit-standardized targets"):
            assert_unit_scale_targets(train_y)

    def test_single_observation_is_silent(self) -> None:
        """A single-row tensor cannot have a meaningful std → no warning."""
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            assert_unit_scale_targets(torch.tensor([42.0]))
        assert all(w.category is not UserWarning for w in captured)


class TestTurboStateInvariants:
    """``TurboState.__post_init__`` rejects garbage tolerance configurations.

    Defense in depth alongside the Pydantic ``TurboConfig`` validators: when
    a backend builds a ``TurboState`` directly (state deserialization, tests,
    non-MCP entry points) the dataclass still refuses invariants that would
    silently break the expand/contract dynamics in ``update_turbo_state``.

    Reference: Eriksson et al., NeurIPS 2019, Algorithm 1 — the algorithm
    assumes a non-empty trust-region operating band (``length_min <
    length_max``) and at least one consecutive success / failure batch
    before adapting the trust region; a zero tolerance would freeze the
    region forever.
    """

    def test_non_positive_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="length must be positive"):
            TurboState(dim=4, batch_size=2, length=0.0)

    def test_non_positive_length_min_rejected(self) -> None:
        with pytest.raises(ValueError, match="length_min must be positive"):
            TurboState(dim=4, batch_size=2, length_min=-0.1)

    def test_non_positive_length_max_rejected(self) -> None:
        with pytest.raises(ValueError, match="length_max must be positive"):
            TurboState(dim=4, batch_size=2, length=0.5, length_max=-1.0)

    def test_inverted_length_band_rejected(self) -> None:
        """``length_min >= length_max`` collapses the operating band."""
        with pytest.raises(ValueError, match="strictly less than"):
            TurboState(
                dim=4,
                batch_size=2,
                length=0.5,
                length_min=1.5,
                length_max=1.0,
            )

    def test_success_tolerance_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="success_tolerance"):
            TurboState(dim=4, batch_size=2, success_tolerance=0)

    def test_failure_tolerance_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="failure_tolerance"):
            TurboState(dim=4, batch_size=2, failure_tolerance=0)

    def test_create_turbo_state_rejects_invalid_overrides(self) -> None:
        """The ``create_turbo_state`` factory inherits the invariants."""
        with pytest.raises(ValueError, match="strictly less than"):
            create_turbo_state(
                dim=4,
                batch_size=2,
                length_min=2.0,
                length_max=1.0,
            )

    def test_length_above_length_max_rejected(self) -> None:
        """``length > length_max`` cannot occur in the live algorithm.

        ``update_turbo_state`` clamps every expansion at ``length_max`` so a
        post-update state always satisfies ``length <= length_max``. Direct
        construction with ``length > length_max`` is therefore incoherent.
        """
        with pytest.raises(ValueError, match="must not exceed length_max"):
            TurboState(dim=4, batch_size=2, length=2.0, length_max=1.6)

    def test_length_below_length_min_without_restart_flag_rejected(self) -> None:
        """A below-min length is only coherent paired with ``restart_triggered=True``.

        That pairing *is* the restart signal :func:`update_turbo_state`
        emits after the final contraction step. Direct construction with
        ``length < length_min`` and ``restart_triggered=False`` would
        masquerade an already-restarted state as still operational.
        """
        with pytest.raises(ValueError, match="below length_min"):
            TurboState(
                dim=4,
                batch_size=2,
                length=0.001,
                length_min=0.01,
                restart_triggered=False,
            )

    def test_length_below_length_min_with_restart_flag_accepted(self) -> None:
        """The post-contraction restart state is the legitimate exception."""
        state = TurboState(
            dim=4,
            batch_size=2,
            length=0.001,
            length_min=0.01,
            restart_triggered=True,
        )
        assert state.restart_triggered is True
        assert state.length == pytest.approx(0.001)

    def test_update_turbo_state_preserves_post_contraction_invariant(self) -> None:
        """End-to-end: contraction past ``length_min`` produces a coherent state.

        Mirrors the test_botorch_tutorials_turbo restart fixture — a
        ``length`` that contracts below ``length_min`` must come back paired
        with ``restart_triggered=True`` so the new ``__post_init__`` check
        accepts it.
        """
        state = TurboState(
            dim=10,
            batch_size=2,
            length=0.01,  # contraction → 0.005 < length_min
            length_min=0.5**7,
            failure_counter=4,
            failure_tolerance=5,
            best_value=10.0,
        )
        updated = update_turbo_state(state, torch.tensor([5.0]), minimize=False)
        assert updated.restart_triggered is True
        assert updated.length < updated.length_min


class TestTrustRegionNonUnitCube:
    """Trust-region geometry on raw (non-unit-cube) parameter bounds.

    The TuRBO trust region is defined in the normalized [0, 1] cube
    (Eriksson et al., "Scalable Global Optimization via Local Bayesian
    Optimization", NeurIPS 2019, §3.1: side length ``L`` of the hypercube
    after rescaling the domain to the unit cube). The suggestion pipeline
    therefore normalizes raw inputs before computing the region and maps
    the result back to raw bounds. These cases pin that round-trip on a
    domain that is *not* the unit cube — where interpreting raw
    coordinates as normalized degenerates the region to a sliver at a
    bound.
    """

    BOUNDS = ((-5.0, 10.0), (-5.0, 10.0))

    def _fit_and_compute(
        self, train_x: torch.Tensor, train_y_bo: torch.Tensor, length: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from bo_engine.models import create_and_fit_single_task_model
        from bo_engine.suggestions_single_objective import _compute_turbo_bounds

        bounds = torch.tensor(
            [[b[0] for b in self.BOUNDS], [b[1] for b in self.BOUNDS]],
            dtype=torch.float64,
        )
        model = create_and_fit_single_task_model(train_x, train_y_bo, bounds)
        state = TurboState(dim=2, batch_size=2, length=length, best_value=0.0)
        opt_bounds, _info = _compute_turbo_bounds(state, train_x, train_y_bo, model, bounds)
        return opt_bounds, bounds

    def test_trust_region_brackets_best_point(self) -> None:
        """The raw-scale trust region must contain the incumbent best point."""
        torch.manual_seed(3)
        train_x = torch.rand(20, 2, dtype=torch.float64) * 15.0 - 5.0
        best_point = torch.tensor([2.0, 2.5], dtype=torch.float64)
        train_x[7] = best_point
        # Maximization-form targets with a unique best at index 7.
        train_y_bo = -((train_x - best_point).pow(2).sum(dim=-1, keepdim=True)) / 100.0

        opt_bounds, bounds = self._fit_and_compute(train_x, train_y_bo, length=0.2)

        assert (opt_bounds[0] <= best_point + 1e-9).all(), (
            f"Trust region lower bound {opt_bounds[0].tolist()} does not "
            f"bracket the best point {best_point.tolist()}"
        )
        assert (opt_bounds[1] >= best_point - 1e-9).all(), (
            f"Trust region upper bound {opt_bounds[1].tolist()} does not "
            f"bracket the best point {best_point.tolist()}"
        )
        # The region must be a genuine local restriction of the domain, not
        # a degenerate sliver pinned at a bound and not the full domain.
        widths = opt_bounds[1] - opt_bounds[0]
        full_range = bounds[1] - bounds[0]
        assert (widths > 0.01 * full_range).all(), f"Degenerate trust region: {widths.tolist()}"
        assert (widths < full_range).all(), f"Trust region did not restrict: {widths.tolist()}"

    def test_trust_region_width_matches_length_geometry(self) -> None:
        """Normalized widths obey the lengthscale-weighted geometry of the paper.

        Per Eriksson et al. (2019) §3.1 the per-dimension half-widths are
        ``weights * length / 2`` with the weights normalized to geometric
        mean 1, so (absent clamping) the *product* of normalized widths
        equals ``length ** d``.
        """
        torch.manual_seed(3)
        length = 0.2
        train_x = torch.rand(20, 2, dtype=torch.float64) * 15.0 - 5.0
        best_point = torch.tensor([2.0, 2.5], dtype=torch.float64)
        train_x[7] = best_point
        train_y_bo = -((train_x - best_point).pow(2).sum(dim=-1, keepdim=True)) / 100.0

        opt_bounds, bounds = self._fit_and_compute(train_x, train_y_bo, length=length)

        widths_norm = (opt_bounds[1] - opt_bounds[0]) / (bounds[1] - bounds[0])
        expected_volume = length ** widths_norm.numel()
        assert float(widths_norm.prod().item()) == pytest.approx(expected_volume, rel=0.05), (
            f"Normalized widths {widths_norm.tolist()} are inconsistent with "
            f"trust-region length {length} (geometric-mean-1 weights)."
        )

    def test_trust_region_centers_on_argmin_for_minimize_campaign(self) -> None:
        """For a minimize campaign the region centers on the argmin's inputs.

        The pipeline hands ``get_turbo_bounds`` maximization-form targets
        (``-y`` for minimize), so the ``argmax`` centering used by the
        helper must land on the raw objective's *minimum*.
        """
        torch.manual_seed(5)
        train_x = torch.rand(20, 2, dtype=torch.float64) * 15.0 - 5.0
        best_point = torch.tensor([2.0, 2.5], dtype=torch.float64)
        train_x[11] = best_point
        raw_y = (train_x - best_point).pow(2).sum(dim=-1, keepdim=True) / 100.0
        assert int(raw_y.argmin().item()) == 11

        # Boundary conversion exactly as in the suggestion pipeline.
        train_y_bo = -raw_y

        opt_bounds, _bounds = self._fit_and_compute(train_x, train_y_bo, length=0.2)
        center = (opt_bounds[0] + opt_bounds[1]) / 2
        assert torch.allclose(center, best_point, atol=1e-6), (
            f"Trust region center {center.tolist()} is not the argmin's "
            f"inputs {best_point.tolist()} — wrong incumbent selection."
        )
