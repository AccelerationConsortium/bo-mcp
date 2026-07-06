"""Tests for bo_engine.constants helpers."""

from bo_engine.constants import MIN_OBSERVATIONS_FOR_MODEL, resolve_initial_design_size


class TestResolveInitialDesignSize:
    """Tests for the observation floor gating the switch from Sobol initial
    design to a fitted GP model in ``bo_engine.suggestions.generate_next_batch``.

    A GP with one lengthscale hyperparameter per parameter needs more
    observations than free hyperparameters for a well-posed (non-degenerate)
    maximum-likelihood fit, so the floor is ``n_parameters + 1``, bounded
    below by ``MIN_OBSERVATIONS_FOR_MODEL`` for low-dimensional campaigns.
    """

    def test_floor_scales_with_parameter_count(self) -> None:
        """With no explicit request, the floor grows with n_parameters + 1."""
        assert resolve_initial_design_size(1, None) == MIN_OBSERVATIONS_FOR_MODEL
        assert resolve_initial_design_size(5, None) == 6

    def test_floor_never_drops_below_the_absolute_minimum(self) -> None:
        """A degenerate zero-parameter spec still respects the absolute floor."""
        assert resolve_initial_design_size(0, None) == MIN_OBSERVATIONS_FOR_MODEL

    def test_requested_above_floor_is_honored(self) -> None:
        """A request above the floor is returned unchanged."""
        assert resolve_initial_design_size(2, 10) == 10

    def test_requested_equal_to_floor_is_honored(self) -> None:
        """A request exactly at the floor is returned unchanged."""
        assert resolve_initial_design_size(3, 4) == 4

    def test_requested_below_floor_is_raised_to_the_floor(self) -> None:
        """A request for fewer points than the kernel needs is still raised
        to the floor rather than honored verbatim."""
        assert resolve_initial_design_size(5, 1) == 6
