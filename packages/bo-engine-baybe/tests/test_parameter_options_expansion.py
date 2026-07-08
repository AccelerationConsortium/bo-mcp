"""Parameter-level BayBE options: tolerance, generalized active_values, kwargs.

Covers the §-completion of the per-parameter surface: the
numerical-discrete measurement-matching ``tolerance`` (with its
``< min_gap / 2`` intake guard), ``active_values`` on plain categorical /
substance / custom roles (previously task-only), and the substance
fingerprint/conformer keyword passthroughs.

Reference: BayBE parameters userguide
(https://emdgroup.github.io/baybe/stable/userguide/parameters.html).
"""

from __future__ import annotations

from typing import Any

import pytest
from baybe.parameters import CategoricalParameter, NumericalDiscreteParameter

from bo_engine.types import (
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.converters import spec_to_parameters
from bo_engine_baybe.options import BayBEParameterOptions


def _spec(parameter: ParameterSpec) -> OptimizationSpec:
    return OptimizationSpec(
        parameters=[parameter],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
    )


def _discrete(baybe_options: dict[str, Any] | None = None) -> ParameterSpec:
    return ParameterSpec(
        name="a",
        type=ParameterType.DISCRETE,
        values=[0.0, 1.0, 2.0],
        parameter_options={"baybe": baybe_options} if baybe_options is not None else None,
    )


def _categorical(baybe_options: dict[str, Any] | None = None) -> ParameterSpec:
    return ParameterSpec(
        name="c",
        type=ParameterType.CATEGORICAL,
        categories=["x", "y", "z"],
        parameter_options={"baybe": baybe_options} if baybe_options is not None else None,
    )


class TestTolerance:
    def test_tolerance_reaches_numerical_discrete_parameter(self) -> None:
        built = spec_to_parameters(_spec(_discrete({"tolerance": 0.25})))
        assert isinstance(built[0], NumericalDiscreteParameter)
        assert built[0].tolerance == pytest.approx(0.25)

    def test_unset_tolerance_keeps_baybe_default(self) -> None:
        built = spec_to_parameters(_spec(_discrete()))
        assert isinstance(built[0], NumericalDiscreteParameter)
        assert built[0].tolerance == pytest.approx(0.0)

    def test_tolerance_above_half_min_gap_reported_with_bound(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(_discrete({"tolerance": 0.5})))
        assert not result.is_compatible
        report = next(r for r in result.unsupported if "tolerance" in r.key)
        assert "tolerance < 0.5" in report.reason

    def test_tolerance_on_categorical_rejected(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(_categorical({"tolerance": 0.1})))
        assert not result.is_compatible


class TestActiveValues:
    def test_active_values_on_plain_categorical(self) -> None:
        built = spec_to_parameters(_spec(_categorical({"active_values": ["x", "y"]})))
        assert isinstance(built[0], CategoricalParameter)
        assert set(built[0].active_values) == {"x", "y"}

    def test_active_values_subset_check_applies_to_all_roles(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(_categorical({"active_values": ["nope"]}))
        )
        assert not result.is_compatible
        assert any("active_values" in r.key for r in result.unsupported)

    def test_active_values_on_numeric_parameter_rejected(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(_discrete({"active_values": ["1.0"]})))
        assert not result.is_compatible


class TestSubstanceKwargs:
    def test_kwargs_require_substance_role(self) -> None:
        result = BayBEBackend().validate_capabilities(
            _spec(_categorical({"kwargs_fingerprint": {"n_bits": 512}}))
        )
        assert not result.is_compatible
        assert any("kwargs_fingerprint" in r.key for r in result.unsupported)

    def test_options_model_accepts_kwargs_dicts(self) -> None:
        options = BayBEParameterOptions.model_validate(
            {
                "role": "substance",
                "substance_data": {"x": "C"},
                "kwargs_fingerprint": {"fp_size": 512},
                "kwargs_conformer": {"max_gen_attempts": 3000},
            }
        )
        assert options.kwargs_fingerprint == {"fp_size": 512}
        assert options.kwargs_conformer == {"max_gen_attempts": 3000}


class TestToleranceBoundsOnlyGrid:
    """Minor: the tolerance guard covers bounds-defined integer grids too.

    A bounds-only discrete parameter spans the integer grid over
    ``[ceil(lo), floor(hi)]`` (gap 1.0, so tolerance must stay < 0.5);
    the pre-fix guard derived the grid from explicit ``values`` only and
    let the over-large tolerance crash inside BayBE's constructor.
    """

    @staticmethod
    def _bounds_only(tolerance: float) -> ParameterSpec:
        return ParameterSpec(
            name="a",
            type=ParameterType.DISCRETE,
            bounds=(0.0, 5.0),
            parameter_options={"baybe": {"tolerance": tolerance}},
        )

    def test_over_large_tolerance_is_reported_at_intake(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(self._bounds_only(0.5)))
        assert not result.is_compatible
        report = next(r for r in result.unsupported if "tolerance" in r.key)
        assert "tolerance < 0.5" in report.reason

    def test_valid_tolerance_on_bounds_only_grid_is_accepted(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(self._bounds_only(0.25)))
        assert result.is_compatible
        built = spec_to_parameters(_spec(self._bounds_only(0.25)))
        assert isinstance(built[0], NumericalDiscreteParameter)
        assert built[0].tolerance == pytest.approx(0.25)


class TestActiveValuesContract:
    """Minor: empty active_values are rejected; non-categorical gets one report."""

    def test_empty_active_values_rejected_by_schema(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            BayBEParameterOptions.model_validate({"active_values": []})

    def test_empty_active_values_surface_as_capability_report(self) -> None:
        result = BayBEBackend().validate_capabilities(_spec(_categorical({"active_values": []})))
        assert not result.is_compatible

    def test_non_categorical_active_values_yield_single_report(self) -> None:
        """One clear wrong-type report, not a duplicate 'not in []' echo."""
        result = BayBEBackend().validate_capabilities(_spec(_discrete({"active_values": ("x",)})))
        assert not result.is_compatible
        reports = [r for r in result.unsupported if "active_values" in r.key]
        assert len(reports) == 1
        assert "categorical-family" in reports[0].reason
