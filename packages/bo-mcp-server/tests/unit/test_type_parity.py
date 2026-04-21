"""Tests that verify type parity between bo-mcp-server domain and bo-engine.

Step 7, item 1.10: Both packages define ParameterType, ConstraintType, and
AcquisitionMethod enums. These tests ensure the enum values stay in sync
so that the converter in converters.py never hits a KeyError.

Reference: https://docs.python.org/3/library/enum.html
"""

from bo_engine.types import AcquisitionMethod as BOAcquisitionMethod
from bo_engine.types import ConstraintType as BOConstraintType
from bo_engine.types import ParameterType as BOParameterType

from bo_mcp_server.domain import (
    AcquisitionMethod,
    ConstraintType,
    ParameterType,
)


class TestEnumParity:
    """Verify enum values match between server domain and bo-engine."""

    def test_parameter_type_values_match(self) -> None:
        """Server ParameterType values must be a subset of bo-engine values."""
        server_values = {m.value for m in ParameterType}
        engine_values = {m.value for m in BOParameterType}
        missing = server_values - engine_values
        assert not missing, f"Server ParameterType has values not in bo-engine: {missing}"

    def test_constraint_type_values_match(self) -> None:
        """Server ConstraintType values must be a subset of bo-engine values."""
        server_values = {m.value for m in ConstraintType}
        engine_values = {m.value for m in BOConstraintType}
        missing = server_values - engine_values
        assert not missing, f"Server ConstraintType has values not in bo-engine: {missing}"

    def test_acquisition_method_values_match(self) -> None:
        """Server AcquisitionMethod values must be a subset of bo-engine."""
        server_values = {m.value for m in AcquisitionMethod}
        engine_values = {m.value for m in BOAcquisitionMethod}
        missing = server_values - engine_values
        assert not missing, f"Server AcquisitionMethod has values not in bo-engine: {missing}"

    def test_parameter_type_bidirectional(self) -> None:
        """Both packages should have the same ParameterType values."""
        server_values = {m.value for m in ParameterType}
        engine_values = {m.value for m in BOParameterType}
        assert server_values == engine_values

    def test_constraint_type_bidirectional(self) -> None:
        """Both packages should have the same ConstraintType values."""
        server_values = {m.value for m in ConstraintType}
        engine_values = {m.value for m in BOConstraintType}
        assert server_values == engine_values

    def test_acquisition_method_bidirectional(self) -> None:
        """Both packages should have the same AcquisitionMethod values."""
        server_values = {m.value for m in AcquisitionMethod}
        engine_values = {m.value for m in BOAcquisitionMethod}
        assert server_values == engine_values
