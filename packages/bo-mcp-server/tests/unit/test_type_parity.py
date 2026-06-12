"""Tests that verify type identity between bo-mcp-server domain and bo-engine.

``ParameterType``, ``ConstraintType``, and ``AcquisitionMethod`` are declared
once in :mod:`bo_engine.types` (the lower-dependency package) and re-exported
through :mod:`bo_mcp_server.domain`. Both import paths must resolve to the
*same* enum object so the converter in :mod:`bo_mcp_server.converters` does
not need a manual mapping layer.

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


class TestEnumIdentity:
    """Verify domain enums are the bo-engine enums (not separate copies)."""

    def test_parameter_type_is_engine_enum(self) -> None:
        """``bo_mcp_server.domain.ParameterType`` is the bo-engine enum."""
        assert ParameterType is BOParameterType

    def test_constraint_type_is_engine_enum(self) -> None:
        """``bo_mcp_server.domain.ConstraintType`` is the bo-engine enum."""
        assert ConstraintType is BOConstraintType

    def test_acquisition_method_is_engine_enum(self) -> None:
        """``bo_mcp_server.domain.AcquisitionMethod`` is the bo-engine enum."""
        assert AcquisitionMethod is BOAcquisitionMethod
