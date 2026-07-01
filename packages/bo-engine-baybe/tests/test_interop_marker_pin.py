"""Pin the engine↔BayBE substance interop markers to their source of truth.

The BoTorch auto-routing guardrail in ``bo_engine.botorch_backend`` vetoes a
``role=substance`` parameter by reading string markers from
:mod:`bo_engine.interop`. ``bo-engine`` cannot import ``bo_engine_baybe``
(that would invert the dependency), so the markers are duplicated values, not
references. This suite is the only one allowed to import *both* packages, so it
is where we assert the engine-side markers still equal the BayBE-side source of
truth. A future BayBE-side rename (the ``SubstanceEncoding``/role enums have
drifted historically) then fails CI here instead of silently disabling the
guardrail at a user's runtime.
"""

from __future__ import annotations

from bo_engine.interop import (
    BAYBE_BACKEND_NAME,
    BAYBE_CUSTOM_ROLE,
    BAYBE_PARAMETER_ROLE_KEY,
    BAYBE_SUBSTANCE_ROLE,
)
from bo_engine_baybe.backend import BayBEBackend
from bo_engine_baybe.options import BayBEParameterOptions, BayBEParameterRole


def test_substance_role_marker_matches_baybe_enum() -> None:
    assert BAYBE_SUBSTANCE_ROLE == BayBEParameterRole.SUBSTANCE.value


def test_custom_role_marker_matches_baybe_enum() -> None:
    assert BAYBE_CUSTOM_ROLE == BayBEParameterRole.CUSTOM.value


def test_backend_name_marker_matches_registered_backend() -> None:
    # The same string keys ``parameter_options["baybe"]`` and names the
    # backend in the entry-point registry.
    assert BAYBE_BACKEND_NAME == BayBEBackend().name


def test_role_key_marker_is_a_real_option_field() -> None:
    # The guardrail reads ``parameter_options["baybe"]["role"]``; pin the
    # field name so renaming the model attribute fails here.
    assert BAYBE_PARAMETER_ROLE_KEY in BayBEParameterOptions.model_fields
