"""Substance-encoding hardening: default pin + BayBE-enum drift guard.

``BayBESubstanceEncoding`` is a hand-curated subset of BayBE's 37-member
``SubstanceEncoding`` enum, which has been renamed across versions. These fast
tests pin the documented default and assert every curated member still exists
in the installed BayBE so a BayBE bump that renames an encoding fails in CI
rather than at a user's runtime. Mirrors BayBE's ``SubstanceParameter``
encoding choices
(https://emdgroup.github.io/baybe/stable/examples/Basics/parameters.html).
"""

from __future__ import annotations

from baybe.parameters.enum import SubstanceEncoding

from bo_engine_baybe.options import (
    DEFAULT_SUBSTANCE_ENCODING,
    BayBESubstanceEncoding,
)


def test_default_substance_encoding_is_mordred() -> None:
    # The documented default (kept as a named constant, not an inline string).
    assert DEFAULT_SUBSTANCE_ENCODING is BayBESubstanceEncoding.MORDRED


def test_curated_encodings_exist_in_pinned_baybe() -> None:
    valid = {m.name for m in SubstanceEncoding}
    for member in BayBESubstanceEncoding:
        assert member.value in valid, (
            f"BayBESubstanceEncoding.{member.name}={member.value!r} is no longer a "
            f"member of baybe.parameters.enum.SubstanceEncoding; a BayBE bump renamed it."
        )


def test_curated_encodings_use_current_baybe_names() -> None:
    assert "RDKIT" not in {m.value for m in BayBESubstanceEncoding}
    assert BayBESubstanceEncoding.RDKIT2DDESCRIPTORS.value == "RDKIT2DDESCRIPTORS"
    assert BayBESubstanceEncoding.RDKITFINGERPRINT.value == "RDKITFINGERPRINT"
