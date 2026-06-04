"""Canary: ``baybe[chem]`` is wired in as a default dependency.

The ``role=substance`` path builds a BayBE ``SubstanceParameter`` whose
descriptor table needs RDKit + scikit-fingerprints (the ``baybe[chem]``
extra). Those are now declared as default dependencies of
``bo-engine-baybe`` (no longer optional). If a future lockfile/resolution
change drops them, the descriptor path would fail deep inside BayBE at
suggestion time; this fast test catches the regression at import time
instead. ``_CHEMISTRY_AVAILABLE`` is the single source of truth the
capability layer consults, so it is pinned here too.
"""

from __future__ import annotations

from bo_engine_baybe.capabilities import _CHEMISTRY_AVAILABLE


def test_baybe_optional_chem_imports() -> None:
    import baybe._optional.chem  # noqa: F401 — import side effect is the assertion


def test_chemistry_available_flag_is_true() -> None:
    assert _CHEMISTRY_AVAILABLE is True
