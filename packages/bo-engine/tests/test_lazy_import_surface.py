"""Import-surface contract for the lazy ``bo_engine`` facade.

``bo_engine/__init__.py`` resolves its ~300 public symbols via a PEP 562
module ``__getattr__`` instead of eager imports, so consumers that only
need light modules (``bo_engine.types``, ``bo_engine.constants``) no
longer pay a multi-second torch/botorch/gpytorch/sklearn cold import.
These tests pin both halves of the contract: the documented names stay
importable, and the light entry points stay light.

Reference: PEP 562 "Module __getattr__ and __dir__"
(https://peps.python.org/pep-0562/) — the standard mechanism for
"deprecating or lazily importing module attributes".
"""

from __future__ import annotations

import subprocess
import sys

import pytest

HEAVY_MODULES = ("torch", "botorch", "gpytorch", "sklearn")


def _run_in_fresh_interpreter(code: str) -> None:
    """Run ``code`` in a clean child interpreter; assert it exits 0."""
    completed = subprocess.run(  # noqa: S603 -- fixed argv, test-controlled code
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_package_import_does_not_load_heavy_dependencies() -> None:
    code = "\n".join(
        [
            "import sys",
            "import bo_engine",
            "import bo_engine.types",
            "import bo_engine.constants",
            *[
                f"assert '{mod}' not in sys.modules, '{mod} loaded eagerly'"
                for mod in HEAVY_MODULES
            ],
        ]
    )
    _run_in_fresh_interpreter(code)


def test_documented_entry_points_resolve_lazily() -> None:
    """The names the package docstring advertises stay importable."""
    code = "\n".join(
        [
            "from bo_engine import generate_next_batch, OptimizationSpec",
            "from bo_engine.types import ParameterSpec, ObjectiveSpec, ParameterType",
            "from bo_engine import BoTorchBackend, BOBackend",
            "assert callable(generate_next_batch)",
        ]
    )
    _run_in_fresh_interpreter(code)


def test_every_all_entry_resolves() -> None:
    """No name in ``__all__`` is a dangling map entry."""
    import bo_engine

    for name in bo_engine.__all__:
        assert getattr(bo_engine, name) is not None


def test_dir_exposes_the_lazy_surface() -> None:
    import bo_engine

    listing = dir(bo_engine)
    assert "generate_next_batch" in listing
    assert "OptimizationSpec" in listing


def test_unknown_attribute_raises_attribute_error() -> None:
    import bo_engine

    with pytest.raises(AttributeError, match="definitely_not_a_symbol"):
        _ = bo_engine.definitely_not_a_symbol
