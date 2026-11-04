"""
pytest fixtures + path wiring for the in-tree RIFT.hyperpipe test suite.

Resolves ``RIFT_ROOT`` in two ways, in order:
  1. the ``$RIFT_ROOT`` environment variable, if set (e.g. by pixi's
     activation block);
  2. walking up from this file's location --- with the canonical layout
     ``$RIFT_ROOT/MonteCarloMarginalizeCode/Code/test/hyperpipe/tests/conftest.py``,
     ``parents[5]`` of ``__file__`` is ``$RIFT_ROOT``.

This means the suite works from any RIFT clone with no user-specific
paths to edit.
"""
from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


_HERE = Path(__file__).resolve()


def _is_rift_root(p: Path) -> bool:
    return (p / "MonteCarloMarginalizeCode" / "Code" / "RIFT" / "hyperpipe").exists()


def _rift_root() -> Path:
    # 1. honor explicit env var if it points at a real RIFT clone
    env = os.environ.get("RIFT_ROOT")
    if env:
        p = Path(env).resolve()
        if _is_rift_root(p):
            return p
    # 2. canonical in-tree location: parents[5] of this file
    if len(_HERE.parents) > 5:
        candidate = _HERE.parents[5]
        if _is_rift_root(candidate):
            return candidate
    # 3. give up
    pytest.skip(
        "Could not locate RIFT root. Either set $RIFT_ROOT or run this "
        "suite from its canonical location at "
        "$RIFT_ROOT/MonteCarloMarginalizeCode/Code/test/hyperpipe/tests/."
    )


@pytest.fixture(scope="session")
def rift_root() -> Path:
    return _rift_root()


@pytest.fixture(scope="session")
def hyperpipe_dir(rift_root: Path) -> Path:
    return rift_root / "MonteCarloMarginalizeCode" / "Code" / "RIFT" / "hyperpipe"


@pytest.fixture(scope="session")
def rift_bin(rift_root: Path) -> Path:
    return rift_root / "MonteCarloMarginalizeCode" / "Code" / "bin"


@pytest.fixture(scope="session")
def rift_py(rift_root: Path) -> Path:
    return rift_root / "MonteCarloMarginalizeCode" / "Code"


@pytest.fixture(scope="session")
def hp_modules(rift_py: Path):
    """Import the four lightweight hyperpipe modules and expose on a namespace."""
    if str(rift_py) not in sys.path:
        sys.path.insert(0, str(rift_py))
    coords = importlib.import_module("RIFT.hyperpipe.coords")
    config = importlib.import_module("RIFT.hyperpipe.config")
    marg_list = importlib.import_module("RIFT.hyperpipe.marg_list")
    drivers_base = importlib.import_module("RIFT.hyperpipe.drivers.base")
    return types.SimpleNamespace(
        coords=coords,
        config=config,
        marg_list=marg_list,
        drivers_base=drivers_base,
    )
