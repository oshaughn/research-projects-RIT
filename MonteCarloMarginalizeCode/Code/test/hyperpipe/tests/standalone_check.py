"""
Standalone (no-pytest) smoke test --- useful for ``pixi run test-minimal``.

Verifies the same things as the pytest suite, but runs as a regular
script so it's easy to invoke from a make rule or a CI job that doesn't
want a pytest dependency.

Locates RIFT_ROOT by:
  1. honoring the ``$RIFT_ROOT`` env var if set;
  2. otherwise walking up from this file's location
     (``parents[5]`` of the canonical
     ``test/hyperpipe/tests/standalone_check.py`` path is the RIFT root).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path


_HERE = Path(__file__).resolve()


def _is_rift_root(p: Path) -> bool:
    return (p / "MonteCarloMarginalizeCode" / "Code" / "RIFT" / "hyperpipe").exists()


def _rift_root() -> Path:
    env = os.environ.get("RIFT_ROOT")
    if env:
        p = Path(env).resolve()
        if _is_rift_root(p):
            return p
    if len(_HERE.parents) > 5:
        candidate = _HERE.parents[5]
        if _is_rift_root(candidate):
            return candidate
    raise SystemExit(
        "Could not locate RIFT root. Set $RIFT_ROOT or run from "
        "$RIFT_ROOT/MonteCarloMarginalizeCode/Code/test/hyperpipe/tests/."
    )


RIFT_ROOT = _rift_root()
HP = RIFT_ROOT / "MonteCarloMarginalizeCode" / "Code" / "RIFT" / "hyperpipe"
RIFT_PY = RIFT_ROOT / "MonteCarloMarginalizeCode" / "Code"

# Bypass RIFT/__init__.py so we don't need the full lalsuite stack to run
# the smoke test --- this script is intended to be runnable from any
# Python env that has numpy.
fake_rift = types.ModuleType("RIFT")
fake_rift.__path__ = [str(RIFT_PY / "RIFT")]
sys.modules["RIFT"] = fake_rift
fake_hp = types.ModuleType("RIFT.hyperpipe")
fake_hp.__path__ = [str(HP)]
sys.modules["RIFT.hyperpipe"] = fake_hp


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


coords = _load("RIFT.hyperpipe.coords", HP / "coords.py")
config = _load("RIFT.hyperpipe.config", HP / "config.py")
marg_list = _load("RIFT.hyperpipe.marg_list", HP / "marg_list.py")
drivers_base = _load("RIFT.hyperpipe.drivers.base", HP / "drivers" / "base.py")


def run_checks() -> None:
    # 1. coord-spec
    spec = coords.HyperCoordSpec.from_strings(
        coords_fit="x y z",
        coords_sample="x:[-8,8] y:[-8,8] z:[-8,8]",
    )
    spec.validate(strict_import=False)
    assert "--integration-parameter-range x:[-8,8]" in spec.to_post_args()

    # 2. mono marg-list assembly
    with tempfile.TemporaryDirectory() as tmpd:
        base = Path(tmpd) / "base"
        run = Path(tmpd) / "run"
        base.mkdir()
        run.mkdir()
        (base / "example.py").write_text("#!/usr/bin/env python\n")
        os.chmod(base / "example.py", 0o755)
        cfg = {
            "marg-list": [
                {"name": "g", "exe": "example.py", "args": "--ok",
                 "event-file": None, "n-chunk": 100, "coord-module": None}
            ]
        }
        m = marg_list.assemble_marg_list(cfg, base_dir=str(base), run_dir=str(run))
        assert m.n_chunks == [100]
        assert (run / "event-0.net").read_text().strip() == "empty_event_file"

    # 3. validate_config rejects empty config
    try:
        config.validate_config({})
    except ValueError:
        pass
    else:
        raise AssertionError("validate_config({}) should have raised")

    # 4. base driver round-trip
    with tempfile.TemporaryDirectory() as tmpd:
        grid = Path(tmpd) / "g.dat"
        grid.write_text("# lnL sigma_lnL x y z\n0 0 1.0 2.0 3.0\n")
        rows, cols = drivers_base.read_grid(f"file:{grid}")
        assert cols == ["x", "y", "z"]
        rows[0, 0] = "-3.1415926535"
        out = drivers_base.write_marg_output(
            rows, cols,
            fname_output_integral="f.txt",
            outdir=str(Path(tmpd) / "out"),
            fname=None,
            conforming_output_name=True,
        )
        assert "-3.1415926535" in Path(out).read_text()

    print(f"standalone_check: ALL OK  (RIFT_ROOT={RIFT_ROOT})")


if __name__ == "__main__":
    run_checks()
