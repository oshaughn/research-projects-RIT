"""CIP startup must not import optional heavy backends unless the options select them.

Base rift_O4d defers these until after argument parsing; an eager top-level import
(torch via senni/gpytorch_wrapper, sklearn via matern_gp) costs ~8 s per CIP job.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest

CIP = Path(__file__).resolve().parents[1] / "bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py"
HEAVY = ("jax", "jaxlib", "torch", "gpytorch", "linear_operator", "cupy", "sklearn", "tensorflow")
PROBE = r"""
import json, runpy, sys
path = sys.argv[1]; sys.argv = [path] + sys.argv[2:]
try:
    runpy.run_path(path, run_name='__main__')
except BaseException:
    pass
sys.__stdout__.write('\nMODULES ' + json.dumps(sorted(m for m in {heavy!r} if m in sys.modules)) + '\n')
""".format(heavy=HEAVY)


def _loaded(args, tmp_path):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONPATH=str(CIP.parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, "-c", PROBE, str(CIP)] + args, cwd=tmp_path, env=env,
                         capture_output=True, text=True, timeout=600)
    line = [l for l in out.stdout.splitlines() if l.startswith("MODULES ")]
    assert line, out.stdout[-2000:] + out.stderr[-2000:]
    return set(json.loads(line[-1][len("MODULES "):]))


def test_help_imports_no_heavy_backend(tmp_path):
    pytest.importorskip("lal")
    assert _loaded(["--help"], tmp_path) == set()


def test_default_rf_av_startup_imports_only_sklearn(tmp_path):
    # rf itself needs sklearn (base loads it too); nothing else may load before the data read fails.
    pytest.importorskip("lal")
    args = ["--fname", str(tmp_path / "missing.net"), "--fit-method", "rf", "--sampler-method", "AV",
            "--parameter", "mc", "--parameter", "delta_mc", "--n-output-samples", "10"]
    assert _loaded(args, tmp_path) <= {"sklearn"}
