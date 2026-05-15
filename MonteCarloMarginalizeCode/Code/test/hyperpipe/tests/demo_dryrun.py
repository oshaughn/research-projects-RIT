"""
End-to-end dry-run demo: drives util_RIFT_hyperpipe.py with a
Gaussian-like config, lets it write all the per-stage args files into
./demo_run/, and prints the resulting create_eos_posterior_pipeline
command line --- WITHOUT actually executing it.

Run via: ``pixi run demo-dryrun``
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
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
    raise SystemExit("RIFT root not found.")


RIFT_ROOT = _rift_root()
RIFT_PY = RIFT_ROOT / "MonteCarloMarginalizeCode" / "Code"
SCRIPT = RIFT_PY / "bin" / "util_RIFT_hyperpipe.py"
GAUSS_EXE = RIFT_PY / "bin" / "util_HyperMargGaussian.py"

run_dir = Path("demo_run").resolve()
base_dir = Path("demo_base").resolve()

for d in (run_dir, base_dir):
    if d.exists():
        shutil.rmtree(d)
base_dir.mkdir()

grid = base_dir / "blind_gaussian_plus_minus.dat"
with open(grid, "w") as f:
    f.write("# lnL sigma_lnL x y z\n")
    for v in (-4, -2, 0, 2, 4):
        f.write(f"0 0 {v} 0 0\n")
        f.write(f"0 0 {v} 1 -1\n")

overrides = [
    f"general.rundir={run_dir}",
    "general.dry-run=true",
    "general.use-osg=true",
    "general.use-singularity=true",
    "general.condor-local-nonworker=true",
    "general.condor-local-nonworker-igwn-prefix=true",
    "general.retries=5",
    "general.request-disk=2G",
    "arch.n-iterations=20",
    "arch.explode-marg-jobs=5",
    f"init.file={grid}",
    'marg-list=[{name:gaussian, exe:' + shlex.quote(str(GAUSS_EXE))
        + ', args:"--outdir Gaussian_example --conforming-output-name", '
        'event-file:null, n-chunk:100, coord-module:null, extra-args:""}]',
]

cmd = [sys.executable, str(SCRIPT), *overrides]
env = dict(os.environ)
env["PYTHONPATH"] = str(RIFT_PY) + os.pathsep + env.get("PYTHONPATH", "")

print("=" * 72)
print("RIFT_ROOT =", RIFT_ROOT)
print("Invoking:", " ".join(cmd))
print("=" * 72)
proc = subprocess.run(cmd, env=env, cwd=base_dir)
if proc.returncode != 0:
    sys.exit(proc.returncode)

print("\n" + "=" * 72)
print("Artefacts written into:", run_dir)
print("=" * 72)
for p in sorted(run_dir.iterdir()):
    if p.is_file() and p.stat().st_size < 4096:
        print(f"\n--- {p.name} ---")
        print(p.read_text().rstrip())
