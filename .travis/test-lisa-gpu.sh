#!/usr/bin/env bash
# LISA gate on a cupy-capable host.
#
# The LISA likelihood is host numpy, but it calls helpers in
# RIFT/likelihood/SphericalHarmonics_gpu.py whose backend used to default to cupy
# on any host where cupy merely IMPORTS -- no GPU work intended or wanted.  That
# combination raised `TypeError: Unsupported type <class 'numpy.ndarray'>` on
# every GPU host and passed on every cupy-free one, which is why neither the
# GitHub `lisa-check` job nor the GitLab `import_check` job (both cupy-free) ever
# saw it.
#
# So this script REFUSES to run without cupy rather than degrading to the CPU
# backend: a green LISA suite that silently only exercised numpy is the exact
# hole being closed.  Run it on the GitLab `gpu` runner (see gpu_integration in
# .gitlab-ci.yml), or by hand on a GPU node.
set -euo pipefail
# Same interpreter resolution as .travis/test-lisa.sh, so the two LISA gates
# cannot silently run under different pythons.
PY="${RIFT_LISA_PYTHON:-${PYTHON:-python}}"
command -v "$PY" >/dev/null 2>&1 || PY="$(command -v python3)"
export OMP_NUM_THREADS=1

"$PY" - <<'PY'
try:
    import cupy
except Exception as exc:
    raise SystemExit(f"test-lisa-gpu.sh requires cupy, which did not import: {exc}") from exc

try:
    n_devices = cupy.cuda.runtime.getDeviceCount()
except Exception as exc:
    raise SystemExit(f"test-lisa-gpu.sh requires a CUDA device; cupy could not query one: {exc}") from exc
if n_devices < 1:
    raise SystemExit("test-lisa-gpu.sh requires a CUDA device; cupy reported zero")

from RIFT.likelihood import SphericalHarmonics_gpu as sh

# Assert this host is genuinely in the configuration that used to break.  Without
# this the suite would report PASS on a cupy-free runner and prove nothing.
if not sh.cupy_here:
    raise SystemExit(
        "cupy imports, but RIFT.likelihood.SphericalHarmonics_gpu did not enable it "
        "(cupy_here=False) -- the GPU dispatch path is not under test"
    )
if sh.xpy_default is not cupy:
    raise SystemExit(
        f"expected SphericalHarmonics_gpu.xpy_default to be cupy on this host, got {sh.xpy_default!r}"
    )

print(f"LISA GPU preflight OK: cupy={cupy.__version__}, cuda_devices={n_devices}")
PY

"$PY" -m pytest -q \
  MonteCarloMarginalizeCode/Code/test/test_lisa_operational_synthetic.py \
  MonteCarloMarginalizeCode/Code/test/test_spherical_harmonics_backend.py

echo "LISA GPU-host gate: PASS"
