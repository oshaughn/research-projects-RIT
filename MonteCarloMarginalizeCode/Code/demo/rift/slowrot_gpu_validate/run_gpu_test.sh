#!/bin/bash
# run_gpu_test.sh -- validate the GPU (xpy=cupy) rotation likelihood on a real GPU.
# Runs inside the RIFT singularity container (has cupy/CUDA); the modified rift_slowrot code
# is transferred as rift_code.tar.gz (the production image predates this branch).
set -e
PYTHON="${PYTHON:-python3}"
if [ -f rift_code.tar.gz ]; then
  mkdir -p _code && tar xzf rift_code.tar.gz -C _code
  export RIFT_CODE="$PWD/_code/Code"
fi
: "${RIFT_CODE:?set RIFT_CODE or provide rift_code.tar.gz}"
export PYTHONPATH="$RIFT_CODE:$PYTHONPATH"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$PWD}"

echo "=== node / GPU / cupy ==="
hostname
$PYTHON - <<'PY' || echo "!!! cupy import FAILED (no GPU visible or CUDA/container mismatch)"
import cupy
p = cupy.cuda.runtime.getDeviceProperties(0)
print("cupy", cupy.__version__, "| device", p['name'].decode(),
      "| ccap %d.%d" % (p['major'], p['minor']),
      "| CUDA runtime", cupy.cuda.runtime.runtimeGetVersion())
PY

echo "=== GPU<->CPU rotation consistency (Path A/B, the actual validation) ==="
$PYTHON "$RIFT_CODE/RIFT/likelihood/test_slowrot_gpu.py"

echo "=== GPU<->CPU freqresponse consistency (Path D finite-size, same fused-kernel port) ==="
$PYTHON "$RIFT_CODE/RIFT/likelihood/test_slowrot_freqresponse_gpu.py"

echo "=== CPU rotation sanity (baseline, must still pass) ==="
$PYTHON "$RIFT_CODE/RIFT/likelihood/test_slowrot_noloop.py" 2>&1 | grep -E 'PASSED|Assert|\(B\)'
echo "=== done ==="
