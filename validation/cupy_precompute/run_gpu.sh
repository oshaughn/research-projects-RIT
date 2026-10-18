#!/bin/bash
set -euo pipefail
root=/scratch/richard.oshaughnessy/o4c_fullrate_cupy_review_20261018
band=$1
case "$band" in
 ampere) sif=/scratch/richard.oshaughnessy/containers/rift_o4d_cc60-90_cuda118_20260717.sif ;;
 blackwell) sif=/scratch/richard.oshaughnessy/containers/rift_o4d_cc90-120_cuda128_20260717.sif ;;
 *) exit 2 ;;
esac
cd "$root/$band"
mkdir -p cupy_cache
export APPTAINERENV_PYTHONPATH="$root/code:/scratch/richard.oshaughnessy/early_time_compression/xpytest/pylib"
export APPTAINERENV_GW_SURROGATE=''
export APPTAINERENV_OMP_NUM_THREADS=1 APPTAINERENV_OPENBLAS_NUM_THREADS=1 APPTAINERENV_MKL_NUM_THREADS=1
export APPTAINERENV_CUPY_CACHE_DIR="$root/$band/cupy_cache"
export APPTAINERENV_RIFT_CI_REQUIRE_GPU=1
nvidia-smi > nvidia-smi.txt
apptainer exec --nv --bind /scratch "$sif" python -m pytest -v -rs -p no:cacheprovider \
 "$root/code/test/test_precompute_array.py" "$root/code/test/test_o4c_critical_ports.py" \
 "$root/code/test/test_fd_end_taper_cap.py" "$root/code/test/test_noloop_time_interp.py" \
 "$root/code/test/test_noloop_bandlimited_row_chunks.py" > pytest.txt 2>&1
apptainer exec --nv --bind /scratch "$sif" python "$root/benchmark.py" \
 --output "$root/$band/benchmark.json" > benchmark.log 2>&1
apptainer exec --nv --bind /scratch "$sif" python "$root/cli_check.py" "$root/code" \
 /scratch/richard.oshaughnessy/early_time_compression/a4 "$root/$band/ile" > cli.log 2>&1
touch PASS
