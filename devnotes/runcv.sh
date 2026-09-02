#!/bin/bash
export PYTHONPATH=$HOME/rift_ghlaplace_20260902/MonteCarloMarginalizeCode/Code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd $HOME/rift_ghlaplace_20260902/devnotes
exec /cvmfs/software.igwn.org/conda/envs/igwn/bin/python "$@"
