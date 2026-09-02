export SNAP=$HOME/rift_ghlaplace_20260902/MonteCarloMarginalizeCode/Code
export PYTHONPATH=$SNAP
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export PY=$HOME/.conda/envs/rift_jax/bin/python
