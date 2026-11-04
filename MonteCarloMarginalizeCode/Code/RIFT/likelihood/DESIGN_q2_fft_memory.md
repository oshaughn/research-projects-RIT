# Bounded GPU FFT row batches

AV accepts a varying number of independent extrinsic rows. Passing every such
count directly to CuPy FFT creates many plan shapes and allocator size classes.
On CUDA 11.8/cuFFT 10.9, varying complex128 batches retained device memory even
after clearing the FFT cache and memory pool; disabling the cache did not help.

`_gpu_fft_rows` fixes the batch capacity for each time length. It uses a power of
two, at most 64 rows, with an 8 MiB padded complex128 array budget (or one row for
longer transforms). Larger groups are tiled. Assembled outputs also allocate a
power-of-two row count to avoid one pool size class per AV acceptance count.
Only real rows are returned; zero dummy rows are never passed to the likelihood.

This changes resource scheduling. The time axis, transform lengths, reflection,
Nyquist weights, dtype, integration nodes, refinement factors, quadrature, priors,
sampler settings, row order and all actual rows are preserved. NumPy FFT routing
is unchanged. Each branch retains its existing conditional-time-export semantics.

Only an explicit `_RetainedFFTUnsupported` decline permits the full-padding
reference retry. GPU/resource failures and unexpected transform failures
propagate. The ILE handler recognizes allocation errors by exception ancestry
as well as CUDA/cuFFT/library messages, and raises before empty-product creation
or legacy soft-success options. Retrying a larger FFT after a damaged context or
resource failure can crash the worker; silently skipping an intrinsic point is
also unacceptable.

## Validation

Run the ordinary numerical and error-propagation checks in a supported RIFT
runtime:

```sh
python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_q2_fft_memory.py
```

On one assigned GPU, require the real CuPy lane (no CPU fallback):

```sh
RIFT_TEST_GPU=1 python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_q2_fft_memory.py
python MonteCarloMarginalizeCode/Code/test/validate_q2_fft_lifetime.py
```

The tests compare deterministic complex rows, retained and full-padding
interpolants, marginalized integrals, conditional draws and input knot recovery.
They cover partial tiles, strided inputs, long transforms and the actual CLI
exception handler. A nearly flat inverse CDF permits a 10 ns draw discrepancy
against a 244 us physical knot; integral and draw-density differences retain the
2e-10-nat bound.

The lifetime script uses a fresh process, warms the shape family, then measures
six 80-call cycles with 64 through 143 rows of 819 complex128 samples and factor 8. It
requires zero allocator/device growth after the first measured cycle and zero
residual growth after final cleanup, and writes `lifetime-summary.json`. Keep
the assigned device otherwise stable while measuring. Full-worker tests must
also verify every requested intrinsic point produced a nonempty finite result.
Component tests alone do not qualify a production release or establish
whole-run throughput. Published release qualification must use the new release
without development source overlays.
