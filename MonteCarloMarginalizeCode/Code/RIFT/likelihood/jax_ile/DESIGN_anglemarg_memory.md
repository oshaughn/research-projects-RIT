# JAX angle-marginalization memory model

The evaluation cap in `samplers.py` protects only callers using `eval_lnL*`.
Direct `log_likelihood` calls and scalar value/gradient/Hessian entry points
bypass it, and a fraction of reported device memory does not bound the sum of
live buffers, allocator reservations, or reverse-mode residuals.

Let `S` be batch size, `T=data.npts`, `F` a phi chunk, `D` a distance block,
`Q=16` the Laplace u chunk, `E` the exact dense-angle chunk, `G` the exact
distance block, and `P` the rolled sample-time point block. Float64 and
complex128 occupy 8 and 16 bytes.

## Common storage

For source mode bound `m`, the coefficient tables have shapes
`(m+1,3,S,T)` and `(2m+1,5,S,T)` complex128. Together they contain

```
16 S T [3(m+1) + 5(2m+1)] bytes.
```

At `m=2` this is `544 S T` bytes: 2.42 GiB at `S=4000,T=1193`.
Their angle-sample loop is rolled, but coefficient construction is not yet
tiled over the evaluation sample/time axes.

## Exact

The dense angle grid is scanned in `E=8` chunks and distance in `G=32`
blocks. The dominant exponent slab is `(E S,T,G)` float64, or
`8 E G S T = 2048 S T` bytes (9.10 GiB at `4000 x 1193`). Grid length is
bounded; sample and time still multiply the slab. Exact therefore remains
under the conservative outer cap pending point-axis tiling.

## Laplace

Before this patch the pure-quadrature branch materialized

```
(Q,D,F,S,T) float64 = 8 Q D F S T = 8192 S T bytes
```

at shipped `Q=16,D=4,F=16`. At `S=4000,T=1193` this is 36.41 GiB, the
failed XLA allocation that motivated the cap. It lived alongside coefficient
tables, five phi fields (`64 F S T` bytes), carries, and AD residuals.

Laplace now flattens the independent `(S,T)` axes, edge-pads only the last
tile, and maps distance/psi marginalization over fixed tiles. Its expensive
slab is bounded by

```
8 Q D F min(S T,P),  P=LAPLACE_POINT_BLOCK=4096,
```

or 32 MiB with shipped inner blocks. Padding repeats a finite edge point and
is discarded before the phi reduction. Every real bin retains the same
distance nodes, psi quadrature, per-bin reduction order, phi reduction, and
Simpson time marginalization. The map body is checkpointed for reverse AD.
Coefficient tables and phi fields remain `O(S T)`, so this is a bound on the
measured multiplicative wall, not a claim that total memory is 32 MiB.

## Peak-local

The u-node axis is already streamed with `U_live<=8`, and phi with `F=16`.
The documented node slab per sample-time point is
`8 F N_x 4 U_live` bytes: 1 MiB at `N_x=256`. Nested
`vmap(vmap(_one))` still multiplies it by `S T`. A follow-up should roll those
axes around `_one` and GPU-profile a suitably smaller point tile.

## Validation boundary

Checkpointing the exact/Laplace phi scans and peak-local phi/u scans bounds
saved loop residuals, but does not by itself shrink primal `S*T`
vectorization. Tests inspect the traced Laplace kernel-input shape and compare
tiled versus one-block values and gradients, including a padded tail.

CPU tests cannot establish CUDA allocator peaks, GPU XLA fusion, or the
throughput-optimal `P`. Before relaxing `angle_marg_eval_chunk`, profile all
three schemes on a production CUDA host at `T≈1193`, batches spanning the
current cap and nominal 1000/4000, and exercise value, gradient, and
Fisher/Hessian calls while recording allocator peak statistics.
