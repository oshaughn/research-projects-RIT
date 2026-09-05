# JAX angle-marginalization memory model

These are logical array-size and lifetime models for the JAX-only
angle-marginalization kernels. Except for the historical XLA allocation request
identified below, they are not measurements of CUDA allocator peak memory.
They must not be read as the footprint of conventional production ILE.

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
tiled over the evaluation sample/time axes. These tables persist across the
phi scan; the quoted number is their logical payload, not an allocator peak.

## Exact

The dense angle grid is scanned in `E=8` chunks and distance in `G=32`
blocks. The dominant exponent slab is `(E S,T,G)` float64, or
`8 E G S T = 2048 S T` bytes (9.10 GiB at `4000 x 1193`). Grid length is
bounded; sample and time still multiply the slab. Exact therefore remains
under the conservative outer cap pending point-axis tiling.

## Laplace

Before this patch, one step of the u scan formed the logical f64 result
`blk` with shape

```
(Q,D,F,S,T) float64 = 8 Q D F S T = 8192 S T bytes
```

at shipped `Q=16,D=4,F=16`. It is reduced over `Q` immediately; the distance
and phi scans do not keep all of their blocks simultaneously. The complex128
products used to form `blk` have the same shape but are eligible for compiler
fusion. At `S=4000,T=1193`, the f64 `blk` alone is 36.407 GiB. Commit
`c5b81dd6` records that XLA requested this single allocation during a pre-cap
SNR-40 JAX acceptance run against a 25-GiB cgroup. This investigation does not
have the original allocator log, did not reproduce that run, and did not
measure a 36-GiB current-production footprint.

The other source-visible live values include the persistent coefficient tables,
five phi fields (`64 F S T` bytes: A0/B0 real, A1/B1/B2 complex), distance-scan
carries and, for differentiated calls, residuals selected by XLA/AD. Their
simultaneous physical lifetime cannot be obtained by summing source-level
shapes and requires an allocator profile.

Laplace now flattens the independent `(S,T)` axes, edge-pads only the last
tile, and maps distance/psi marginalization over fixed tiles. Its expensive
slab is bounded by

```
8 Q D F min(S T,P),  P=LAPLACE_POINT_BLOCK=4096,
```

or 32 MiB with shipped inner blocks for a direct call whose only batched axes
are the explicit `S,T` axes. Padding repeats a finite edge point and is discarded
before the phi reduction. Every real bin retains the same distance nodes, psi
quadrature, per-bin reduction order, phi reduction, and Simpson time
marginalization. The map body is checkpointed for reverse AD. Coefficient tables
and phi fields remain `O(S T)`, so this is neither a claim that total memory is
32 MiB nor a bound on an arbitrary transformed caller.

In particular, `flowMC` applies an outer `vmap` over its chains to the scalar AD
target. The scalar wrapper has explicit `S=1`, so its `pblk` calculation cannot
see that mapped chain axis. For the usual 20-chain driver call at `T=1193`, the
corresponding logical primal slab is at most about 186 MiB before accounting for
AD residuals, not 36.41 GiB, but it is also not covered by the 32-MiB statement.

## Production call paths

Conventional `integrate_likelihood_extrinsic_batchmode` does not call this JAX
kernel. Its maintained GPU NoLoop path samples distance, phi and psi and carries
primarily `(S,T)` arrays (`kappa_sq` complex128 and `rho_sq` float64); it has no
`Q*D*F` angle-quadrature multiplier. Operation on 4-GB cards therefore does not
contradict the JAX shape above.

The separate `integrate_likelihood_extrinsic_jax` reaches this kernel only for
the distance+phi+psi-marginalized mode with a resolved Laplace scheme. Its host
pilot/reweight evaluations call `angle_marg_eval_chunk`; the sampler helpers do
the same. At `T=1193`, the 4-GiB fallback target caps the old model at `S=439`,
so the current production call path does not submit `S=4000`. Scalar
value/gradient/Hessian calls use explicit `S=1`; flowMC normally maps those over
20 chains.

There is nevertheless a real weakness in the current heuristic: on a GPU whose
total reported limit is 4 GiB, `_angle_marg_buffer_target()` still returns its
4-GiB floor, and the resulting `S=439` cap budgets 3.996 GiB for this one old
slab alone. That is not a defensible total-memory bound. It is a theoretical
finding here, not a measured 4-GB JAX failure; direct `log_likelihood` calls also
bypass the cap altogether.

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
Fisher/Hessian calls while recording allocator peak statistics. Profile the
flowMC outer-vmap path separately: explicit point tiling does not bound that
hidden chain axis.
