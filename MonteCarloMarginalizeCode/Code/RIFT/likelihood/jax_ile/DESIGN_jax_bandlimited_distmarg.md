# Band-limited time quadrature for the distance-marginalized JAX likelihood

RECORD, 2026-09-08. Method and measured numbers for
`time_quadrature="bandlimited"` on `JAXDistanceMarginalizedLikelihood`. The code
carries the tested constants and a pointer here. It carries no numbers.

## What changed

`fused_log_likelihood_distmarg` used to reduce over distance on the data time
grid and hand the reduced `lnL(t)` to `_time_marginalize_terminal`, which
refuses `bandlimited`. Interpolating an already-reduced nonlinear field can
converge to the wrong function. The kernel now gathers the guarded complex
primitive, refines it, and applies the same distance quadrature at every refined
node.

Order of operations under `bandlimited`:

1. `_accumulate_unit(..., guard=g)` with `g = bandlimited_time_guard(npts)[1]`,
   the certified guard.
2. Per row, inside `lax.map`: raised-cosine support pad, then even-extension FFT
   refinement of `kappa(t)` at a curvature-derived power-of-two factor.
3. Crop to the integrated window.
4. `log sum_g exp(K x_g - R x_g^2 / 2 + log w_g)` at every remaining node, in
   blocks sized from the refined row length.
5. Stable trapezoid over the original closed window.

Step 3 comes before step 4. `reduce_fn` is pointwise in the node, so the value
is unchanged, and the distance reduction then runs on `(npts-1)*f+1` nodes
instead of the guard-padded `2*(npts+2g-1)*f`. Reducing first cost 6.50 GB for
one scalar `value_and_grad`; cropping first costs 2.56 GB.

Certificates are three of the fixed-distance four, applied to the
distance-marginalized field: factor doubling to 1e-3 nat, remeasured peak-width
resolution, and agreement between the certified guard and half of it. The
fourth, the 15-nat endpoint gap, is off on this path; the section "The endpoint
certificate" below records why. A row that fails any certificate returns NaN and
the driver refuses the run.

`bandlimited_time_guard` is new and is the single definition of the
`(initial, certified)` guard pair. The fixed-distance kernel, the
distance-marginalized kernel and the driver's storage-window sizing all read it.

The curvature probe moved inside the row-local `lax.map`. It used to be computed
for the whole batch, which for the distance path would have carried an
`(S, 2*n_guarded, block)` temporary.

`JAX_ILE_DISTMARG_GH` is not read by `fused_log_likelihood_distmarg` on either
quadrature. The per-sample Gauss-Hermite placement is implemented only in the
phi/psi-marginalized kernels, which refuse `bandlimited`. Both branches of the
distance-marginalized kernel call one helper, so a future GH branch there lands
on the refined grid as well as the coarse one.

## Scope, and why the other wrappers still refuse

- `JAXDistPhiMargLikelihood` and `fused_log_likelihood_distphimarg`: the phi_ref
  grid sum streams one primitive per grid point through a `lax.scan` carrying
  only `(S, npts)`. Primitive-first requires `(nphi, n_fine)` per row. At the
  shipped `nphi = 32` and the certified factor cap that is about 200 times the
  row-local budget. This is a cost limit, not a correctness gap.
- `JAXDistPsiMargLikelihood`, `JAXDistPhiPsiMargLikelihood`, the exact-angle and
  Laplace schemes: these reach `anglemarg.py` coefficient-table kernels that
  return an already-reduced `lnL(t)`. There is no primitive at that seam to
  refine without an adapter. `time_first_peaklocal.py` prototypes one.

## Measurement setup

Zero-noise synthetic injection through the production `PrecomputeLikelihoodTerms`:
35 + 30 Msun, `IMRPhenomD`, H1L1, `fmin = fref = 40 Hz`, `fmax = 300 Hz`,
`deltaF = 0.25 Hz`, srate 1024 Hz, integration half-window 75 ms (npts 153,
guard 128 initial / 256 certified), distance grid 512 uniform nodes over
[50, 4000] Mpc with the Euclidean prior, evaluated at the injected angles
(1.2, -0.4, 0.7, 0.9, 2.1). Host `ldas-grid`, CPU, float64,
`~/.cache/jaxci_venv/bin/python`.

Amplitude is set by the injected distance. The rho below is the network
optimal SNR of the zero-noise data, sqrt(sum <d|d>) over 40-300 Hz: 390 Mpc
gives rho 45.9 (H1 28.5, L1 36.0) and 48.5 Mpc gives rho 369 (H1 229, L1 289).
`extras["guess_snr"]` reads 19.92 and 160.18 on the same data, 2.305x lower:
it is the precompute's guess sqrt(sum max|Q_lm|^2 / U_lm) / 2.3, quoted as rho
in earlier versions of this record.

## Agreement with an independent reference

The reference reconstructs the primitive with a plain periodic zero-padded FFT,
reduces over distance with a numpy log-sum-exp, and integrates with a numpy
trapezoid. Its extension is periodic where the shipped one is an even
reflection, and its guard taper ramps on `(k+1)/(g+1)` where the shipped one
ramps on `k/g`.

| rho | shipped bandlimited | reference (guard 512, factor 512) | difference (nat) |
|---|---|---|---|
| 19.92 | 589.877456830 | 589.877457058 | -2.3e-07 |
| 160.18 | 39193.075978776 | 39193.075993296 | -1.5e-05 |

Reference convergence, tapered. Successive differences along each ladder:

| ladder | rho 45.9 | rho 369 |
|---|---|---|
| factor 64→128→256→512→1024 at guard 512 | 0, 0, 0, 0 | -5.6e-02, +1.2e-03, +7e-12, 0 |
| guard 128→256→512→1024 at factor 512 | +7.4e-07, +1.8e-07, +4.5e-08 | +4.7e-05, +1.2e-05, +2.9e-06 |

The taper is required. An untapered periodic reconstruction leaves a step at the
periodic seam, and its Gibbs ringing decays like 1/guard:

| untapered reference, factor 128 | rho 45.9 | rho 369 |
|---|---|---|
| guard 128 | 589.929584 | 39196.403097 |
| guard 256 | 589.897169 | 39194.333390 |
| guard 512 | 589.883333 | 39193.449964 |
| guard 1024 | 589.878738 | 39193.156559 |

Each doubling halves the residual instead of removing it. At guard 1024 the
untapered reference is still 1.3e-03 nat (rho 46) and 8.1e-02 nat (rho 369) from
the shipped value, so it certifies nothing at the tolerance this work asserts.

The shipped value is not sitting on its own stopping tolerance. Forcing
`_TIME_ADAPTIVE_RTOL` to 1e-4 selects the same factor and returns the same
number, 39193.075978776. At 1e-5 and 1e-6 the row returns NaN, because the
doubling cannot be met within `_TIME_ADAPTIVE_FACTOR_MAX`.

## The option changes the answer

| rho | native Simpson | bandlimited | gap (nat) |
|---|---|---|---|
| 19.92 | 541.265029 | 589.877457 | 48.6 |
| 160.18 | 35923.702300 | 39193.075979 | 3269.4 |

Reduce-then-refine, built explicitly in the test with the same numpy pieces,
lands 3.31 nat (rho 46) and 99.35 nat (rho 369) from the shipped value.

Mutating `at_factor` to reduce on the coarse grid and refine the reduced field
makes the rho 46 agreement test fail by -3.366 nat, about 3400 times its
tolerance, and makes the rho 369 row return NaN. The resolution and doubling
certificates reject the wrong-order field on their own.

## The endpoint certificate

The fixed-distance kernel refuses a row whose refined endpoint is within 15 nat
of its peak. On the distance-marginalized field that rule rejects rows whose
integral is converged, and it rejects most blind draws.

The reason is a floor. At every node the distance sum is at least the
far-distance prior mass, so the field never falls below about the value it takes
where the template is orthogonal to the data. A row's peak-to-endpoint contrast
is therefore bounded by its own peak height, and a fixed 15-nat gap cannot be met
by any row with a peak under about 15 nat, however well the trapezoid has
converged. The fixed-distance field has no floor: it falls to -rho^2/2 away from
the peak, so the same gap measures something there.

Blind full-sky, isotropic-orientation draws are exactly the low-contrast rows.
Every prior-seeded driver mode evaluates them by the thousand (`--mode map` and
`nuts` pilot on 4000, `laplace-is` on `n_max/4`, `prior-mc` on `n_max`), and the
driver stops the run on one NaN.

Measured, same injection as above at 900 Mpc (rho 19.9), 20 ms half-window,
seeds 0-3 of the driver's prior, 64 rows each:

| certificate set | uncertified rows |
|---|---|
| all four (gap on) | 90 / 256 |
| gap off | 36 / 256 |
| fixed-distance 6-D kernel, blind distance, gap on (unchanged) | 92 / 256 |

Every one of the 90 failed the endpoint gap and nothing else. Twenty-four of
them, six per seed, against the independent reference (periodic FFT, guard 128,
factor 512): worst disagreement 1.06e-04 nat, and all of them 130-180 nat below
the batch maximum.

The 36 that remain with the gap off all have a detector arrival peak at or beyond
the window edge. A wrong sky moves a detector's arrival by up to 2 R_earth / c,
about 43 ms, past a 20 ms half-window. Two signatures: the trapezoid value drops
by ln 2 per doubling (the integrand is an edge sliver narrower than a sample), or
the two guards disagree at 1e-3 to 1e-2 nat with the doubling converged to 1e-14
(structure under the taper, outside the window). Neither is a certificate the
path should drop; the window has to contain the shifts.

With the window widened to contain the shifts, same seeds and rows, gap off:

| half-window | gap on | gap off | 6-D kernel, gap on |
|---|---|---|---|
| 20 ms | 90 / 256 | 36 / 256 | 92 / 256 |
| 50 ms | 14 / 256 | 0 / 256 | 32 / 256 |
| 75 ms | 14 / 256 | 0 / 256 | 30 / 256 |

The 14 rows the gap alone rejects at 50 and 75 ms agree with the reference to
1.0e-06 and 3.8e-07 nat at worst, and sit 137-169 nat below the batch maximum.

The 6-D kernel's column is base-branch behaviour and is unchanged by this
branch. Its blind-draw stop is a separate follow-up.

The driver's stop is unchanged. Its message now counts the failed rows in the
chunk, prints the first three, and names the window as the usual cause.

Driver runs on the merged tree, same injection, 50 ms half-window, 32 distance
nodes, seed 3, `--n-max 400` unless stated:

| mode | quadrature | result |
|---|---|---|
| prior-mc | bandlimited | row written, lnZ 167.4, neff 2.0, 11 s |
| map | bandlimited | row written, peak lnL 186.3, 47 s |
| laplace-is, n_max 4000 | bandlimited | refused: adapted lnZ 165.3 is 6.3 nat below the prior pilot's Markov floor, neff 6.1 |
| laplace-is, n_max 4000 | simpson | row written, lnZ 166.5, neff 16.4 |

The laplace-is refusal is the estimator's own gate, reached after every
evaluation certified: the resolved peak is narrower than one moment-matched
Gaussian covers. It is not a certificate failure. The default mode therefore
needs a larger budget or another mode under `bandlimited` on a narrow posterior.

The injected angles are not where this likelihood peaks. On the 900 Mpc
injection the fixed-distance lnL at the injected angles is 84.0 in the
conventional code and 84.3 in JAX. The 2-D (psi, phiref) maximum is 194.9 in
both codes, so the offset is a property of the fixture and not of the
quadrature. The fixture hands the precompute the same P as the injection. For
IMRPhenomD the template route (`SimInspiralTDModesFromPolarizations`) bakes
that P's phiref and psi into the (2,2) mode as exp(-2i phiref) exp(+4i psi)
(measured, lalsimulation 6.2.0). ILE then applies both again through Y_lm and
F. Production drivers zero P.phiref and P.psi before the precompute.
`lalsimutils` is unchanged.

### The fixed-distance kernel (follow-up, 2026-09-08)

The 6-D kernel kept the gap when the rows above were measured. Re-measured on
rift_O4d `d84597c2a` (92 / 256 at 20 ms, unchanged by this branch) and on this
branch with the gap switched per row: same injection, seeds 0-3 of the driver's
prior with distance, 64 rows each, ldas-grid, `~/.cache/jaxci_venv`.

| half-window | gap on | gap off | gap alone |
|---|---|---|---|
| 20 ms | 92 / 256 | 35 / 256 | 57 |
| 50 ms | 32 / 256 | 0 / 256 | 32 |

Twenty-four gap-only rows per window against the reference (periodic FFT,
guard twice the certified value, factor 512): worst 1.1e-04 nat at 20 ms and
6.1e-06 nat at 50 ms, all 54-128 nat below the batch maximum, at 1600-3900 Mpc.

The floor argument has a fixed-distance twin. The field is Re kappa(t) -
rho^2/2, so its contrast is at most 2 max|kappa|, which scales with the row's
own amplitude. A far or wrong-sky draw has no 15 nat to give up, converged or
not. The gap certifies the row's amplitude, not the quadrature.

Decision: the endpoint certificate is off on both fields. The kernel's
`endpoint_log_gap` defaults to `None`; the threshold constant stays for the
tests that pin what it rejected. The other three certificates are unchanged,
and the 35 edge-peak rows at 20 ms still stop the driver.

The driver refuses `--mode prior-mc`, `laplace-is`, `map` and `nuts` with
`bandlimited` at parse time when `--data-integration-window-half` is below
2 R_earth / c = 42.6 ms. Those modes push full-sky draws through `eval_lnL`,
which stops on one uncertified row; the flowMC family and `multistart-nuts`
pilot through the samplers' own draw and are not refused. Before the change,
`--mode prior-mc` without distance marginalization at 50 ms stopped in its
first chunk with 30 of 400 rows failed, every one at the gap.

Test: `test/jax/test_jax_bandlimited_6d_blind.py`, ten tests. The end-to-end
runs cover `prior-mc` and `map`; `laplace-is` at this injection walks off the
peak and fails `require_finite_evidence` at pilots of 100, 250 and 500, on the
distance-marginalized field of the unchanged base as well, so it is covered
by the parse-time test only.

## Memory

Peak RSS, one JAX process, same data. `vmap 8` is eight chains of
`value_and_grad`, which is the shape flowMC's MALA proposal builds.

| path | scalar | vmap 8 |
|---|---|---|
| fixed-distance 6-D bandlimited (shipped) | 1.15 GB | 3.25 GB |
| distance-marginalized, reduce before crop | 6.50 GB | - |
| distance-marginalized, crop before reduce | 2.56 GB | 9.96 GB |

The remaining factor over the fixed-distance path is the distance grid itself.
`_BANDLIMITED_GRID_ELEMENTS` sits near the minimum of a two-sided trade-off, so
lowering it makes things worse. At `1<<22` the `vmap 8` peak is 9.97 GB; at
`1<<18` it is 22.77 GB, because a smaller block multiplies the scan carries the
reverse pass retains, and smaller values exceed the 25 GiB per-user cgroup.

Driver runs of `--mode flowmc --distance-marginalization`, integration
half-window 20 ms, 32 distance nodes, one training and one production loop:

| steps (local = global) | quadrature | peak RSS | wall | result row |
|---|---|---|---|---|
| 20 | simpson | 1.43 GB | 0:21 | yes |
| 20 | bandlimited | 23.4 GB | 4:50 | yes |
| 4 | bandlimited | 6.36 GB | 2:03 | yes |
| 2 | bandlimited | 4.46 GB | 2:09 | yes |

flowMC unrolls its per-step proposal, so the compiled graph is multiplied by the
step count. The eleven refinement branches make that graph large, and 20 steps
exceeds the 25 GiB cgroup on `ldas-grid`. The gated driver test uses 2 steps and
brackets the peak with `--n-prior-pilot` instead.

## Files

- `core.py`: `bandlimited_time_guard`, `_time_marginalize_reflected_primitive`
  (`reduce_fn`, row-local probe, crop before reduce),
  `_logsumexp_grid_scanned`, `fused_log_likelihood_distmarg`.
- `wrapper.py`: `JAXDistanceMarginalizedLikelihood` accepts the option and
  publishes `time_guard_initial` and `time_guard_certified`.
- `bin/integrate_likelihood_extrinsic_jax`: `eval_lnL` failure message.
- `test/jax/test_jax_bandlimited_distmarg.py`: 20 tests, about 3.5 min on `ldas-grid` (the flowMC driver run is 100 s of it; CI deselects that one).
