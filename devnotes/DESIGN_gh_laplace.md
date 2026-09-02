# psi-marginal distance-node placement for `laplace` + `JAX_ILE_DISTMARG_GH`

Branch `ghlaplace_o4d`, off `524331989ba7c135f2226ef4f33916684792c5a4`.
All numbers here are measured; the code carries the constants and a pointer.

## The problem

`core._distmarg_gh_logL` places frozen distance nodes at
`clip(K/R, x_min, x_max) +- 7/sqrt(R)` for a FIXED psi.  On the `laplace` path
psi has already been integrated out analytically, so `K` and `R` do not exist
pointwise: the nodes must bracket the psi-MARGINAL distance integrand

    I(x) = (1/pi) int dpsi exp(x A(u) - x^2/2 B(u)),   u = 2 psi
    A(u) = A0 + Re(A1 e^{iu}),  B(u) = B0 + Re(B1 e^{iu}) + Re(B2 e^{2iu})

which is a MIXTURE over u of Gaussians of centre `x*(u) = A(u)/B(u)` and width
`1/sqrt(B(u))`.  The function used to `raise` rather than guess.

## The rule that ships

    w      = B0*A1 - conj(A1)*B2
    e^{iu*}= +- conj(w)/|w|      (sign chosen so A(u*) > 0)
    centre = stop_gradient(clip(A(u*)/B(u*), x_min, x_max))
    sigma  = stop_gradient(min(1/sqrt(max(R_lo, 1e-30)), (x_max-x_min)/44))
    R_lo   = B0 - |B1| - |B2|   (== min_u B; see the identity below)
    nodes  = clip(centre + sigma * z, x_min, x_max),  z = linspace(-22, +22, n)

with `n = max(49, 1 + ceil((N-1) * 22/7))` for `JAX_ILE_DISTMARG_GH = N`, so the
node DENSITY the caller asked for at +-7 sigma is preserved, not diluted.
Trapezoid weights `0.5*(x[k+1]-x[k-1])` with the index clamped at both ends --
algebraically identical to `_distmarg_gh_logL`'s `diff`/`concatenate` form, but
computable one block at a time so the distance axis stays scanned.
Gated on `m_max <= 2`; richer mode content still raises.

`u*` is the EXACT maximiser of `A(u)^2/(2 B(u))`, not of `A(u)`.  With
`A0 = B1 = 0` the stationary condition `2 A' B = A B'` reduces in `z = e^{iu}`
to `z^2 w = conj(w)`, so the maximiser is closed form and angle-free, and
reduces to `conj(A1)/|A1|` when `B2 = 0`.  Verified against a 400,001-point
brute-force argmax on 20,000 random `(A1, B0, B2)` with `|B2|/B0` up to
0.99999: brute force never beats it by more than 6.5e-16 relative.

## Task 1: why the closed form is enough (m_max = 2 only)

Ladder-2 injection (35+30 Msun, H1/L1/V1, SEOBNRv4, `--l-max 2`), sky points
drawn from the measured rho-40.77 whole-sky AV posterior (the peer session's
draw, reproduced verbatim), 24 dense-phi x 16 sky x 614 time = 235,776 bins per
rung, psi ranked by the CLIPPED exponent `x_c A - x_c^2 B/2`,
`x_c = clip(A/B, x_min, x_max)`, u grid 721 points (du = 8.71e-3 rad).

| quantity | rho 40.77 | rho 163.08 | pre-registered cut | branch taken |
|---|---|---|---|---|
| `W = sqrt(min_u B / R_lo)` | med 1.0000 max 1.0000 | med 1.0000 max 1.0000 | `<= 1.25` -> closed form as-is | closed form as-is |
| `C = |x*(u_cf)-x*(u_exact)|/sigma` | med 0.0014 p99 0.6361 max 0.6886 | med 0.0090 p99 0.0968 max 0.1117 | `<= 1` -> `u* = arg(A1)` adequate | closed-form centring |
| `S` (psi span, sigma) | med 2.607 p99 3.899 max 4.085 | med 0.641 p99 0.863 max 0.887 | half-width `(7+ceil(S_p99))` | 11 sigma, budget 12 |
| `R_lo <= 0` | 0.0000% of ALL 235,776 bins | 0.0000% | HARD REJECT if any | not triggered |

(W, C, S over the bins within 100 nats of the best bin's clipped exponent; the
`R_lo <= 0` fraction over every bin, as the cut demands.  Per-sample and global
weight masks give the same numbers to 3 dp.)

Directly measured operational quantity -- the one-sided reach from the
closed-form centre to the furthest weight-carrying component centre, which is
what the half-width must actually cover:

| rung | reach p99 | reach max | needed half-width `7 + reach` |
|---|---|---|---|
| 40.77 | 3.280 | 3.461 | 10.46 sigma |
| 163.08 | 0.762 | 0.772 | 7.77 sigma |
| 652.31 (spot check, 8 sky x 32 phi, nu 8192) | 4.510 | 4.510 | 11.51 sigma |

so 12 sigma covers all three rungs with margin, and the pre-registered
`(7 + ceil(S_p99)) = 11` plus `C_p99 = 0.64` gives the same 12.

**Control reproduction.**  The peer session's committed record
(`analyses/va_rebuild_20260902/records/angle_coeff_structure.json`, paper repo
branch `claude/elated-merkle-c4dda4`) reports, from an INDEPENDENT harness that
goes through `anglemarg._reconstruct_field`, W inflation median 1.0000157 max
1.0000610 at rho 40.77 and `frac_R_lo_nonpositive` 0.0 at both rungs.  Its
`W - 1` is pure u-grid discretization: its `bound_tightness_rel_median` is
8.53e-05, exactly `(1/2)|B''|(du/2)^2 / min B` at its du = 2pi/361.  This
harness at du = 2pi/721 gives W - 1 = 4e-6 and at du = 2pi/16384 gives 1e-8 --
the same identity seen at three resolutions.  C and S are NOT in that record
(the file the coordinator named, `scripts/bracket_stats.py`, does not exist on
that branch or any other in the paper repo); the C/S values relayed
(med 0.00151 / p99 0.63555; med 2.607 / p99 3.899 / max 4.085) are reproduced
here to 3-4 significant figures, so whatever produced them agrees with this.

## Why it is m_max = 2 ONLY

`W == 1` is an IDENTITY, not a lucky bound.  The spin-2 antenna response makes
A0 and B1 vanish identically for (2,+-2) content -- measured
`|A0|max/|A1|max = 6.7e-17`, `|B1|max/|B0|max = 5.6e-16` at rho 40.77
(5.99e-17 / 5.53e-16 at rho 163.08), and independently 4.4e-17 / 4.5e-16 on the
synthetic fixture with random U/V, so it is algebra, not a property of this
injection.  Then `B(u) = B0 + Re(B2 e^{2iu})` and `R_lo = B0 - |B2|` IS
`min_u B`, and `A(u)` is a pure first harmonic whose maximiser is available in
closed form.  Neither statement survives odd-m or l >= 3 content, which is why
the ship gate keys on `m_max`, following `angle_sample_grid_sizes`'s precedent.
The higher-mode verdict is owned by another session; the gate stays until it
lands, whatever it says.

## Beyond the fixture: an exhaustive scan of the reachable family

Because `A0 == 0` and `B1 == 0` hold for EVERY mode set, after scaling
`B0 -> 1`, `sigma0 = 1/sqrt(B0)` and rotating `arg(A1) -> 0` the bracket problem
depends on exactly three numbers:

    rho = |A1|/sqrt(B0)     r = |B2|/B0 in [0,1)     delta = relative phase

and `rho` enters only as an overall factor of the exponent, so the whole family
can be SCANNED rather than sampled.  222,950 points (r to 0.9999, rho 0.2 to
3000, 91 phases, v grid 262,144, 100-nat weight threshold), of which 214,849
have >= 32 samples inside the weight-carrying window; the rest are grid-limited
and excluded.  Clipping into `[x_min, x_max]` is 1-Lipschitz, so these unclipped
reaches are UPPER bounds on the clipped ones.

| centring | reach p50 | reach p99 | reach MAX |
|---|---|---|---|
| `argmax A(u)` (the naive closed form) | 6.02 | 3136.7 | 10417.2 |
| `argmax A(u)^2/(2B(u))` (shipped) | 2.79 | 12.80 | **14.134** |

The naive centring is catastrophic at large `|B2|/B0` -- and `|B2|/B0` is a
property of the network response, not of the mode content, so no `m_max` gate
would have caught it.  The ladder-2 fixture never reaches that corner
(`|B2|/B0` median 0.011, p99 0.023 at rho 40.77 and 163.08), which is precisely
why a fixture-only validation would have shipped the wrong rule: this is the
"exact on a fixture, loose in production" defect class in its usual form.

With the shipped centring the reach is bounded across every peak-exponent
decade:

| peak exponent | rows | reach p99 | reach MAX |
|---|---|---|---|
| [0, 30) | 74,092 | 7.24 | 7.74 |
| [30, 100) | 13,463 | 13.86 | 14.11 |
| [100, 300) | 12,503 | 13.80 | 14.13 |
| [300, 1e3) | 13,727 | 12.05 | 13.82 |
| [1e3, 1e4) | 26,402 | 11.20 | 13.50 |
| [1e4, 1e5) | 26,282 | 10.62 | 13.06 |
| >= 1e5 | 48,380 | 10.11 | 12.65 |

The maximum saturates at `sqrt(2T) = sqrt(200) = 14.142` -- attained in the
weak-signal corner, where the 100-nat window is the whole circle.  Hence the
shipped half-span `7 + 14.14 -> 22 sigma`, and the node floor
`2*22/0.92 + 1 = 49`.  The pre-registered `(7 + ceil(S_p99)) = 11` plus
`C_p99 = 0.64` gives 12 sigma, which the ladder-2 operating point needs; 22 is
the family-wide requirement and is what ships.  Widening is the conservative
direction, and it costs nothing at the operating point: the answer at rho 40.77
is bit-identical between 27 nodes at 12 sigma and 49 nodes at 22 sigma to the
six decimals printed.

## Task 3: validation, in nats

Ladder-2, `--data-integration-window-half 0.005` (npts 40), 4 sky points from
the same posterior draw, dense phi grid from `estimate_angle_amplitude` (the
production route), max over sky points:

| comparison | rho 40.77 | rho 163.08 |
|---|---|---|
| laplace+GH16 vs exact+GH16 | 9.196e-05 | see log |
| laplace+GH65 vs exact+GH65 | 9.196e-05 | see log |
| laplace+GH65 vs laplace+uniform-4096 | 3.662e-04 | see log |
| laplace+GH16 vs laplace+GH129 (self-convergence) | 2.76e-09 | see log |
| laplace+GH33/65 vs laplace+GH129 | 0.0 | see log |

The laplace-vs-exact residual is flat in node count, so it is the psi-Laplace
error alone, not the distance quadrature.  The uniform-4096 residual is that
grid's own discretization (at rho 163 a 4096-point uniform grid over
[1, 10000] Mpc puts under one point across the distance peak, so it is NOT a
converged reference there -- reported because it was asked for, not as truth).
The placement is converged at the 27-node floor.
