"""TASK 3 -- laplace + JAX_ILE_DISTMARG_GH against independent references.

Ladder-2 injection (35+30 Msun, H1/L1/V1, SEOBNRv4, l_max=2), rho 40.77 and
163.08, at the sky points the campaign's own posterior occupies.

Three comparisons, all in nats on the SAME data and the SAME dense phi grid:
  * laplace + GH(N)   vs  exact + GH(N)        -- distance treatment held fixed
  * laplace + GH(N)   vs  laplace + uniform-M  -- angle treatment held fixed
  * laplace + GH(N)   vs  laplace + GH(4N)     -- self-convergence in N

The time window is narrowed (--data-integration-window-half) so the CPU cost of
the uniform-M reference is bearable; the node-placement rule under test is
per-(phi, sample, time) and does not depend on how many time bins there are.
"""
import argparse, json, sys
import numpy as np
import jax.numpy as jnp

import probe
from RIFT.likelihood.jax_ile import anglemarg as AM
from RIFT.likelihood.jax_ile import core as core_mod
from RIFT.likelihood.jax_ile.core import make_distance_grid

p = argparse.ArgumentParser()
p.add_argument("--snr", type=int, default=40)
p.add_argument("--nsky", type=int, default=4)
p.add_argument("--iwh", type=float, default=0.005)
p.add_argument("--uniform", type=int, default=4096)
p.add_argument("--gh", type=int, nargs="+", default=[16, 33, 65, 129])
a = p.parse_args()

like, ld, prov, opts, drv = probe.build(
    a.snr, angle_marg="laplace", approximant="SEOBNRv4", l_max=2,
    iwh=a.iwh)
ra, dec, incl = probe.sky_gauss(a.snr, a.nsky)
ra = jnp.asarray(ra); dec = jnp.asarray(dec); incl = jnp.asarray(incl)
x_sup, lw_sup = make_distance_grid(opts.d_min, opts.d_max, 256,
                                   distMpcRef=ld.distMpcRef)
amp = max(float(AM.estimate_angle_amplitude(ld, x_sup, prov["interp"])),
          AM.ANGLE_MARG_CROSSOVER_AMPLITUDE)
nphi_d, nu_d = AM._dense_grid_sizes(amp, m_max=2)
print("CONFIG " + json.dumps(dict(snr=a.snr, npts=int(ld.npts), nsky=a.nsky,
                                  iwh=a.iwh, amp_sizing=amp, nphi_d=nphi_d,
                                  nu_d=nu_d, lms=prov["lms"],
                                  half_sigma=AM._GH_PSI_HALF_SIGMA,
                                  min_nodes=AM._GH_PSI_MIN_NODES)), flush=True)

EX = AM.fused_log_likelihood_distphipsimarg_exact
LP = AM.fused_log_likelihood_distphipsimarg_laplace


def run(fn, xg, lw, gh):
    core_mod._DISTMARG_GH_N = int(gh)
    try:
        return np.asarray(fn(ld, ra, dec, incl, xg, lw,
                             interp=prov["interp"], amp_sizing=amp))
    finally:
        core_mod._DISTMARG_GH_N = 0


res = {}
for g in a.gh:
    res["lap_gh%d" % g] = run(LP, x_sup, lw_sup, g)
    print("  lap_gh%-4d %s  (nodes %d)" % (
        g, np.array2string(res["lap_gh%d" % g], precision=6),
        AM._gh_psi_node_offsets(g)[3]), flush=True)
for g in a.gh:
    res["ex_gh%d" % g] = run(EX, x_sup, lw_sup, g)
    print("  ex_gh%-5d %s" % (g, np.array2string(res["ex_gh%d" % g], precision=6)),
          flush=True)
if a.uniform:
    xu, lwu = make_distance_grid(opts.d_min, opts.d_max, a.uniform,
                                 distMpcRef=ld.distMpcRef)
    res["lap_uni"] = run(LP, xu, lwu, 0)
    print("  lap_uni%-4d %s" % (a.uniform,
                                np.array2string(res["lap_uni"], precision=6)),
          flush=True)

print("== DISAGREEMENT, nats (max over the %d sky points) ==" % a.nsky)
out = {}
for g in a.gh:
    for lbl, ref in (("exact+GH%d" % g, res.get("ex_gh%d" % g)),
                     ("laplace+uniform%d" % a.uniform, res.get("lap_uni"))):
        if ref is None:
            continue
        d = float(np.abs(res["lap_gh%d" % g] - ref).max())
        out["laplace+GH%d vs %s" % (g, lbl)] = d
        print("  laplace+GH%-4d vs %-22s  %.3e" % (g, lbl, d))
for i in range(len(a.gh) - 1):
    d = float(np.abs(res["lap_gh%d" % a.gh[i]]
                     - res["lap_gh%d" % a.gh[-1]]).max())
    out["laplace+GH%d vs laplace+GH%d" % (a.gh[i], a.gh[-1])] = d
    print("  laplace+GH%-4d vs laplace+GH%-11d  %.3e" % (a.gh[i], a.gh[-1], d))
print("VALIDATE " + json.dumps(dict(snr=a.snr, amp=amp, nats=out)))
