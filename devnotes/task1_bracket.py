"""TASK 1 -- can a FROZEN psi-marginal distance bracket be placed for 'laplace'?

Measures, on the ladder-2 injection, the pre-registered quantities that decide
whether 'laplace' can use the adaptive distance quadrature JAX_ILE_DISTMARG_GH:

  W = sqrt(min_u B / R_lo)        width inflation of the closed-form ENVELOPE
                                  R_lo = B0-|B1|-|B2|  vs the exact min_u B
  C = |x_c(u_cf) - x_c(u*)|/sig   centre error of the closed-form rule
                                  u_cf = argmax_u A(u) = -arg(A1)
  S = span of x_c over weight-carrying psi, in sigma
  reach_<rule> = max over weight-carrying psi of |x_c(u) - centre_rule| / sigma
                 -- the OPERATIONAL number: half-width = (7 + ceil(reach))*sigma

A(u)=A0+Re(A1 e^{iu}), B(u)=B0+Re(B1 e^{iu})+Re(B2 e^{2iu}), u = 2 psi -- the
exact convention of fused_log_likelihood_distphipsimarg_laplace.

CRITICAL: psi are ranked by the CLIPPED exponent
    E(u) = x_c A(u) - 0.5 x_c^2 B(u),   x_c = clip(A/B, x_min, x_max)
i.e. at the best PHYSICAL distance, exactly as _distmarg_gh_logL's
center = clip(K/R, x_min, x_max).  Ranking by the unconstrained A^2/(2B) is
exactly degenerate under u -> u+pi when A0 == 0, keeps an unphysical negative-x
branch, and reports a ~300x too large span.

Three centring candidates, the last being the SHIPPABLE rule:
  cf     u = -arg(A1)                       (closed form, no scan)
  exact  argmax over the fine u grid        (upper bound on what is achievable)
  scan   argmax over N_SCAN uniform u, then NEWTON_STEPS Newton steps on A^2/2B
"""
import argparse, json
import numpy as np
import jax.numpy as jnp

import probe
from RIFT.likelihood.jax_ile.anglemarg import angle_coefficient_tables, _kp_weights

N_SCAN = 32
NEWTON_STEPS = 4
THRESH = (30.0, 100.0, 300.0, 1000.0)

p = argparse.ArgumentParser()
p.add_argument("--snr", type=int, default=640)
p.add_argument("--approximant", default="SEOBNRv4")
p.add_argument("--l-max", type=int, default=2)
p.add_argument("--nsky", type=int, default=32)
p.add_argument("--nphi", type=int, default=64)
p.add_argument("--nu", type=int, default=16384)
p.add_argument("--block", type=int, default=4000)
p.add_argument("--sky", default="cloud", choices=("cloud", "gauss"))
p.add_argument("--tag", default="")
a = p.parse_args()

like, ld, prov, opts, drv = probe.build(
    a.snr, angle_marg="laplace", approximant=a.approximant, l_max=a.l_max)
lms = prov["lms"]
m_max = int(np.max(np.abs(np.asarray(lms)[:, 1])))
x_min = float(np.min(np.asarray(like.x_grid)))
x_max = float(np.max(np.asarray(like.x_grid)))
ra, dec, incl = (probe.sky_cloud(a.snr, a.nsky) if a.sky == "cloud"
                 else probe.sky_gauss(a.snr, a.nsky))

C_A, C_B, meta = angle_coefficient_tables(
    ld, jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(incl), prov["interp"])
C_A = np.asarray(C_A); C_B = np.asarray(C_B)
assert int(meta["m_max"]) == m_max, (meta["m_max"], m_max)

wA = np.asarray(_kp_weights(m_max + 1)); wB = np.asarray(_kp_weights(2 * m_max + 1))
phi = np.linspace(0.0, 2 * np.pi, a.nphi, endpoint=False)
EA = np.exp(1j * phi[:, None] * np.arange(m_max + 1)[None, :]) * wA[None, :]
EB = np.exp(1j * phi[:, None] * np.arange(2 * m_max + 1)[None, :]) * wB[None, :]
MA = lambda k: np.einsum("ck,kst->cst", EA, C_A[:, k])
MB = lambda k: np.einsum("ck,kst->cst", EB, C_B[:, k])
A0 = MA(1).real.ravel(); A1 = (MA(2) + np.conj(MA(0))).ravel()
B0 = MB(2).real.ravel(); B1 = (MB(3) + np.conj(MB(1))).ravel()
B2 = (MB(4) + np.conj(MB(0))).ravel()
samp = np.broadcast_to(np.arange(a.nsky)[None, :, None],
                       (a.nphi, a.nsky, ld.npts)).ravel()
N = A0.size
u = np.linspace(0.0, 2 * np.pi, a.nu, endpoint=False)


def AB(A0, A1, B0, B1, B2, uu):
    e1 = np.exp(1j * uu); e2 = np.exp(2j * uu)
    A = A0 + (A1 * e1).real
    Ap = -(A1 * e1).imag
    App = -(A1 * e1).real
    B = B0 + (B1 * e1).real + (B2 * e2).real
    Bp = -(B1 * e1).imag - 2.0 * (B2 * e2).imag
    Bpp = -(B1 * e1).real - 4.0 * (B2 * e2).real
    return A, Ap, App, B, Bp, Bpp


R_lo = B0 - np.abs(B1) - np.abs(B2)
Bmin = np.empty(N); Emax = np.empty(N)
xstar = np.empty(N); xcf = np.empty(N); xsc = np.empty(N)
clip_act = np.empty(N, bool); newton_du = np.empty(N)
span = {t: np.empty(N) for t in THRESH}
reach = {(nm, t): np.empty(N) for nm in ("cf", "exact", "scan") for t in THRESH}
carryfrac = {t: np.empty(N) for t in THRESH}

u_s = np.linspace(0.0, 2 * np.pi, N_SCAN, endpoint=False)
for i0 in range(0, N, a.block):
    sl = slice(i0, min(i0 + a.block, N))
    a0, a1, b0, b1, b2 = A0[sl], A1[sl], B0[sl], B1[sl], B2[sl]
    Au, _, _, Bu, _, _ = AB(a0[:, None], a1[:, None], b0[:, None],
                            b1[:, None], b2[:, None], u)
    Bmin[sl] = Bu.min(-1)
    xs = np.clip(Au / np.maximum(Bu, 1e-30), x_min, x_max)
    E = xs * Au - 0.5 * np.square(xs) * Bu
    em = E.max(-1); iu = np.argmax(E, -1)
    Emax[sl] = em
    xstar[sl] = np.take_along_axis(xs, iu[:, None], -1)[:, 0]
    clip_act[sl] = np.take_along_axis(
        (np.abs(xs - x_min) < 1e-12) | (np.abs(xs - x_max) < 1e-12),
        iu[:, None], -1)[:, 0]
    del Au, Bu
    # closed-form centring
    Acf, _, _, Bcf, _, _ = AB(a0, a1, b0, b1, b2, -np.angle(a1))
    xcf[sl] = np.clip(Acf / np.maximum(Bcf, 1e-30), x_min, x_max)
    # scan + Newton centring (the shippable rule)
    As, _, _, Bs, _, _ = AB(a0[:, None], a1[:, None], b0[:, None],
                            b1[:, None], b2[:, None], u_s)
    xss = np.clip(As / np.maximum(Bs, 1e-30), x_min, x_max)
    u0 = u_s[np.argmax(xss * As - 0.5 * np.square(xss) * Bs, -1)]
    del As, Bs, xss
    un = u0.copy()
    for _ in range(NEWTON_STEPS):
        A_, Ap_, App_, B_, Bp_, Bpp_ = AB(a0, a1, b0, b1, b2, un)
        Bs_ = np.maximum(B_, 1e-30)
        f1 = A_ * Ap_ / Bs_ - 0.5 * A_ ** 2 * Bp_ / Bs_ ** 2
        f2 = ((Ap_ ** 2 + A_ * App_) / Bs_ - 2.0 * A_ * Ap_ * Bp_ / Bs_ ** 2
              - 0.5 * A_ ** 2 * Bpp_ / Bs_ ** 2 + A_ ** 2 * Bp_ ** 2 / Bs_ ** 3)
        step = np.where(f2 < 0, -f1 / np.where(f2 < 0, f2, -1.0), 0.0)
        un = un + np.clip(np.where(np.isfinite(step), step, 0.0),
                          -np.pi / N_SCAN, np.pi / N_SCAN)
    An, _, _, Bn, _, _ = AB(a0, a1, b0, b1, b2, un)
    xsc[sl] = np.clip(An / np.maximum(Bn, 1e-30), x_min, x_max)
    newton_du[sl] = np.abs(((un - u0 + np.pi) % (2 * np.pi)) - np.pi)
    for t in THRESH:
        carry = E > (em[:, None] - t)
        carryfrac[t][sl] = carry.mean(-1)
        xc = np.where(carry, xs, np.nan)
        span[t][sl] = np.nanmax(xc, -1) - np.nanmin(xc, -1)
        for nm, ctr in (("cf", xcf[sl]), ("exact", xstar[sl]), ("scan", xsc[sl])):
            reach[(nm, t)][sl] = np.nanmax(np.abs(xc - ctr[:, None]), -1)
        del carry, xc
    del E, xs

glob = Emax.max()
persamp = np.full(N, -np.inf)
for s in range(a.nsky):
    m = samp == s
    persamp[m] = Emax[m].max()

print("== CONFIG ==")
print(json.dumps(dict(approximant=a.approximant, l_max=a.l_max, lms=lms,
                      m_max=m_max, snr=a.snr, guess_snr=prov["guess_snr"],
                      nsky=a.nsky, nphi=a.nphi, nu=a.nu, sky=a.sky,
                      du=2*np.pi/a.nu, n_scan=N_SCAN,
                      newton_steps=NEWTON_STEPS, npts=int(ld.npts),
                      x_support=[x_min, x_max], n_lattice=int(N),
                      peak_exponent=float(glob))))
print("== STRUCTURE ==")
print("  |A0|max/|A1|max = %.3e   |B1|max/|B0|max = %.3e   median |B2|/B0 = %.4f"
      % (np.abs(A0).max() / np.abs(A1).max(),
         np.abs(B1).max() / np.abs(B0).max(),
         np.median(np.abs(B2) / np.maximum(B0, 1e-300))))

sig = 1.0 / np.sqrt(np.where(R_lo > 0, R_lo, np.nan))


def rep(name, v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        print("  %-30s (empty)" % name); return {}
    q = np.percentile(v, [50, 90, 99, 99.9])
    print("  %-30s median %9.4f  p90 %9.4f  p99 %9.4f  p99.9 %9.4f  max %9.4f"
          % (name, q[0], q[1], q[2], q[3], v.max()))
    return dict(median=float(q[0]), p90=float(q[1]), p99=float(q[2]),
                p999=float(q[3]), max=float(v.max()))


out = {}
for lbl, keep in (("ALL lattice", np.ones(N, bool)),
                  ("weight-carrying per-sample @100nat", Emax > persamp - 100.0),
                  ("weight-carrying global @100nat", Emax > glob - 100.0)):
    print("== [%s]  n=%d (%.4f%%) ==" % (lbl, keep.sum(), 100 * keep.mean()))
    nonpos = R_lo[keep] <= 0
    print("  R_lo <= 0 (HARD REJECT if any): %d / %d (%.4f%%);  min_u B <= 0: %d"
          % (nonpos.sum(), keep.sum(), 100 * nonpos.mean(),
             (Bmin[keep] <= 0).sum()))
    print("  clip active at argmax: %.3f%% ;  Newton |du| max %.3e"
          % (100 * clip_act[keep].mean(), newton_du[keep].max()))
    safe = keep & (R_lo > 0)
    r = dict(n=int(keep.sum()), frac_Rlo_nonpositive=float(nonpos.mean()),
             n_Rlo_nonpositive=int(nonpos.sum()),
             clip_active_frac=float(clip_act[keep].mean()))
    r["W"] = rep("W = sqrt(minB/R_lo)", np.sqrt(Bmin[safe] / R_lo[safe]))
    r["C_cf"] = rep("C (closed-form centre)", np.abs(xcf - xstar)[safe] / sig[safe])
    for t in THRESH:
        r["S@%g" % t] = rep("S span @%gnat" % t, span[t][safe] / sig[safe])
    for nm in ("cf", "exact", "scan"):
        for t in THRESH:
            r["reach_%s@%g" % (nm, t)] = rep(
                "reach %-5s @%gnat" % (nm, t), reach[(nm, t)][safe] / sig[safe])
    r["carryfrac@100"] = rep("psi frac carrying @100nat", carryfrac[100.0][keep])
    out[lbl] = r
print("TASK1 " + json.dumps(dict(tag=a.tag, approximant=a.approximant, snr=a.snr,
                                 l_max=a.l_max, m_max=m_max, lms=lms, stats=out)))
