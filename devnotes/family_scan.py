"""EXHAUSTIVE check of the +-12 sigma half-span over the whole reachable family.

Measured (identity_check.py) on real IMRPhenomXHM data through m_max = 4, and
on synthetic data with random U/V: A0 == 0 and B1 == 0 to machine precision for
EVERY mode set -- the spin-2 antenna response F(psi) ~ e^{-2 i psi} puts the
kappa term at exactly one u-harmonic and the rho^2 term at exactly harmonics
0 and 2, whatever the mode content.  So

    A(u) = |A1| cos(u - alpha),   B(u) = B0 + |B2| cos(2u - beta)

ALWAYS, and after scaling B0 -> 1, sigma0 = 1/sqrt(B0), and shifting alpha -> 0,
the bracket problem depends on exactly three numbers:

    rho = |A1|/sqrt(B0)      r = |B2|/B0 in [0,1)      delta = beta - 2 alpha

The shipped rule's one-sided reach -- max over weight-carrying u of
|x*(u) - x*(0)| / sigma_rule, sigma_rule = 1/sqrt(B0 (1-r)) -- is therefore a
function of (rho, r, delta) alone and can be scanned exhaustively rather than
sampled on a fixture.  Clipping into [x_min, x_max] is 1-Lipschitz, so the
UNCLIPPED reach computed here is an upper bound on the clipped one.
"""
import numpy as np, json

T = 100.0
nu = 20001
u = np.linspace(0.0, 2 * np.pi, nu, endpoint=False)
rhos = np.concatenate([np.linspace(0.5, 20, 40), np.geomspace(20, 5000, 60)])
rs = np.concatenate([np.linspace(0.0, 0.9, 46), 1 - np.geomspace(0.1, 1e-3, 20)])
ds = np.linspace(0.0, 2 * np.pi, 181)

rows = []
for r in rs:
    for d in ds:
        B = 1.0 + r * np.cos(2 * u - d)                     # (nu,)
        C = np.cos(u)
        for rho in rhos:
            A = rho * C
            xs = A / B                                       # x*/sigma0
            E = np.where(A > 0, A * A / (2.0 * B), 0.0)      # clipped at x>=0
            em = E.max()
            keep = E > (em - T)
            x0 = xs[np.argmin(np.abs(((u - 0.0 + np.pi) % (2 * np.pi)) - np.pi))]
            reach = np.abs(xs[keep] - x0).max() * np.sqrt(1.0 - r)
            rows.append((rho, r, d, em, reach, keep.mean()))
R = np.array(rows)
rho_, r_, d_, em_, re_, kf_ = R.T
print("family scan: %d (rho,r,delta) points, u grid %d, T = %g nats"
      % (len(R), nu, T))
for lo in (0.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 1e4):
    m = em_ >= lo
    if not m.any():
        continue
    i = np.argmax(np.where(m, re_, -np.inf))
    print("  peak exponent >= %8.0f nats (%6d pts): reach p99 %8.3f  MAX %8.3f "
          " at rho %8.2f r %.4f delta %.3f  (E_max %.4g, carrying frac %.4f)"
          % (lo, m.sum(), np.percentile(re_[m], 99), re_[i], rho_[i], r_[i],
             d_[i], em_[i], kf_[i]))
print("  needed half-width = 7 + reach")
bad = re_ > 5.0
print("  reach > 5 sigma at %d/%d points; of those, max peak exponent = %.4g nats"
      % (bad.sum(), len(R), em_[bad].max() if bad.any() else float("nan")))
th = []
for lim in (5.0, 4.0, 3.0):
    m = re_ > lim
    th.append((lim, float(em_[m].max()) if m.any() else float("nan")))
    print("  reach > %.0f sigma requires peak exponent <= %.4g nats"
          % (lim, th[-1][1]))
print("FAMILY " + json.dumps(dict(T=T, nu=nu, n=len(R),
                                  max_reach=float(re_.max()),
                                  thresholds=th)))
