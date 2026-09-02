"""Exhaustive reach scan over the reachable (rho, r, delta) family -- corrected.

With A0 == 0 and B1 == 0 (structural; see identity_check.py), and writing
v = 2u, B0 = 1, alpha = 0:

    E(v) = rho^2 g(v),    g(v) = (1 + cos v) / (4 (1 + r cos(v - delta)))
    x*(v)/sigma0 = rho h(v),  h(v) = sqrt((1+cos v)/2) / (1 + r cos(v - delta))
    sigma_rule/sigma0 = 1/sqrt(1 - r)

rho enters ONLY as an overall factor, so g and h are computed once per
(r, delta) on a fine v grid and every rho is a re-threshold of the same arrays.
That is what makes an exhaustive scan affordable at a v resolution fine enough
to resolve the weight-carrying window (half-width ~ sqrt(2T)/rho).

Reported for three centrings:
  cf     v = 0            (the shipped closed form: argmax_u A(u))
  exact  argmax of E      (an upper bound on what any centring can achieve)
  span   full range of x* over the carrying set (centring-free)
"""
import numpy as np, json, sys

T = 100.0
NV = int(sys.argv[1]) if len(sys.argv) > 1 else 262144
rs = np.concatenate([np.linspace(0.0, 0.9, 28), [0.93, 0.95, 0.97, 0.99, 0.995, 0.999, 0.9999]])
ds = np.linspace(0.0, 2 * np.pi, 91)
rhos = np.geomspace(0.2, 3000.0, 70)

v = np.linspace(0.0, 2 * np.pi, NV, endpoint=False)
cv = np.cos(v)
half = np.sqrt(np.maximum((1 + cv) / 2.0, 0.0))          # |cos u| on the + branch
rows = []
for r in rs:
    for d in ds:
        D = 1.0 + r * np.cos(v - d)
        D = np.maximum(D, 1e-300)
        g = (1.0 + cv) / (4.0 * D)
        h = half / D
        gmax = g.max(); ig = int(np.argmax(g))
        h0 = h[0]                                        # v = 0 -> u = 0
        hstar = h[ig]
        for rho in rhos:
            keep = g > (gmax - T / rho ** 2)
            hk = h[keep]
            sc = rho * np.sqrt(1.0 - r)                  # rho * sigma0/sigma_rule
            rows.append((rho, r, d, rho * rho * gmax,
                         np.abs(hk - h0).max() * sc,
                         np.abs(hk - hstar).max() * sc,
                         (hk.max() - hk.min()) * sc,
                         keep.mean(), keep.sum()))
R = np.array(rows)
rho_, r_, d_, em_, rcf_, rex_, sp_, kf_, kn_ = R.T
print("family scan v2: %d points, v grid %d, T = %g nats; carrying-window "
      "samples: min %d median %d" % (len(R), NV, T, kn_.min(), np.median(kn_)))
print("  (rows with < 32 samples in the carrying window are grid-limited: %d)"
      % (kn_ < 32).sum())
ok = kn_ >= 32


def tab(lbl, val):
    print("  %-22s  p50 %9.3f  p99 %9.3f  MAX %11.3f" %
          (lbl, np.percentile(val, 50), np.percentile(val, 99), val.max()))


print("== over the WHOLE family (%d well-resolved rows) ==" % ok.sum())
tab("reach, closed-form", rcf_[ok]); tab("reach, exact argmax", rex_[ok])
tab("span", sp_[ok])
for lim in (0.05, 0.2, 0.5, 0.9):
    m = ok & (r_ <= lim)
    print("== |B2|/B0 <= %.2f  (%d rows) ==" % (lim, m.sum()))
    tab("reach, closed-form", rcf_[m]); tab("reach, exact argmax", rex_[m])
    tab("span", sp_[m])
print("== reach with the EXACT-argmax centring, binned by peak exponent ==")
edges = [0, 30, 100, 300, 1e3, 1e4, 1e5, 1e12]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = ok & (em_ >= lo) & (em_ < hi)
    if m.sum():
        print("  E_max in [%8.0f,%9.0f): %7d rows  p99 %8.3f  MAX %8.3f"
              % (lo, hi, m.sum(), np.percentile(rex_[m], 99), rex_[m].max()))
print("  sqrt(2T) = %.4f" % np.sqrt(2*T))
i = np.argmax(np.where(ok, rex_, -np.inf))
print("  worst EXACT-argmax reach %.3f at rho %.1f r %.4f delta %.3f "
      "(E_max %.4g)" % (rex_[i], rho_[i], r_[i], d_[i], em_[i]))
j = np.argmax(np.where(ok, rcf_, -np.inf))
print("  worst CLOSED-FORM reach %.3f at rho %.1f r %.4f delta %.3f "
      "(E_max %.4g)" % (rcf_[j], rho_[j], r_[j], d_[j], em_[j]))
# largest r at which each centring still fits inside the shipped 12 sigma
for nm, val in (("closed-form", rcf_), ("exact argmax", rex_)):
    bad = ok & (val > 5.0)
    print("  %-13s exceeds 7+5=12 sigma first at |B2|/B0 = %s"
          % (nm, ("%.4f" % r_[bad].min()) if bad.any() else "never"))
print("FAMILY2 " + json.dumps(dict(
    T=T, nv=NV, n=int(ok.sum()),
    max_reach_cf=float(rcf_[ok].max()), max_reach_exact=float(rex_[ok].max()),
    r_first_fail_cf=float(r_[ok & (rcf_ > 5)].min()) if (ok & (rcf_ > 5)).any() else None,
    r_first_fail_exact=float(r_[ok & (rex_ > 5)].min()) if (ok & (rex_ > 5)).any() else None)))
