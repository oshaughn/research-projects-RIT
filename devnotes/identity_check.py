"""Is A0 == 0 / B1 == 0 a (2,+-2) accident, or structural for ANY mode set?

If it is structural then R_lo = B0 - |B1| - |B2| is min_u B EXACTLY for every
mode set, and the psi-marginal bracket problem collapses to the three-parameter
family (|A1|/sqrt(B0), |B2|/B0, relative phase) -- which can be verified
exhaustively rather than on a fixture.
"""
import sys, numpy as np, jax.numpy as jnp
import probe
from RIFT.likelihood.jax_ile import anglemarg as AM


def resid(ld, ra, dec, incl, interp, nphi=32):
    C_A, C_B, meta = AM.angle_coefficient_tables(
        ld, jnp.asarray(ra), jnp.asarray(dec), jnp.asarray(incl), interp)
    C_A = np.asarray(C_A); C_B = np.asarray(C_B)
    m = int(meta["m_max"])
    wA = np.asarray(AM._kp_weights(m + 1)); wB = np.asarray(AM._kp_weights(2 * m + 1))
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    EA = np.exp(1j * phi[:, None] * np.arange(m + 1)) * wA
    EB = np.exp(1j * phi[:, None] * np.arange(2 * m + 1)) * wB
    MA = lambda k: np.einsum("ck,kst->cst", EA, C_A[:, k])
    MB = lambda k: np.einsum("ck,kst->cst", EB, C_B[:, k])
    kA = (C_A.shape[1] - 1) // 2; kB = (C_B.shape[1] - 1) // 2
    A0 = MA(kA).real; A1 = MA(kA + 1) + np.conj(MA(kA - 1))
    B0 = MB(kB).real; B1 = MB(kB + 1) + np.conj(MB(kB - 1))
    B2 = MB(kB + 2) + np.conj(MB(kB - 2))
    return dict(m_max=m, a0=float(np.abs(A0).max() / np.abs(A1).max()),
                b1=float(np.abs(B1).max() / np.abs(B0).max()),
                b2med=float(np.median(np.abs(B2) / np.maximum(B0, 1e-300))),
                b2p99=float(np.percentile(np.abs(B2) / np.maximum(B0, 1e-300), 99)),
                rlo_nonpos=float(((B0 - np.abs(B1) - np.abs(B2)) <= 0).mean()))


for ap, lm in (("SEOBNRv4", 2), ("IMRPhenomXHM", 3), ("IMRPhenomXHM", 4)):
    like, ld, prov, opts, drv = probe.build(160, angle_marg="laplace",
                                            approximant=ap, l_max=lm, iwh=0.005)
    ra, dec, incl = probe.sky_gauss(160, 16)
    r = resid(ld, ra, dec, incl, prov["interp"])
    print("STRUCT %-14s l_max=%d lms=%s m_max=%d  |A0|/|A1|=%.3e  |B1|/|B0|=%.3e"
          "  |B2|/B0 med=%.4f p99=%.4f  R_lo<=0 frac=%.4f"
          % (ap, lm, prov["lms"], r["m_max"], r["a0"], r["b1"], r["b2med"],
             r["b2p99"], r["rlo_nonpos"]), flush=True)
