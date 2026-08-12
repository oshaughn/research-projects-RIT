"""
Debug the injection<->recovery consistency for the finite-size (Path D) path.

Injects a finite-size signal with slowrot_fs_lib and evaluates the JAX
freqresponse likelihood at the EXACT injected extrinsic parameters, comparing to
the Cauchy-Schwarz bound half<d|d>.  If lnL(truth) ~= half<d|d> and the sky scan
peaks at truth, the likelihood is consistent (any offset is a sampler artifact);
if lnL(truth) < half<d|d> and a shifted sky scores higher, the injection and
recovery models are inconsistent (a real convention bug).

Also runs the SAME check on a POINT-response (L_arm->0) injection to isolate
whether the offset is specific to the finite-size path or a general sky/time
convention issue.

Run in the JAX container:
  apptainer exec --nv <sif> env ... PYTHONPATH=<Code>:<fslib> \
    python test/jax/debug_fs_consistency.py
"""
import os
import sys
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

_FSLIB = os.environ.get("SLOWROT_FS_LIB_DIR",
                        os.path.expanduser("~/RIFT_roboto_paper/analyses/slowrot_finite-size"))
sys.path.insert(0, _FSLIB)
import slowrot_fs_lib as fslib

from RIFT.likelihood.jax_ile.wrapper import build_freqresponse_data_from_precompute
from RIFT.likelihood.jax_ile.core import fused_log_likelihood

NET = os.environ.get("SLOWROT_NET", "CE+ET")
SNR = float(os.environ.get("SLOWROT_SNR", "300"))
QMAX = int(os.environ.get("SLOWROT_QMAX", "4"))
IWH, TBUF = 0.03, 0.12


def _eval(data, ra, dec, psi, incl, phiref, dist, interp):
    return float(np.asarray(fused_log_likelihood(
        data, np.array([ra]), np.array([dec]), np.array([psi]),
        np.array([incl]), np.array([phiref]), np.array([dist]), interp=interp))[0])


def run(net_name, arm_scale=1.0, tag=""):
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=0.4,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0, approx="IMRPhenomD")
    net = fslib.network(net_name)
    if arm_scale != 1.0:                     # shrink arms -> point (LWL) response
        net = {d: (psd, L * arm_scale) for d, (psd, L) in net.items()}
    dist = fslib.distance_for_snr(src, net, SNR)
    dd, pd, arm, meta = fslib.build_finite_size_data(src, net, dist)
    P0 = fslib._base_params(src, dist, meta["deltaT"], meta["deltaF"])
    data, _ = build_freqresponse_data_from_precompute(
        P0, dd, pd, fslib.EVENT_TIME, IWH, fslib.LMAX, src.fmax,
        t_window=TBUF, Qmax=QMAX, L_arm=arm, analyticPSD_Q=True, verbose=False)
    half_dd = meta["half_dd"]
    rt, dt_, pt, it, ft = src.ra, src.dec, src.psi, src.incl, src.phiref
    print("\n=== %s%s  SNR=%.0f  half<d|d>=%.1f  (arm_scale=%g) ===" %
          (net_name, tag, meta["snr"], half_dd, arm_scale))
    for interp in ("nearest", "linear"):
        lnL_t = _eval(data, rt, dt_, pt, it, ft, dist, interp)
        print("  lnL(truth, %s) = %.1f   half<d|d>-lnL = %.1f  (%.2f%%)" %
              (interp, lnL_t, half_dd - lnL_t, 100 * (half_dd - lnL_t) / half_dd))
    # fine sky scan about truth (fixed true dist/incl/psi/phiref), CUBIC interp
    best = (-np.inf, 0, 0)
    for ddec in np.linspace(-2.5, 2.5, 41):
        for dra in np.linspace(-2.5, 2.5, 41):
            ra = rt + np.radians(dra) / np.cos(dt_)
            dec = dt_ + np.radians(ddec)
            v = _eval(data, ra, dec, pt, it, ft, dist, "cubic")
            if v > best[0]:
                best = (v, dra, ddec)
    lnL_t = _eval(data, rt, dt_, pt, it, ft, dist, "cubic")
    print("  sky scan peak: lnL=%.1f at (dRA*cosd,dDec)=(%+.2f,%+.2f) deg  vs lnL(truth)=%.1f  "
          "=> peak-offset=%.2f deg" % (best[0], best[1], best[2], lnL_t,
                                       np.hypot(best[1], best[2])))
    return best


def main():
    print("QMAX=%d" % QMAX)
    # 1) finite-size injection + finite-size recovery (the case that showed 1.6 deg)
    run(NET, arm_scale=1.0, tag=" [finite-size]")
    # 2) near-point injection (arms x0.001 -> LWL response) + same recovery:
    #    isolates whether the offset is the finite-size response or a general
    #    sky/time convention (a point injection must peak at truth).
    run(NET, arm_scale=1e-3, tag=" [near-point/LWL]")
    print("\nDEBUG DONE")


if __name__ == "__main__":
    main()
