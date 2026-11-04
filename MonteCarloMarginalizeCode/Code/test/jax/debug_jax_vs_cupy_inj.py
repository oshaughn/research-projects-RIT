"""
Decisive consistency test: on ONE finite-size injection, evaluate lnL(truth)
with (a) the cupy/numpy freqresponse NoLoop and (b) the JAX banded likelihood,
from the SAME packed precompute, for several t_window / tvals configs.

If JAX==cupy at truth, the JAX port is consistent with the validated reference
and any lnL(truth) deficit is a precompute/config effect (fixable by matching the
validated t_window/tvals).  If JAX!=cupy, the JAX build has a convention bug.
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
import RIFT.likelihood.factored_likelihood as flib
import RIFT.likelihood.factored_likelihood_freqresponse as flfr
import RIFT.likelihood.slowrot_freqresponse as sfr
import RIFT.lalsimutils as lsu
import lal

from RIFT.likelihood.jax_ile.banded import build_freqresponse_data
from RIFT.likelihood.jax_ile.core import fused_log_likelihood

NET = os.environ.get("SLOWROT_NET", "CE+ET")
SNR = float(os.environ.get("SLOWROT_SNR", "300"))
QMAX = int(os.environ.get("SLOWROT_QMAX", "4"))


def main():
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=0.4,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0, approx="IMRPhenomD")
    net = fslib.network(NET)
    dist = fslib.distance_for_snr(src, net, SNR)
    dd, pd, arm, meta = fslib.build_finite_size_data(src, net, dist)
    deltaT, deltaF = meta["deltaT"], meta["deltaF"]
    half_dd = meta["half_dd"]
    print("NET=%s SNR=%.0f half<d|d>=%.1f  deltaT=%.3e" % (NET, meta["snr"], half_dd, deltaT))

    rt, dt_, pt, it, ft = src.ra, src.dec, src.psi, src.incl, src.phiref
    det_geom = {d: sfr.detector_geometry(d, L_arm=arm.get(d)) for d in dd}

    for t_window in (0.06, 0.10):
        Psig = fslib._base_params(src, dist, deltaT, deltaF)
        pk = fslib._pack_finite(fslib.EVENT_TIME, t_window, Psig, dd, pd, arm, src.fmax, QMAX)
        for iwh in (0.03, 0.06):
            tvals = flib.marginalization_time_grid(iwh, deltaT)
            # cupy/numpy NoLoop at truth (nearest + cubic)
            Pv = Psig.manual_copy()
            Pv.phi = np.array([rt]); Pv.theta = np.array([dt_]); Pv.psi = np.array([pt])
            Pv.incl = np.array([it]); Pv.phiref = np.array([ft])
            Pv.dist = np.array([dist]) * 1e6 * lsu.lsu_PC
            Pv.tref = float(fslib.EVENT_TIME); Pv.deltaT = deltaT
            cu = {}
            for ti in ("nearest", "cubic"):
                cu[ti] = float(np.asarray(flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
                    tvals, Pv, pk["meta"], pk["lk"], pk["rbp"], pk["ubp"], pk["vbp"], pk["ep"],
                    Lmax=fslib.LMAX, time_interp=ti, xpy=np))[0])
            # JAX banded from the SAME packed data
            data = build_freqresponse_data(pk["meta"], pk["lk"], pk["rbp"], pk["ubp"],
                                           pk["vbp"], pk["ep"], deltaT, tvals, det_geom)
            jx = {ti: float(np.asarray(fused_log_likelihood(
                    data, np.array([rt]), np.array([dt_]), np.array([pt]),
                    np.array([it]), np.array([ft]), np.array([dist]),
                    interp=ti))[0])
                  for ti in ("nearest", "cubic")}
            print("  t_win=%.2f tvals=+/-%.2f : cupy(near/cubic)=%.1f/%.1f  "
                  "JAX(near/cubic)=%.1f/%.1f  half-cupy_near=%.1f (%.2f%%)"
                  % (t_window, iwh, cu["nearest"], cu["cubic"], jx["nearest"], jx["cubic"],
                     half_dd - cu["nearest"], 100 * (half_dd - cu["nearest"]) / half_dd))
    print("DEBUG2 DONE")


if __name__ == "__main__":
    main()
