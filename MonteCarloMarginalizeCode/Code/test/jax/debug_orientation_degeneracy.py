"""Is the SNR~600 orientation offset a genuine (2,2)-mode degeneracy or a sampler miss?

Rebuilds the representative finite-size event and evaluates the distance-
marginalized lnL at (truth sky, TRUTH orientation) vs (truth sky, RECOVERED
orientation from samples_snr600.npz) and a small scan over (incl, phiref).  If
lnL(truth) ~= lnL(recovered), the two orientations are observationally
degenerate for the dominant quadrupole (expected; HM would break it); if
lnL(truth) >> lnL(recovered) the sampler missed the injected mode.
"""
import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

_FSLIB = os.environ.get("SLOWROT_FS_LIB_DIR",
                        os.path.expanduser("~/RIFT_roboto_paper/analyses/slowrot_finite-size"))
sys.path.insert(0, _FSLIB)
import slowrot_fs_lib as fslib
from RIFT.likelihood.jax_ile.wrapper import (
    build_freqresponse_data_from_precompute, JAXDistanceMarginalizedLikelihood)

NET = os.environ.get("SLOWROT_NET", "CE+ET+K")
SNR = float(os.environ.get("SLOWROT_SNR_REP", "600"))
QMAX = int(os.environ.get("SLOWROT_QMAX", "4"))
IWH, TBUF = 0.03, 0.12
FIGDIR = os.environ.get("SLOWROT_FIG_DIR",
                        os.path.join(_FSLIB, "3g", "figdata_3site"))


def main():
    INCL = float(os.environ.get("SLOWROT_INCL", "0.4"))
    src = fslib.Source(m1=1.6, m2=1.4, ra=1.2, dec=0.3, psi=0.5, incl=INCL,
                       phiref=0.0, fmin=50.0, fmax=1024.0, seglen=32.0, approx="IMRPhenomD")
    net = fslib.network(NET)
    dist = fslib.distance_for_snr(src, net, SNR)
    dd, pd, arm, meta = fslib.build_finite_size_data(src, net, dist)
    P0 = fslib._base_params(src, dist, meta["deltaT"], meta["deltaF"])
    half_dd = meta["half_dd"]
    rt, dt = src.ra, src.dec
    print("=== %s SNR=%.0f  half<d|d>=%.1f ===" % (NET, meta["snr"], half_dd))

    # Qmax truncation sweep: does lnL(truth)->half<d|d> and does the injected
    # orientation become the MAP as the finite-size basis is resolved?
    qsweep = [int(x) for x in os.environ.get("SLOWROT_QSWEEP", "4,8,12,16").split(",")]
    for q in qsweep:
        data, _ = build_freqresponse_data_from_precompute(
            P0, dd, pd, fslib.EVENT_TIME, IWH, fslib.LMAX, src.fmax,
            t_window=TBUF, Qmax=q, L_arm=arm, analyticPSD_Q=True, verbose=False)
        d_min = max(1.0, dist * 0.3); d_max = dist * 2.5
        like = JAXDistanceMarginalizedLikelihood(data, d_min, d_max, n_grid=256, interp="cubic")

        def L(ra, dec, psi, incl, phiref):
            return float(np.asarray(like.log_likelihood(
                np.array([ra]), np.array([dec]), np.array([psi]),
                np.array([incl]), np.array([phiref]))[0]))

        lnL_truth = L(rt, dt, src.psi, src.incl, src.phiref)
        # best-incl at truth sky/psi/phiref (map the degeneracy displacement)
        incs = np.radians(np.linspace(2, 80, 40))
        li = np.array([L(rt, dt, src.psi, ic, src.phiref) for ic in incs])
        ibest = int(np.argmax(li))
        print("Qmax=%2d: lnL(truth)=%.1f  deficit=half<d|d>-lnL=%.1f (%.3f%%)  "
              "best-incl=%.1f deg (truth 22.9), lnL@best-truth=%+.1f"
              % (q, lnL_truth, half_dd - lnL_truth,
                 100 * (half_dd - lnL_truth) / half_dd,
                 np.degrees(incs[ibest]), li[ibest] - lnL_truth))


if __name__ == "__main__":
    main()
