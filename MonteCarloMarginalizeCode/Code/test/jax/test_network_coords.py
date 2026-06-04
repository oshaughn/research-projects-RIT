"""
Demonstrate that network-frame sky coordinates fold the *likelihood's*
time-delay ring (not just the geometric delay).

We build a two-detector (H1,L1) zero-noise injection, evaluate the
distance-marginalized log-likelihood over the whole sky, collect the
high-likelihood points (the time-delay ring), and show that in the network
frame (polar axis = H1-L1 baseline) those points occupy a NARROW band in the
network polar angle theta_n while spreading broadly in the network azimuth
phi_n.  That is exactly the structure that makes (cos theta_n, phi_n) a good
sampling parametrization: the ring degeneracy lies along phi_n at fixed theta_n.

Run:
  PYTHONPATH=<...>/Code  python test/jax/test_network_coords.py
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
from RIFT.likelihood.jax_ile import (
    build_network_frame, equatorial_to_network,
)

MSUN, PC = lal.MSUN_SI, lal.PC_SI


def build_injection(detectors=("H1", "L1")):
    fid = 1126259462.0
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = 35 * MSUN, 30 * MSUN
    P.s1z, P.s2z = 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 4096, 0.25
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = fid
    P.phi, P.theta, P.psi, P.incl, P.phiref = 1.2, -0.4, 0.7, 0.9, 2.1
    P.dist = 600e6 * PC
    dd, pp = {}, {}
    for det in detectors:
        Pd = P.copy(); Pd.detector = det
        dd[det] = lalsimutils.non_herm_hoff(Pd)
        pp[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    data, _ = build_data_from_precompute(
        P.copy(), dd, pp, fid, 0.15, 0.075, 2, 1000.0,
        analyticPSD_Q=True, verbose=False)
    return data, fid, detectors


def main():
    data, fid, detectors = build_injection(("H1", "L1"))
    like = JAXDistanceMarginalizedLikelihood(data, 1.0, 5000.0, n_grid=128)

    # sky grid (fixed psi/incl/phiref at the injected values to isolate the sky)
    n = 240
    ra = np.linspace(0, 2 * np.pi, n)
    dec = np.arcsin(np.linspace(-0.999, 0.999, n))
    RA, DEC = np.meshgrid(ra, dec)
    RAf, DECf = RA.ravel(), DEC.ravel()
    psi = np.full(RAf.shape, 0.7)
    incl = np.full(RAf.shape, 0.9)
    phiref = np.full(RAf.shape, 2.1)

    lnL = np.empty(RAf.size)
    for i in range(0, RAf.size, 4000):
        sl = slice(i, i + 4000)
        lnL[sl] = np.asarray(like.log_likelihood(
            RAf[sl], DECf[sl], psi[sl], incl[sl], phiref[sl]))

    # high-likelihood points = the time-delay ring.  (With only two detectors
    # and fixed orientation, the antenna pattern modulates the ring, so we take
    # a generous band below the peak to populate it.)
    thresh = lnL.max() - 25.0
    ring = lnL > thresh
    print("sky-grid lnL max = %.2f ; %d/%d points within 25 of peak (the ring)"
          % (lnL.max(), ring.sum(), lnL.size))

    # map ring points to network coordinates (H1-L1 baseline)
    loc1 = np.asarray(lalsim.DetectorPrefixToLALDetector("H1").location)
    loc2 = np.asarray(lalsim.DetectorPrefixToLALDetector("L1").location)
    gmst = lal.GreenwichMeanSiderealTime(fid)
    R = build_network_frame(loc1, loc2, gmst)
    tn, pn = equatorial_to_network(RAf[ring], DECf[ring], R, gmst)
    tn = np.asarray(tn); pn = np.asarray(pn)

    # circular spread of phi_n (azimuth) vs linear spread of theta_n (polar)
    std_tn = float(np.std(tn))
    # circular std of phi_n
    C, Smean = np.mean(np.cos(pn)), np.mean(np.sin(pn))
    Rbar = np.hypot(C, Smean)
    circ_std_pn = float(np.sqrt(-2 * np.log(max(Rbar, 1e-12))))
    print("ring spread:  theta_n std = %.4f rad   phi_n circular std = %.4f rad"
          % (std_tn, circ_std_pn))
    print("              -> the ring is %.1fx narrower in theta_n than in phi_n"
          % (circ_std_pn / max(std_tn, 1e-6)))

    # The fold is meaningful if theta_n is markedly more concentrated than phi_n
    # (the ring degeneracy lies along the azimuth at fixed polar angle).
    assert circ_std_pn > 2.0 * std_tn, \
        "network polar angle not concentrated -> ring not folded as expected"
    print("\nNETWORK-COORDINATE FOLD CONFIRMED: the time-delay ring is a narrow "
          "band in theta_n, spread over phi_n.")


if __name__ == "__main__":
    main()
