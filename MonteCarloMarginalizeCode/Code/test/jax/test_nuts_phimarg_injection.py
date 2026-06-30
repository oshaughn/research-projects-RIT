#!/usr/bin/env python3
"""Smoke test: fisher_nuts_sample_phimarg on the standard analytic injection.

Tiny budgets — checks the path runs end-to-end, recovers the truth sky
(RA, DEC) = (1.2, -0.4), and returns finite logZ / neff / post_weight.
No frames needed (analytic PSD, synthesized data).
"""
import sys, time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim
import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistPhiMargLikelihood
from RIFT.likelihood.jax_ile import samplers as S

MSUN, PC = lal.MSUN_SI, lal.PC_SI
fiducial_epoch = 1126259462.0
P = lalsimutils.ChooseWaveformParams()
P.m1, P.m2 = 35.0 * MSUN, 30.0 * MSUN
P.s1z, P.s2z = 0.1, -0.2
P.fmin = P.fref = 30.0
P.deltaT = 1.0 / 4096
P.deltaF = 1.0 / 4
P.dist = 600.0 * 1e6 * PC
P.fmax = 0.0
P.approx = lalsim.IMRPhenomD
P.radec = True
P.tref = fiducial_epoch
P.phi, P.theta = 1.2, -0.4          # truth RA, DEC
P.psi, P.incl, P.phiref = 0.7, 0.9, 2.1

data_dict, psd_dict = {}, {}
for det in ("H1", "L1", "V1"):
    Pd = P.copy(); Pd.detector = det
    data_dict[det] = lalsimutils.non_herm_hoff(Pd)
    psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower

print("precompute ...", flush=True)
data, extras = build_data_from_precompute(
    P.copy(), data_dict, psd_dict, fiducial_epoch,
    storage_window_half=0.15, integration_window_half=0.075,
    Lmax=2, fMax=1000.0, analyticPSD_Q=True, verbose=False)
print("guess SNR:", float(extras["guess_snr"]), flush=True)

like = JAXDistPhiMargLikelihood(data, 1.0, 5000.0, nphi=32, n_grid=128)

t0 = time.time()
res = S.fisher_nuts_sample_phimarg(
    like, num_warmup=60, num_samples=120, n_starts=6, n_modes=3,
    n_prior_pilot=3000, n_is=4000, seed=3, verbose=True)
wall = time.time() - t0

th, lnL, pw = res["theta"], res["lnL"], res["post_weight"]
print("\n--- results (%.0f s) ---" % wall)
print("draws: %d   max lnL=%.2f" % (len(th), lnL.max()))
print("modes (ra, dec, lnL):")
for m, l in zip(res["modes"], res["mode_lnL"]):
    print("   (%.3f, %.3f)  lnL=%.2f" % (m[0], m[1], l))
print("logZ=%.3f  sigma/Z=%.3f  neff=%.1f" %
      (res["logZ"], res["sigma_over_Z"], res["neff"]))

# weighted sky mean vs truth
ra_m = float((pw * th[:, 0]).sum()); dec_m = float((pw * th[:, 1]).sum())
dsky = np.arccos(np.clip(np.sin(dec_m) * np.sin(-0.4)
                         + np.cos(dec_m) * np.cos(-0.4) * np.cos(ra_m - 1.2),
                         -1, 1))
print("weighted sky mean (%.3f, %.3f); great-circle dist to truth = %.4f rad"
      % (ra_m, dec_m, dsky))

ok = (np.isfinite(res["logZ"]) and np.isfinite(res["neff"])
      and res["neff"] > 5 and dsky < 0.15
      and np.isclose(pw.sum(), 1.0))
print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
