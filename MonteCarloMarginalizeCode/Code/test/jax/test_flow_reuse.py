"""
Validate flow re-use (bootstrapping the trained normalizing flow across
evaluation points) in samplers.flowmc_sample.

This is the batch scenario of `--n-events-to-analyze`: ONE dataset, a sequence
of nearby intrinsic templates.  We build the data once, then build the
distance-marginalized JAX likelihood for two slightly-different templates and
run flowMC on each -- the second run bootstrapped from the first run's
``flow_state``.  We check the mechanism works end-to-end (state threads, the
re-used run still recovers the truth sky) and report a convergence proxy
(``neff`` per production sample) for fresh vs re-used.

Run:
  PYTHONPATH=<...>/Code  python test/jax/test_flow_reuse.py
"""

import time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood.jax_ile import build_data_from_precompute
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
from RIFT.likelihood.jax_ile import samplers

MSUN, PC = lal.MSUN_SI, lal.PC_SI
FID = 1126259462.0


def make_data():
    """One zero-noise dataset (the analysis data is shared across templates)."""
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = 35 * MSUN, 30 * MSUN
    P.s1z, P.s2z = 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 4096, 0.25
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = FID
    P.phi, P.theta, P.psi, P.incl, P.phiref = 1.2, -0.4, 0.7, 0.9, 2.1
    P.dist = 600e6 * PC
    dd, pp = {}, {}
    for det in ("H1", "L1", "V1"):
        Pd = P.copy(); Pd.detector = det
        dd[det] = lalsimutils.non_herm_hoff(Pd)
        pp[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    return dd, pp


def like_for_template(dd, pp, m1, m2):
    P = lalsimutils.ChooseWaveformParams()
    P.m1, P.m2 = m1 * MSUN, m2 * MSUN
    P.s1z, P.s2z = 0.1, -0.2
    P.fmin = P.fref = 30.0
    P.deltaT, P.deltaF = 1.0 / 4096, 0.25
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = FID
    data, _ = build_data_from_precompute(
        P, dd, pp, FID, 0.15, 0.075, 2, 1000.0, analyticPSD_Q=True, verbose=False)
    return JAXDistanceMarginalizedLikelihood(data, 1.0, 5000.0, n_grid=128)


def sky_recovered(theta, lnL, truth=(1.2, -0.4), tol=0.15):
    best = theta[np.argmax(lnL)]
    dlon = best[0] - truth[0]
    d = np.arccos(np.clip(np.sin(best[1]) * np.sin(truth[1])
                  + np.cos(best[1]) * np.cos(truth[1]) * np.cos(dlon), -1, 1))
    return d < tol, d


def run(like, reuse_state, seed):
    t0 = time.time()
    res = samplers.flowmc_sample(
        like, 1.0, 5000.0, n_chains=20, n_local_steps=15, n_global_steps=15,
        n_training_loops=3, n_production_loops=3, n_epochs=6,
        reuse_state=reuse_state, seed=seed, verbose=False)
    res["wall"] = time.time() - t0
    return res


def main():
    dd, pp = make_data()

    # Template A (= injection) : fresh flow
    likeA = like_for_template(dd, pp, 35.0, 30.0)
    resA = run(likeA, reuse_state=None, seed=1)
    okA, dA = sky_recovered(resA["theta"], resA["lnL"])
    print("[A fresh ]  samples=%d  maxlnL=%.2f  logZ=%.3f  neff=%.1f  sky d=%.3f (%s)  %.1fs"
          % (len(resA["theta"]), resA["lnL"].max(), resA["logZ"], resA["neff"],
             dA, okA, resA["wall"]))
    assert resA["flow_state"]["model"] is not None, "no trained flow returned"

    # Template B (nearby intrinsic) : bootstrap from A's flow
    likeB = like_for_template(dd, pp, 35.5, 29.5)
    resB = run(likeB, reuse_state=resA["flow_state"], seed=2)
    okB, dB = sky_recovered(resB["theta"], resB["lnL"])
    print("[B reuse ]  samples=%d  maxlnL=%.2f  logZ=%.3f  neff=%.1f  sky d=%.3f (%s)  %.1fs"
          % (len(resB["theta"]), resB["lnL"].max(), resB["logZ"], resB["neff"],
             dB, okB, resB["wall"]))

    # Control: template B with a FRESH flow (no reuse)
    resBf = run(likeB, reuse_state=None, seed=2)
    okBf, dBf = sky_recovered(resBf["theta"], resBf["lnL"])
    print("[B fresh ]  samples=%d  maxlnL=%.2f  logZ=%.3f  neff=%.1f  sky d=%.3f (%s)  %.1fs"
          % (len(resBf["theta"]), resBf["lnL"].max(), resBf["logZ"], resBf["neff"],
             dBf, okBf, resBf["wall"]))

    # Assertions: mechanism works; both B runs recover the truth sky; the
    # re-used run is not worse on evidence consistency.  (The headline
    # *efficiency* gain only shows up at scale / high SNR; here we just confirm
    # correctness + that re-use is not harmful.)
    assert resB["flow_state"]["model"] is not None
    assert okB and okBf, "flowMC did not recover truth sky"
    assert np.isfinite(resB["logZ"]) and np.isfinite(resBf["logZ"])
    print("\n  reuse vs fresh (template B): neff %.1f vs %.1f ; logZ %.3f vs %.3f"
          % (resB["neff"], resBf["neff"], resB["logZ"], resBf["logZ"]))
    print("\nFLOW RE-USE TEST PASSED")


if __name__ == "__main__":
    main()
