"""
End-to-end validation: synthetic injection -> data_dict/psd_dict ->
production PrecomputeLikelihoodTerms + PackLikelihoodDataStructuresAsArrays ->
JAX likelihood, compared against the production numpy NoLoop reference.

This exercises the *real* precompute/packing glue (epoch bookkeeping, mode set,
cross terms) rather than synthetic arrays, so it catches any wiring mistake in
``build_data_from_precompute``.

Run:
  PYTHONPATH=<...>/Code  python test/jax/test_jax_endtoend.py
"""

import types
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as FL
from RIFT.likelihood.jax_ile import build_data_from_precompute, JAXExtrinsicLikelihood
from RIFT.likelihood.jax_ile.core import fused_log_likelihood

MSUN = lal.MSUN_SI
PC = lal.PC_SI


def make_injection():
    fiducial_epoch = 1126259462.0
    detectors = ["H1", "L1", "V1"]
    P = lalsimutils.ChooseWaveformParams()
    P.m1 = 35.0 * MSUN
    P.m2 = 30.0 * MSUN
    P.s1z = 0.1
    P.s2z = -0.2
    P.fmin = 30.0
    P.fref = 30.0
    P.deltaT = 1.0 / 4096
    P.deltaF = 1.0 / 4         # 4 s segment
    P.dist = 600.0 * 1e6 * PC
    P.fmax = 0.0
    P.approx = lalsim.IMRPhenomD
    P.radec = True
    P.tref = fiducial_epoch
    # extrinsic truth
    P.phi = 1.2      # RA
    P.theta = -0.4   # DEC
    P.psi = 0.7
    P.incl = 0.9
    P.phiref = 2.1

    data_dict = {}
    psd_dict = {}
    for det in detectors:
        Pdet = P.copy()
        Pdet.detector = det
        data_dict[det] = lalsimutils.non_herm_hoff(Pdet)
        psd_dict[det] = lalsim.SimNoisePSDaLIGOZeroDetHighPower
    return P, data_dict, psd_dict, fiducial_epoch, detectors


def build_Pvec(P_template, S, fiducial_epoch, deltaT, seed=4):
    rng = np.random.default_rng(seed)
    P = types.SimpleNamespace()
    # include the truth as sample 0, random elsewhere
    ra = rng.uniform(0, 2 * np.pi, S);    ra[0] = 1.2
    dec = rng.uniform(-1.4, 1.4, S);      dec[0] = -0.4
    psi = rng.uniform(0, np.pi, S);       psi[0] = 0.7
    incl = rng.uniform(0, np.pi, S);      incl[0] = 0.9
    phiref = rng.uniform(0, 2 * np.pi, S); phiref[0] = 2.1
    distMpc = rng.uniform(200.0, 1500.0, S); distMpc[0] = 600.0
    P.phi, P.theta, P.psi, P.incl, P.phiref = ra, dec, psi, incl, phiref
    P.dist = distMpc * PC * 1e6
    P.tref = float(fiducial_epoch)
    P.deltaT = deltaT
    return P, distMpc


def main():
    P, data_dict, psd_dict, fiducial_epoch, detectors = make_injection()
    storage_window_half = 0.15      # rholm buffer half-width
    integration_window_half = 0.075  # marginalization window half-width
    Lmax = 2
    fMax = 1000.0

    data, extras = build_data_from_precompute(
        P.copy(), data_dict, psd_dict, fiducial_epoch,
        storage_window_half, integration_window_half, Lmax, fMax,
        analyticPSD_Q=True, verbose=False)
    print("guessed SNR:", extras["guess_snr"])
    print("modes per detector:", data.lms)

    # Reference inputs packed straight from extras / production packer
    lookupNKDict, rholmsArrayDict, ctUArrayDict, ctVArrayDict, epochDict = {}, {}, {}, {}, {}
    for det in detectors:
        (lookupNK, _kn, _knc, ctU, ctV, rholmArray, _intp, epoch) = \
            FL.PackLikelihoodDataStructuresAsArrays(
                list(extras["rholms"][det].keys()), extras["rholms_intp"][det],
                extras["rholms"][det], extras["cross_terms"][det],
                extras["cross_terms_V"][det])
        lookupNKDict[det] = lookupNK
        rholmsArrayDict[det] = rholmArray
        ctUArrayDict[det] = ctU
        ctVArrayDict[det] = ctV
        epochDict[det] = epoch

    S = 40
    Pvec, distMpc = build_Pvec(P, S, fiducial_epoch, P.deltaT)
    # Compare like with like: hand the numpy reference the SAME time grid the
    # JAX data object was built with.  Both paths consume only tvals[0] and
    # len(tvals) -- each steps by P.deltaT and integrates with dx=deltaT -- so a
    # *sub-sample* difference in tvals[0] rounds ifirst to a DIFFERENT integer
    # sample for a sky-dependent subset of samples, and a different subset per
    # detector, which misaligns the coherent network sum by one sample.  Building
    # an independent grid here therefore reported a ~67.8 nat "mismatch" that was
    # an artifact OF THIS HARNESS.
    #
    # As of issue #146 the same 67.8 nats is no longer ALSO a live disagreement
    # between the two production drivers: both
    # bin/integrate_likelihood_extrinsic_batchmode (all ten window-grid sites) and
    # the jax_ile wrapper now call factored_likelihood.marginalization_time_grid().
    # test/jax/test_tvals_grid_convention.py is what asserts that, at five sample
    # rates including 16384; this test still holds the grid fixed and tests only
    # that the two LIKELIHOODS agree, at 4096.
    tvals = np.asarray(data.tvals)
    # Pin the builder's convention by VALUE, independently reconstructed.  (An
    # `assert len(tvals) == data.npts` would be a tautology -- core.py sets
    # npts = len(tvals) -- and would not have caught the original defect
    # either, since both grids had length 614 and differed only in offset.)
    _npts = int(2 * integration_window_half / P.deltaT)
    np.testing.assert_allclose(tvals, (np.arange(_npts) - _npts // 2) * P.deltaT,
                               rtol=0, atol=0,
                               err_msg="build_data_from_precompute tvals convention changed; "
                                       "a linspace grid here silently misaligns ifirst")

    lnL_ref = FL.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pvec, lookupNKDict, rholmsArrayDict, ctUArrayDict, ctVArrayDict,
        epochDict, Lmax=Lmax, xpy=np)

    lnL_jax = np.asarray(fused_log_likelihood(
        data, Pvec.phi, Pvec.theta, Pvec.psi, Pvec.incl, Pvec.phiref, distMpc,
        interp="nearest"))

    # The production reference subtracts a single GLOBAL max before the time
    # integral, so samples hundreds of lnL below the peak underflow to -inf.
    # The JAX path uses a per-sample max (more robust) and stays finite there.
    # Compare only where the reference is finite.
    finite = np.isfinite(lnL_ref)
    n_uf = int((~finite).sum())
    err = np.max(np.abs(lnL_ref[finite] - lnL_jax[finite]))
    rel = np.max(np.abs(lnL_ref[finite] - lnL_jax[finite])
                 / (1 + np.abs(lnL_ref[finite])))
    print(f"\n[end-to-end nearest vs reference]  max|abs|={err:.3e}  max|rel|={rel:.3e}"
          f"  ({n_uf} reference-underflow samples excluded; JAX finite there)")
    print(f"  lnL at injected truth: ref={lnL_ref[0]:.4f}  jax={lnL_jax[0]:.4f}")
    print(f"  lnL range (ref, finite): [{lnL_ref[finite].min():.3f}, {lnL_ref[finite].max():.3f}]")
    assert np.all(np.isfinite(lnL_jax)), "JAX produced non-finite lnL"
    assert err < 1e-5, f"end-to-end mismatch {err}"

    # AD sanity: gradient + Fisher at the injected truth (linear interp)
    like = JAXExtrinsicLikelihood(data, interp="linear")
    theta0 = np.array([1.2, -0.4, 0.7, 0.9, 2.1, 600.0])
    val, grad = like.value_and_grad(theta0)
    fish = like.fisher(theta0)
    print(f"\n[AD] lnL(truth)={val:.4f}")
    print("  grad =", np.array2string(grad, precision=3))
    print("  Fisher diag =", np.array2string(np.diag(fish), precision=3))
    assert np.all(np.isfinite(grad)) and np.all(np.isfinite(fish))

    # Distance marginalization regulates the amplitude/inclination divergence:
    # the incl -> pi null must drop to a baseline (not blow up), and the peak
    # must sit near the true sky location.
    from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood
    dlike = JAXDistanceMarginalizedLikelihood(data, 1.0, 5000.0, n_grid=128)
    lnL_truth = dlike.value([1.2, -0.4, 0.7, 0.9, 2.1])
    lnL_null = dlike.value([1.2, -0.4, 0.7, np.pi - 1e-3, 2.1])
    print(f"\n[distmarg] lnL(truth angles)={lnL_truth:.3f}  "
          f"lnL(incl->pi null)={lnL_null:.3f}")
    assert np.isfinite(lnL_truth) and np.isfinite(lnL_null)
    assert lnL_null < lnL_truth - 50, \
        "distance marginalization did not regulate the inclination null"
    # finite over a random whole-sky batch (chunked)
    rng = np.random.default_rng(9)
    M = 4000
    ang = np.stack([rng.uniform(0, 2 * np.pi, M), np.arcsin(rng.uniform(-1, 1, M)),
                    rng.uniform(0, np.pi, M), np.arccos(rng.uniform(-1, 1, M)),
                    rng.uniform(0, 2 * np.pi, M)], axis=-1)
    v = np.asarray(dlike.log_likelihood(ang[:, 0], ang[:, 1], ang[:, 2],
                                        ang[:, 3], ang[:, 4]))
    assert np.all(np.isfinite(v)), "distmarg produced non-finite lnL on the sky"
    best = ang[np.argmax(v)]
    print(f"  random-sky max lnL={v.max():.3f} at sky (RA,DEC)="
          f"({best[0]:.2f},{best[1]:.2f})  [truth (1.20,-0.40)]")

    print("\nEND-TO-END TEST PASSED")


def test_endtoend():
    """pytest entry point.  Without this the file defines no test_* function and
    `pytest test/jax/` collects ZERO items from it and exits 5 ("no tests ran"),
    which reads as green -- which is how this test stayed broken for a month."""
    main()


if __name__ == "__main__":
    main()
