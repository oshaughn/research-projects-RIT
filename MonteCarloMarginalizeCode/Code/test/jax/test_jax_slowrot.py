"""
Validation ladder for the slow-rotation (Path A/B) and finite-size (Path D)
JAX likelihoods, mirroring test/jax/test_jax_endtoend.py but for the banded
features.

Gates:
  (a) JAX interp="nearest" reproduces the cupy/numpy NoLoop references
        DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation  (rotation)
        DiscreteFactoredLogLikelihoodFreqResponseNoLoop                (freqresponse)
      on the SAME packed data, to ~1e-13.  Rotation runs at BOTH p_max=0 (Path A) and
      p_max=1 (Path B) -- see check_rotation() for why Path B is a distinct code path
      for the arrival-time post-phase and not just a wider bank.  p_max=2 is NOT run: the
      bank carries |ntilde| <= 2 + p_max (issue #142), so it would be 27 bands / 729 U/V
      cross terms in the precompute (vs 14 / 196 at p_max=1 and 5 / 25 at p_max=0), which
      roughly triples this file's runtime for no branch p_max=1 does not already exercise
      -- the same duplicate-m scatter-add and within-p V reflection.
  (b) interp="linear" gradient (distance-marginalized, smooth) vs finite diff ~1e-6.
  (c) jit / vmap / grad / hessian all execute and stay finite.

Agreement with the NoLoop (gate a) is NECESSARY BUT NOT SUFFICIENT for the rotation path: a
likelihood that drops the arrival-time post-phase from BOTH terms is perfectly self-consistent
and still ~95 nats wrong.  The VALUE is pinned separately, by the Cauchy-Schwarz / explicit-model
ladder in test/jax/test_jax_slowrot_cauchy_schwarz.py.

Run:
  PYTHONPATH=<...>/Code  taskset -c 0-3 python test/jax/test_jax_slowrot.py
"""
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.factored_likelihood_freqresponse as flfr
import RIFT.likelihood.slowrot_freqresponse as sfr

from RIFT.likelihood.jax_ile.core import fused_log_likelihood
from RIFT.likelihood.jax_ile.banded import (build_rotation_data,
                                            build_freqresponse_data)
from RIFT.likelihood.jax_ile.wrapper import JAXDistanceMarginalizedLikelihood

if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

fSample = 4096.0; fmin = 30.0; fmax = 1700.0; event_time = 1e9
t_window = 0.1; Lmax = 2; deltaT = 1.0 / fSample; deltaF = 1.0 / 4.0
HARM = (-2, -1, 0, 1, 2)
L_CE = 40000.0; Qmax = 4
PC = lal.PC_SI

Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector='H1',
    dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=event_time, deltaF=deltaF)
DETS = ("H1", "L1", "V1")
data_dict = {}
for d in DETS:
    _p = Psig.manual_copy(); _p.detector = d
    data_dict[d] = lsu.non_herm_hoff(_p)
psd_dict = {d: lalsim.SimNoisePSDaLIGOZeroDetHighPower for d in data_dict}
TVALS = np.arange(int(2 * 0.03 / deltaT)) * deltaT - 0.03


def _P_vec(K=48, seed=71):
    rng = np.random.RandomState(seed)
    Pv = Psig.manual_copy()
    Pv.phi = rng.uniform(0, 2 * np.pi, K)
    Pv.theta = np.arcsin(rng.uniform(-1, 1, K))
    Pv.psi = rng.uniform(0, np.pi, K)
    Pv.incl = np.arccos(rng.uniform(-1, 1, K))
    Pv.phiref = rng.uniform(0, 2 * np.pi, K)
    Pv.dist = (rng.uniform(100, 800, K) * 1e6 * lsu.lsu_PC)
    Pv.tref = float(event_time); Pv.deltaT = deltaT
    return Pv


def _distMpc(Pv):
    return np.asarray(Pv.dist) / (PC * 1e6)


def _finite_diff_grad(fn, x0, h=1e-4):
    """Central-difference gradient of a scalar fn at vector x0."""
    g = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        g[i] = (fn(xp) - fn(xm)) / (2 * h)
    return g


def check_rotation(p_max=0):
    """Gate (a) for the rotation bank at the given ``p_max``.

    p_max=0 is Path A (amplitude drift only, a=(0,n)); p_max>=1 is Path B, which adds the
    delay-derivative bands a=(p,n).  Path B is not a cosmetic extension of this port: several
    ``p`` then share the same sidereal harmonic ``n``, so the post-phase buckets
    (m = n_a' - n_a) collect (a,a') pairs from DIFFERENT p -- 4-20 pairs per bucket at
    p_max=1 vs 1-5 at p_max=0 -- and the V-term reflection (p,n)->(p,-n) has to resolve
    within p.  Neither branch is exercised at p_max=0.
    """
    print("\n=== ROTATION (Path %s, p_max=%d) ===" % ("A" if p_max == 0 else "B", p_max))
    ri, ct, ctV, rho, meta = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        harmonics=HARM, p_max=p_max, f_sidereal=flwr.F_SIDEREAL, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)
    # NOT `len(HARM)`: the precompute widens the requested harmonics to
    # |ntilde| <= 2 + p_max, because that is what rotation_coefficients actually populates
    # (issue #142).  So HARM=(-2..2) gives 5 bands per p at p_max=0 but 7 at p_max=1.
    # Asserting len(HARM) here hard-coded the TRUNCATED bank and had to be corrected.
    n_bands = 2 * flwr.required_harmonic_width(p_max) + 1
    assert len(meta['harmonics']) == n_bands, \
        "harmonics not widened to 2+p_max: %s" % (meta['harmonics'],)
    assert len(meta['a_list']) == (p_max + 1) * n_bands, "unexpected a_list size"
    lk, rbn, ubn, vbn, ep = flwr.pack_rotation_arrays(meta, rho, ct, ctV)
    Pv = _P_vec()
    lnL_ref = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        TVALS, Pv, meta, lk, rbn, ubn, vbn, ep, Lmax=Lmax, time_interp='nearest',
        xpy=np)
    data = build_rotation_data(meta, lk, rbn, ubn, vbn, ep, deltaT, TVALS)
    lnL_jax = np.asarray(fused_log_likelihood(
        data, Pv.phi, Pv.theta, Pv.psi, Pv.incl, Pv.phiref, _distMpc(Pv),
        interp="nearest"))
    fin = np.isfinite(lnL_ref) & np.isfinite(lnL_jax)
    err = np.max(np.abs(lnL_ref[fin] - lnL_jax[fin]))
    rel = np.max(np.abs(lnL_ref[fin] - lnL_jax[fin]) / (1 + np.abs(lnL_ref[fin])))
    print("(a) nearest vs numpy NoLoop-with-rotation: max|abs| = %.3e  max|rel| = %.3e"
          "  (%d samples, A=%d bands)" % (err, rel, fin.sum(), len(meta['a_list'])))
    # Both sides apply the arrival-time post-phase C~_a = C_a exp(i n_a Omega (t - tref))
    # (factored_likelihood_with_rotation.rotation_post_phase) to the data term AND the model
    # norm, and the JAX accumulator uses the same arrival samples the gather uses, so this is
    # an exact algebraic identity -- only floating-point reassociation separates them.
    ROT_TOL = 1e-10
    assert rel < ROT_TOL, "rotation nearest mismatch (rel) %g at p_max=%d" % (rel, p_max)
    return data


def test_rotation_path_a():
    # check_ad as well as check_rotation: the __main__ block below runs both, and a
    # pytest entry point that ran only half of it would leave the AD/jit/vmap/hessian
    # gates uncollected -- green in CI, exercised only when someone runs the file by
    # hand.  See .travis/test-jax.sh.
    check_ad(check_rotation(p_max=0), "rotation p_max=0")


def test_rotation_path_b():
    check_ad(check_rotation(p_max=1), "rotation p_max=1")


def check_freqresponse():
    print("\n=== FREQRESPONSE (Path D, Qmax=%d, L=%.0f m) ===" % (Qmax, L_CE))
    bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        Qmax=Qmax, L_arm=L_CE, analyticPSD_Q=True, verbose=False, quiet=True,
        skip_interpolation=True)
    meta = bk[4]
    lk, rbp, ubp, vbp, ep = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])
    Pv = _P_vec()
    lnL_ref = flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
        TVALS, Pv, meta, lk, rbp, ubp, vbp, ep, Lmax=Lmax, time_interp='nearest',
        xpy=np)
    det_geom = {d: sfr.detector_geometry(d, L_arm=L_CE) for d in DETS}
    data = build_freqresponse_data(meta, lk, rbp, ubp, vbp, ep, deltaT, TVALS,
                                   det_geom)
    lnL_jax = np.asarray(fused_log_likelihood(
        data, Pv.phi, Pv.theta, Pv.psi, Pv.incl, Pv.phiref, _distMpc(Pv),
        interp="nearest"))
    fin = np.isfinite(lnL_ref) & np.isfinite(lnL_jax)
    err = np.max(np.abs(lnL_ref[fin] - lnL_jax[fin]))
    rel = np.max(np.abs(lnL_ref[fin] - lnL_jax[fin]) / (1 + np.abs(lnL_ref[fin])))
    print("(a) nearest vs numpy FreqResponse NoLoop: max|abs| = %.3e  max|rel| = %.3e"
          "  (%d samples)" % (err, rel, fin.sum()))
    assert rel < 1e-10, "freqresponse nearest mismatch (rel) %g" % rel
    return data


def test_freqresponse():
    check_ad(check_freqresponse(), "freqresponse")


def check_ad(data, tag):
    print("--- AD checks (%s) ---" % tag)
    # (c) jit + vmap of the fixed-distance likelihood
    f = jax.jit(lambda ra, dec, psi, incl, phiref, d: fused_log_likelihood(
        data, ra, dec, psi, incl, phiref, d, interp="linear"))
    th = (jnp.array([1.0]), jnp.array([0.2]), jnp.array([0.4]),
          jnp.array([0.9]), jnp.array([1.1]), jnp.array([300.0]))
    v = np.asarray(f(*th))
    assert np.all(np.isfinite(v)), "jit likelihood non-finite"
    print("(c) jit fused_log_likelihood finite: lnL = %.4f" % v[0])

    # (b,c) distance-marginalized: grad vs finite diff, hessian finite
    dlike = JAXDistanceMarginalizedLikelihood(data, 5.0, 3000.0, n_grid=128,
                                              interp="linear")
    x0 = np.array([1.0, 0.2, 0.4, 0.9, 1.1])
    val, grad = dlike.value_and_grad(x0)
    fd = _finite_diff_grad(lambda x: dlike.value(x), x0, h=1e-4)
    rel = np.max(np.abs(grad - fd) / (1 + np.abs(fd)))
    print("(b) distmarg grad vs finite-diff: max|rel| = %.3e" % rel)
    print("    grad     =", np.array2string(np.asarray(grad), precision=4))
    print("    fin-diff =", np.array2string(fd, precision=4))
    assert np.all(np.isfinite(grad)), "distmarg grad non-finite"
    assert rel < 1e-4, "distmarg grad disagrees with finite diff: %g" % rel
    H = dlike.fisher(x0)
    assert np.all(np.isfinite(H)), "hessian non-finite"
    print("(c) hessian finite, Fisher diag =",
          np.array2string(np.diag(H), precision=2))


if __name__ == "__main__":
    # Call the pytest entry points, not the check_* helpers, so the __main__ path and
    # the collected path cannot drift apart.
    test_rotation_path_a()
    test_rotation_path_b()
    test_freqresponse()
    print("\nSLOWROT + FREQRESPONSE JAX VALIDATION PASSED")
    print("  (agreement with the NoLoop is necessary, not sufficient: the rotation VALUE is")
    print("   pinned by test/jax/test_jax_slowrot_cauchy_schwarz.py.)")
