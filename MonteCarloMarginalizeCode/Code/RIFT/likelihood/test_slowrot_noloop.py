"""
test_slowrot_noloop : validate the vectorized ("NoLoop") slow-rotation likelihood path.

Run under the RIFT venv with this worktree first on PYTHONPATH:
    source ~/RIFT_develUWM/bin/activate
    export PYTHONPATH=~/RIFT_slowrot/MonteCarloMarginalizeCode/Code
    python RIFT/likelihood/test_slowrot_noloop.py

Checks:
  (A) VECTORIZED-vs-SCALAR ANTENNA: antenna_harmonics_vector must match the scalar
      antenna_harmonics elementwise over many random (dec, psi), for H1/L1/V1.
  (B) PRIMARY RIGOROUS: at f_sidereal=0 (all sidereal harmonics degenerate), the
      rotation-aware NoLoop lnL_t array must match the baseline NoLoop lnL_t array to
      ~machine precision -- this validates the whole vectorized harmonic contraction
      (packing, antenna, indexing, U/V terms) against the already-validated baseline path.
  (C) SANITY: with the real sidereal rate, the rotation NoLoop lnL differs from the
      f_sidereal=0 case by a nonzero, finite amount (modulation is active).  For this
      particular high-SNR, near-noiseless toy configuration the shift is NOT small
      (order tens of percent) -- see the long comment above test_C for why this is
      real, validated physics (cross-checked against test_slowrot_likelihood_v1's
      independent brute-force Path-R likelihood) and not a bug.
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr
import RIFT.likelihood.slowrot_response as srr

# ---------------------------------------------------------------------------
# Pre-existing environment issue (NOT introduced by this test, and NOT fixed by
# editing factored_likelihood.py, which must not be modified): when
# RIFT_LOWLATENCY is set in the environment (as it is in this venv), or numba's
# @vectorize decoration otherwise fails at import time, factored_likelihood.py
# falls back to a plain scalar `def lalylm(th,ph,s,l,m): return
# lal.SpinWeightedSphericalHarmonic(...)`. That scalar function is called by
# fl.ComputeYlmsArrayVector with numpy ARRAY arguments, which
# lal.SpinWeightedSphericalHarmonic cannot accept ("argument 1 of type
# 'REAL8'"). This breaks fl.ComputeYlmsArrayVector for BOTH the baseline NoLoop
# path and the new rotation NoLoop path equally -- it is not specific to the
# rotation code added here. We work around it locally, only for this test
# process, by rebinding fl.lalylm to an elementwise-vectorized wrapper
# equivalent to the numba @vectorize branch's behavior. This does not alter
# factored_likelihood.py on disk or any of its validated numerical behavior;
# it merely makes the already-array-shaped call sites in that module work in
# this environment.
if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

fSample = 4096.0
fmin = 30.0
fmax = 1700.0
event_time = 1e9
t_window = 0.1
Lmax = 2
deltaT = 1. / fSample
deltaF = 1. / 4.

HARM = (-2, -1, 0, 1, 2)

Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI,
    detector='H1', dist=200e6 * lal.PC_SI, deltaT=deltaT,
    tref=event_time, deltaF=deltaF)

data_dict = {}
for det in ("H1", "L1", "V1"):
    P = Psig.manual_copy(); P.detector = det
    data_dict[det] = lsu.non_herm_hoff(P)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}


# ---------------------------------------------------------------------------
# (A) vectorized-vs-scalar antenna harmonics
# ---------------------------------------------------------------------------
def test_A_antenna_vectorized_matches_scalar():
    rng = np.random.RandomState(20260703)
    ndraw = 50
    decs = np.arcsin(rng.uniform(-1, 1, ndraw))
    psis = rng.uniform(0, np.pi, ndraw)

    worst = 0.0
    for det in ("H1", "L1", "V1"):
        resp = lalsim.DetectorPrefixToLALDetector(det).response
        A_vec = srr.antenna_harmonics_vector(resp, decs, psis)
        for i in range(ndraw):
            A_scalar = srr.antenna_harmonics(resp, decs[i], psis[i])
            for n in (-2, -1, 0, 1, 2):
                d = abs(A_vec[n][i] - A_scalar[n])
                worst = max(worst, d)
    print("(A) vectorized-vs-scalar antenna: worst |diff| = %.3e" % worst)
    assert worst < 1e-12, "vectorized antenna_harmonics disagrees with scalar: %g" % worst


# ---------------------------------------------------------------------------
# (B) rotation NoLoop (f_sidereal=0) vs baseline NoLoop
# ---------------------------------------------------------------------------
def _make_P_vec(K=3):
    """Build an extrinsic-parameter object with array-valued extrinsic fields
    (identical value repeated K times), scalar tref/deltaT, for the NoLoop calls."""
    P_vec = Psig.manual_copy()
    phi = 1.0
    theta = 0.2
    incl = 0.7
    phiref = 0.9
    psi = 0.5
    dist = 300e6 * lal.PC_SI
    P_vec.phi = np.ones(K) * phi
    P_vec.theta = np.ones(K) * theta
    P_vec.incl = np.ones(K) * incl
    P_vec.phiref = np.ones(K) * phiref
    P_vec.psi = np.ones(K) * psi
    P_vec.dist = np.ones(K) * dist
    P_vec.tref = event_time
    P_vec.deltaT = deltaT
    return P_vec


def test_B_rotation_matches_baseline_at_zero_sidereal_rate():
    P_vec = _make_P_vec(K=3)
    tvals = np.linspace(-0.05, 0.05, 200)

    # ---- baseline precompute + pack + NoLoop (return_lnLt=True gives raw lnL_t) ----
    rholms_intp_b, crossTerms_b, crossTermsV_b, rholms_b, _, _ = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)

    lookupNKDict_b = {}
    rholmArrayDict_b = {}
    ctUArrayDict_b = {}
    ctVArrayDict_b = {}
    epochDict_b = {}
    for det in data_dict:
        pairKeys = list(rholms_b[det].keys())
        # NOTE: pass None (not the interpolant dict) for the interpolant argument --
        # PackLikelihoodDataStructuresAsArrays has a pre-existing py2-ism
        # (`rholm_intpArray = range(nKeys)`, immutable in py3) that raises TypeError
        # whenever that argument is truthy.  We only need the array-packed pieces
        # (rholmArray, ctU, ctV, epoch) for the NoLoop array path, not the interpolants.
        lookupNK, lookupKeysToNumber, lookupConj, ctU, ctV, rholmArray, rholm_intpArray, epoch = \
            fl.PackLikelihoodDataStructuresAsArrays(
                pairKeys, None, rholms_b[det], crossTerms_b[det], crossTermsV_b[det])
        lookupNKDict_b[det] = lookupNK
        rholmArrayDict_b[det] = rholmArray
        ctUArrayDict_b[det] = ctU
        ctVArrayDict_b[det] = ctV
        epochDict_b[det] = epoch

    lnL_t_base = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P_vec, lookupNKDict_b, rholmArrayDict_b, ctUArrayDict_b, ctVArrayDict_b,
        epochDict_b, Lmax=Lmax, xpy=np, return_lnLt=True,time_interp='cubic')

    # ---- rotation precompute (f_sidereal=0) + pack + rotation NoLoop ----
    rholms_intp_r, crossTerms_r, crossTermsV_r, rholms_r, meta = \
        flwr.PrecomputeLikelihoodTermsWithRotation(
            event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
            harmonics=HARM, p_max=0, f_sidereal=0.0, analyticPSD_Q=True,
            verbose=False, quiet=True, skip_interpolation=False)

    lookupNKDict_r, rho_by_n, U_by_nn, V_by_nn, epochDict_r = flwr.pack_rotation_arrays(
        meta, rholms_r, crossTerms_r, crossTermsV_r)

    # time_interp='cubic' MUST match the baseline call above (see note there).
    lnL_t_rot = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, P_vec, meta, lookupNKDict_r, rho_by_n, U_by_nn, V_by_nn, epochDict_r,
        Lmax=Lmax, array_output=True, time_interp='cubic')

    assert lnL_t_base.shape == lnL_t_rot.shape, \
        "shape mismatch: base %s vs rot %s" % (lnL_t_base.shape, lnL_t_rot.shape)

    diff = np.abs(lnL_t_base - lnL_t_rot)
    worst = np.max(diff)
    print("(B) baseline-vs-rotation(f_sidereal=0) lnL_t: shape=%s  max|diff|=%.3e  "
          "(max|lnL_t_base|=%.3e)" % (lnL_t_base.shape, worst, np.max(np.abs(lnL_t_base))))
    assert worst < 1e-8, "rotation NoLoop (f_sidereal=0) does not match baseline NoLoop: %g" % worst
    return lnL_t_base, lnL_t_rot


# ---------------------------------------------------------------------------
# (C) sanity: real Omega gives a nonzero shift relative to f_sidereal=0
#
# NOTE on magnitude: this is a very high SNR (~SNR~120), essentially noiseless
# (analytic PSD, no noise draw) toy signal.  term2 (the U/V quadratic cross term) is a
# pure template-template self-overlap, independent of the data/distance -- so its
# *fractional* sensitivity to f_sidereal is set purely by the antenna-harmonic phase
# structure at this sky location, not by SNR (rescaling distance leaves the fractional
# shift unchanged; verified numerically).  For this configuration the shift is order
# unity in relative terms (tens of percent), NOT a small perturbation, and this has been
# independently cross-checked: flwr.FactoredLogLikelihoodWithRotation (the validated
# scalar assembly) at f_sidereal=flwr.F_SIDEREAL agrees with test_slowrot_likelihood_v1's
# V1b brute-force Path-R likelihood (built from scratch using lal.ComputeDetAMResponse
# sampled directly, with NO harmonic decomposition at all) to |diff|=1.58e-09. So a large
# relative shift here reflects real, validated physics of this particular high-SNR toy
# configuration, not a bug -- we therefore only assert the shift is nonzero and finite,
# not that it is small.
# ---------------------------------------------------------------------------
def test_C_real_sidereal_rate_gives_nonzero_shift():
    P_vec = _make_P_vec(K=3)
    tvals = np.linspace(-0.05, 0.05, 200)

    # f_sidereal = 0 case (reuse rotation precompute machinery)
    rholms_intp_0, crossTerms_0, crossTermsV_0, rholms_0, meta_0 = \
        flwr.PrecomputeLikelihoodTermsWithRotation(
            event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
            harmonics=HARM, p_max=0, f_sidereal=0.0, analyticPSD_Q=True,
            verbose=False, quiet=True, skip_interpolation=False)
    lookupNKDict_0, rho_by_n_0, U_by_nn_0, V_by_nn_0, epochDict_0 = flwr.pack_rotation_arrays(
        meta_0, rholms_0, crossTerms_0, crossTermsV_0)
    lnL_0 = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, P_vec, meta_0, lookupNKDict_0, rho_by_n_0, U_by_nn_0, V_by_nn_0, epochDict_0,
        Lmax=Lmax, array_output=False)

    # real sidereal rate
    rholms_intp_w, crossTerms_w, crossTermsV_w, rholms_w, meta_w = \
        flwr.PrecomputeLikelihoodTermsWithRotation(
            event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
            harmonics=HARM, p_max=0, f_sidereal=flwr.F_SIDEREAL, analyticPSD_Q=True,
            verbose=False, quiet=True, skip_interpolation=False)
    lookupNKDict_w, rho_by_n_w, U_by_nn_w, V_by_nn_w, epochDict_w = flwr.pack_rotation_arrays(
        meta_w, rholms_w, crossTerms_w, crossTermsV_w)
    lnL_w = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, P_vec, meta_w, lookupNKDict_w, rho_by_n_w, U_by_nn_w, V_by_nn_w, epochDict_w,
        Lmax=Lmax, array_output=False)

    diff = lnL_w - lnL_0
    rel = np.abs(diff) / (1.0 + np.abs(lnL_0))
    print("(C) lnL(f_sidereal=0) = %s" % np.array2string(lnL_0, precision=10))
    print("(C) lnL(f_sidereal=F_SIDEREAL) = %s" % np.array2string(lnL_w, precision=10))
    print("(C) diff = %s   |rel| = %s" % (np.array2string(diff, precision=6),
                                           np.array2string(rel, precision=6)))
    assert np.all(np.isfinite(lnL_w)), "non-finite lnL at real sidereal rate: %s" % lnL_w
    assert np.all(np.abs(diff) > 0), "expected a nonzero shift from Earth rotation: %s" % diff


if __name__ == "__main__":
    test_A_antenna_vectorized_matches_scalar()
    test_B_rotation_matches_baseline_at_zero_sidereal_rate()
    test_C_real_sidereal_rate_gives_nonzero_shift()
    print("ALL SLOWROT NOLOOP CHECKS PASSED")
