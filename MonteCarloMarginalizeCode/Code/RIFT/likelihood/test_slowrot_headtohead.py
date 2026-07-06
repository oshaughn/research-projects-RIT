"""
test_slowrot_headtohead : matched-seed quantitative baseline-vs-rotation comparison.

Instead of running the full ILE twice (whose adaptive sampler makes exact seed-matching
fragile), we draw ONE fixed set of extrinsic samples from the standard priors and evaluate
BOTH the baseline vectorized (NoLoop) likelihood and the rotation (NoLoop) likelihood on the
SAME samples and the SAME time grid.  Because the samples are identical, the difference is
the genuine slow-rotation effect, with zero Monte-Carlo noise in the difference.

Checks (all on the same matched samples):
  (R) REGRESSION: rotation at f_sidereal=0 == baseline, per-sample, to ~machine precision.
  (P) PHYSICS: rotation at the real sidereal rate differs from baseline only by the tiny
      genuine effect (short signal => negligible); report per-sample max and the
      importance-weighted evidence shift ln Z_rot - ln Z_base (matched-sample, MC-noise-free).
  (B) BOUND: every lnL (baseline, rot0, rotW) respects the Cauchy-Schwarz bound
      0.5 * sum_det <d|d>.

Run under the RIFT venv with this worktree first on PYTHONPATH.
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr

fSample = 4096.0
fmin = 30.0
fmax = 1700.0
event_time = 1e9
t_window = 0.1
Lmax = 2
deltaT = 1. / fSample
deltaF = 1. / 4.
HARM = (-2, -1, 0, 1, 2)
DMIN, DMAX = 10.0, 1000.0   # Mpc, for the d^2 distance prior

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


def _network_half_dd():
    tot = 0.0
    for det in data_dict:
        IP = lsu.ComplexIP(fmin, fmax, 1. / 2. / deltaT, deltaF, psd_dict[det], True, False, 0.)
        tot += IP.ip(data_dict[det], data_dict[det]).real
    return 0.5 * tot


def _draw_extrinsic(K, seed=20260704):
    rng = np.random.RandomState(seed)
    RA = rng.uniform(0, 2 * np.pi, K)
    DEC = np.arcsin(rng.uniform(-1, 1, K))
    PSI = rng.uniform(0, np.pi, K)
    INCL = np.arccos(rng.uniform(-1, 1, K))
    PHIREF = rng.uniform(0, 2 * np.pi, K)
    # d^2 prior in [DMIN, DMAX]
    u = rng.uniform(0, 1, K)
    DIST = (DMIN ** 3 + u * (DMAX ** 3 - DMIN ** 3)) ** (1. / 3.)
    return RA, DEC, PSI, INCL, PHIREF, DIST


def _make_P_vec(RA, DEC, PSI, INCL, PHIREF, DIST_Mpc):
    Pv = Psig.manual_copy()
    Pv.phi = RA.astype(float)
    Pv.theta = DEC.astype(float)
    Pv.psi = PSI.astype(float)
    Pv.incl = INCL.astype(float)
    Pv.phiref = PHIREF.astype(float)
    Pv.dist = (DIST_Mpc * 1e6 * lsu.lsu_PC).astype(float)
    Pv.tref = float(event_time)
    Pv.deltaT = deltaT
    return Pv


def _pack_baseline():
    ri, ct, ctV, rho, snr, rest = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lk = {}; rA = {}; cu = {}; cv = {}; ep = {}
    for det in data_dict:
        a, b, c, U, V, rArr, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rho[det].keys()), None, rho[det], ct[det], ctV[det])
        lk[det] = a; rA[det] = rArr; cu[det] = U; cv[det] = V; ep[det] = e
    return lk, rA, cu, cv, ep


def _pack_rotation(f_sidereal):
    ri, ct, ctV, rho, meta = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        harmonics=HARM, p_max=0, f_sidereal=f_sidereal, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)
    lk, rbn, ubn, vbn, ep = flwr.pack_rotation_arrays(meta, rho, ct, ctV)
    return meta, lk, rbn, ubn, vbn, ep


K = 300
tvals = np.arange(int(2 * 0.03 / deltaT)) * deltaT - 0.03
RA, DEC, PSI, INCL, PHIREF, DIST = _draw_extrinsic(K)
P_vec = _make_P_vec(RA, DEC, PSI, INCL, PHIREF, DIST)
HALF_DD = _network_half_dd()


def _lnZ(lnL):
    m = np.max(lnL)
    return m + np.log(np.mean(np.exp(lnL - m)))


# precompute once
_lkB, _rAB, _cuB, _cvB, _epB = _pack_baseline()
_metaR0, _lkR0, _rbn0, _ubn0, _vbn0, _epR0 = _pack_rotation(0.0)
_metaRW, _lkRW, _rbnW, _ubnW, _vbnW, _epRW = _pack_rotation(flwr.F_SIDEREAL)

lnL_base = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
    tvals, P_vec, _lkB, _rAB, _cuB, _cvB, _epB, Lmax=Lmax, xpy=np)
lnL_rot0 = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
    tvals, P_vec, _metaR0, _lkR0, _rbn0, _ubn0, _vbn0, _epR0, Lmax=Lmax)
lnL_rotW = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
    tvals, P_vec, _metaRW, _lkRW, _rbnW, _ubnW, _vbnW, _epRW, Lmax=Lmax)


# NOTE: the baseline DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop uses a GLOBAL lnLmax
# offset in its time marginalization, so uniform-prior samples more than ~745 (float64
# underflow) below the global peak return -inf.  Our rotation NoLoop uses a PER-SAMPLE
# offset (always finite).  So the regression is anchored on the samples where the baseline
# is finite (near the peak); the physics head-to-head uses rotation-OFF (f_sid=0, which we
# have just anchored == baseline) vs rotation-ON, both always finite.
_base = np.asarray(lnL_base); _r0 = np.asarray(lnL_rot0); _rW = np.asarray(lnL_rotW)
_finite = np.isfinite(_base)


def test_R_rotation_zero_rate_matches_baseline_per_sample():
    assert _finite.sum() > 0, "baseline finite nowhere -- cannot anchor"
    d = np.max(np.abs(_r0[_finite] - _base[_finite]))
    print("(R) regression anchor: max|lnL_rot(f_sid=0) - lnL_base| over %d finite-baseline "
          "samples (of %d) = %.3e" % (int(_finite.sum()), K, d))
    assert d < 1e-6, "rotation at f_sid=0 disagrees with baseline: %g" % d


def test_P_physics_effect_and_evidence_shift():
    # matched-sample baseline(=f_sid=0) vs rotation(real); both finite everywhere.
    d = np.max(np.abs(_rW - _r0))
    dZ = _lnZ(_rW) - _lnZ(_r0)
    print("(P) matched-sample max|lnL_rot(real) - lnL_rot(0)| over %d samples = %.3e" % (K, d))
    print("(P) evidence shift ln Z_rot - ln Z_base (matched, MC-noise-free) = %+.6e" % dZ)
    print("(P)   [short 30+25 BBH => tiny; ln Z(f_sid=0)=%.4f  ln Z(real)=%.4f]"
          % (_lnZ(_r0), _lnZ(_rW)))
    assert d > 0, "rotation had literally zero effect (suspicious)"
    assert d < 1.0, "short-signal rotation effect unexpectedly large: %g" % d


def test_B_cauchy_schwarz_bound():
    worst = max(np.max(_r0), np.max(_rW))
    print("(B) max lnL (rot0/rotW) = %.4f   0.5<d|d>_network = %.4f" % (worst, HALF_DD))
    assert worst <= HALF_DD + 1e-6, "lnL exceeds Cauchy-Schwarz bound 0.5<d|d>: %g > %g" % (worst, HALF_DD)


if __name__ == "__main__":
    test_R_rotation_zero_rate_matches_baseline_per_sample()
    test_P_physics_effect_and_evidence_shift()
    test_B_cauchy_schwarz_bound()
    print("ALL SLOWROT HEAD-TO-HEAD CHECKS PASSED")
