"""
test_slowrot_gpu : GPU vs CPU consistency for the rotation-aware NoLoop likelihood.

Runs DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation with xpy=numpy and with
xpy=cupy on the SAME packed data (moving the Q banks + U/V to device, exactly as the ILE does
under --gpu) and asserts they agree to ~1e-8.  SKIPPED if cupy / a GPU is unavailable.

The GPU term1 reuses the baseline fused Q_inner_product kernel per elementary template (no
(n_ex,npts,n_lms) temporary), so this also exercises that path.  Run on a GPU node:
    python RIFT/likelihood/test_slowrot_gpu.py
"""
from __future__ import print_function, division
import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr

if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

from RIFT.likelihood._gpu_test_support import skip_without_gpu

try:
    import cupy
    _ = cupy.array(1.0) + 1.0    # force a real device op
    HAVE_GPU = True
except Exception as e:
    HAVE_GPU = False
    _WHY = str(e)

fSample = 4096.0; fmin = 30.0; fmax = 1700.0; event_time = 1e9
t_window = 0.1; Lmax = 2; deltaT = 1. / fSample; deltaF = 1. / 4.
HARM = (-2, -1, 0, 1, 2)

Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI, detector='H1',
    dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=event_time, deltaF=deltaF)
data_dict = {det: lsu.non_herm_hoff(_p) for det, _p in
             ((d, (lambda P, dd: (setattr(P, 'detector', dd), P)[1])(Psig.manual_copy(), d))
              for d in ("H1", "L1", "V1"))}
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}


def _P_vec(K=200, seed=71):
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


def _to_gpu(rho_by_a, U_by_aa, V_by_aa):
    r = {d: {a: cupy.asarray(rho_by_a[d][a]) for a in rho_by_a[d]} for d in rho_by_a}
    u = {d: {p: cupy.asarray(U_by_aa[d][p]) for p in U_by_aa[d]} for d in U_by_aa}
    v = {d: {p: cupy.asarray(V_by_aa[d][p]) for p in V_by_aa[d]} for d in V_by_aa}
    return r, u, v


def test_gpu_matches_cpu():
    if not HAVE_GPU:
        if skip_without_gpu(HAVE_GPU, _WHY): return
    ri, ct, ctV, rho, meta = flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        harmonics=HARM, p_max=0, f_sidereal=flwr.F_SIDEREAL, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)
    lk, rbn, ubn, vbn, ep = flwr.pack_rotation_arrays(meta, rho, ct, ctV)
    Pv = _P_vec()
    tvals = np.arange(int(2 * 0.03 / deltaT)) * deltaT - 0.03
    for interp in ('nearest', 'cubic', 'sinc'):
        lnL_cpu = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
            tvals, Pv, meta, lk, rbn, ubn, vbn, ep, Lmax=Lmax, time_interp=interp, xpy=np)
        rG, uG, vG = _to_gpu(rbn, ubn, vbn)
        lnL_gpu = flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
            cupy.asarray(tvals), Pv, meta, lk, rG, uG, vG, ep, Lmax=Lmax, time_interp=interp, xpy=cupy)
        lnL_gpu = cupy.asnumpy(lnL_gpu)
        d = np.max(np.abs(np.asarray(lnL_cpu) - lnL_gpu))
        print("(GPU) rotation NoLoop xpy=cupy vs xpy=np, interp=%-7s : max|diff| = %.3e" % (interp, d))
        assert d < 1e-8, "GPU rotation disagrees with CPU (%s): %g" % (interp, d)


if __name__ == "__main__":
    test_gpu_matches_cpu()
    print("SLOWROT GPU CHECK DONE" if HAVE_GPU else "SLOWROT GPU CHECK SKIPPED (no GPU)")
