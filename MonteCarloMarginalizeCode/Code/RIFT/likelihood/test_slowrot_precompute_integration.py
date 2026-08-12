"""
Full-stack integration test for factored_likelihood_with_rotation on real waveforms/data.

Run under the RIFT venv with this worktree first on PYTHONPATH, e.g.

    source ~/RIFT_develUWM/bin/activate
    PYTHONPATH=~/RIFT_slowrot/MonteCarloMarginalizeCode/Code \
        python .../RIFT/likelihood/test_slowrot_precompute_integration.py

Inputs are built exactly like RIFT.calmarg.test_precompute_alignment (a short BBH injection,
noiseless detector data via non_herm_hoff, analytic aLIGO PSD), i.e. the same machinery the
ILE-GPU-Paper CI demo exercises.

Checks:
  V0  (recovery): PrecomputeLikelihoodTermsWithRotation(harmonics=(0,), p_max=0) reproduces
      the baseline PrecomputeLikelihoodTerms bit-for-bit -- Q, U, V.  This validates all of
      the new plumbing (mode generation, p=0 / n=0 identity, cross-term assembly).
  Vmag (magnitude sanity): running Path A (harmonics -2..2) end-to-end on real data, the
      a=(0,0) overlap still equals the baseline exactly, and the modulated a=(0,+-1) overlaps
      differ from it by a small but nonzero amount of the expected order Omega*(signal
      duration) -- confirming the sidereal modulation is applied, at the right (tiny) scale.

Full physical validation (V1: the rotation-aware lnL vs a brute-force time-varying-response
likelihood) needs the lnL assembly that contracts this bank with A_n,B_n, and is the next
step.
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr

fSample = 4096.0
fmin = 30.0
fmax = 1700.0
event_time = 1000000000.0
t_window = 0.1
Lmax = 2

Psig = lalsimutils.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI,
    detector='H1', dist=200e6 * lal.PC_SI, deltaT=1. / fSample,
    tref=event_time, deltaF=1. / 4.)

data_dict = {}
for det in ("H1", "L1", "V1"):
    P = Psig.manual_copy(); P.detector = det
    data_dict[det] = lalsimutils.non_herm_hoff(P)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}


def _maxrel(a, b):
    a = np.asarray(a); b = np.asarray(b)
    scale = max(np.max(np.abs(a)), np.max(np.abs(b)), 1e-300)
    return float(np.max(np.abs(a - b)) / scale)


def _run_base():
    # ignore_threshold=None so no mode pruning -> same mode set as the rotation path
    return fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True,
        ignore_threshold=None)


def _run_rot(harmonics, p_max):
    return flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        harmonics=harmonics, p_max=p_max, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)


def test_V0_recovers_baseline():
    _, ct_b, ctV_b, rho_b, _, _ = _run_base()
    _, ct_r, ctV_r, rho_r, meta = _run_rot(harmonics=(0,), p_max=0)
    a0 = (0, 0)
    worst_q = worst_u = worst_v = 0.0
    for det in data_dict:
        for mode in rho_b[det]:
            worst_q = max(worst_q, _maxrel(rho_r[det][a0][mode].data.data,
                                           rho_b[det][mode].data.data))
        for key in ct_b[det]:
            worst_u = max(worst_u, abs(ct_r[det][(a0, a0)][key] - ct_b[det][key]))
            worst_v = max(worst_v, abs(ctV_r[det][(a0, a0)][key] - ctV_b[det][key]))
    print("V0 recovery:  max rel|dQ|=%.2e   max|dU|=%.2e   max|dV|=%.2e"
          % (worst_q, worst_u, worst_v))
    assert worst_q < 1e-8, "Q mismatch vs baseline: %g" % worst_q
    assert worst_u < 1e-8 * (1 + worst_u), "U mismatch vs baseline: %g" % worst_u
    assert worst_v < 1e-8 * (1 + worst_v), "V mismatch vs baseline: %g" % worst_v


def test_Vmag_modulation_applied_at_right_scale():
    _, ct_b, _, rho_b, _, _ = _run_base()
    _, _, _, rho_r, meta = _run_rot(harmonics=(-2, -1, 0, 1, 2), p_max=0)
    # a=(0,0) must still equal baseline exactly even with the extra harmonics present
    worst_0 = 0.0
    for det in data_dict:
        for mode in rho_b[det]:
            worst_0 = max(worst_0, _maxrel(rho_r[det][(0, 0)][mode].data.data,
                                           rho_b[det][mode].data.data))
    print("Vmag a=(0,0) vs baseline: max rel|dQ|=%.2e" % worst_0)
    assert worst_0 < 1e-8, "modulated run corrupts the n=0 overlap: %g" % worst_0

    # modulated harmonics differ from n=0 by ~ Omega * (signal duration), tiny but nonzero.
    # Signal duration scale for a 30+25 BBH from 30 Hz is ~ 0.1-1 s -> Omega*T ~ 1e-5..1e-4.
    for n in (1, -1, 2, -2):
        worst_n = 0.0
        for det in data_dict:
            for mode in rho_b[det]:
                worst_n = max(worst_n, _maxrel(rho_r[det][(0, n)][mode].data.data,
                                               rho_r[det][(0, 0)][mode].data.data))
        print("Vmag a=(0,%+d) vs a=(0,0): max rel|dQ|=%.2e" % (n, worst_n))
        assert worst_n > 1e-9, "harmonic n=%d has no effect (modulation not applied?)" % n
        assert worst_n < 1e-2, "harmonic n=%d far too large (bug in modulation scale?): %g" % (n, worst_n)


if __name__ == "__main__":
    test_V0_recovers_baseline()
    test_Vmag_modulation_applied_at_right_scale()
    print("ALL SLOWROT PRECOMPUTE INTEGRATION CHECKS PASSED")
