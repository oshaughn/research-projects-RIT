"""
V1: validate the rotation-aware log-likelihood assembly against (a) the baseline in the
no-rotation limit and (b) an independent brute-force time-varying-response likelihood.

Run under the RIFT venv with this worktree first on PYTHONPATH:
    source ~/RIFT_develUWM/bin/activate
    PYTHONPATH=~/RIFT_slowrot/MonteCarloMarginalizeCode/Code \
        python .../RIFT/likelihood/test_slowrot_likelihood_v1.py

Checks:
  V1a (assembly algebra):  with f_sidereal -> 0 the rotation lnL must equal the baseline
      FactoredLogLikelihood exactly (all modulations become identity, sum_n A_tilde_n ->
      F(tref)).  This validates the whole harmonic contraction incl. the V-term's A_{-nu}.
  V1b (harmonic decomposition):  with the real sidereal rate, the rotation lnL must agree
      with a brute-force Path-R likelihood that applies the FULL time-varying antenna
      pattern F_k(t), sampled from lal.ComputeDetAMResponse and so independent of the A_n
      harmonic decomposition.  That is what V1b validates: that the 5-harmonic expansion
      reproduces the true F_k(t), and that Q^{(n)} is paired with the right conj(A_tilde_n).

      READ THIS BEFORE TRUSTING V1b FOR ANYTHING ELSE.  Its reference is NOT
      convention-free: _pathR_lnL pushes the modulation onto the data for term1
      (conj(F) * d, an identity that fails for a noise-weighted overlap) and samples F for
      the modes with the template pinned at event_time for term2 (no arrival-time
      post-phase).  Those are exactly the two mistakes that once made this likelihood
      exceed 0.5<d|d>; a reference that shares them cannot detect them.  V1b is therefore
      blind to the post-phase, and its tolerance (1e-4 of |lnL|, i.e. ~0.6 nats here) is far
      too loose to notice.  Since the post-phase was restored, V1b reads |diff| ~ 1.3e-3
      rather than the ~3e-9 it read while both sides were wrong -- that gap IS the
      post-phase plus the term1 commutator, not drift.

      What actually guards this: the Cauchy-Schwarz assertion on the scalar path in
      test_slowrot_pathB.py, and, for the maintained NoLoop, test_slowrot_cauchy_schwarz.py
      plus the rewritten convention-free reference in test_slowrot_noloop_bruteforce.py.
      The scalar entry point here is non-preferred -- production routes through the NoLoop
      -- so this reference was deliberately NOT rebuilt.
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
event_time = 1000000000.0
t_window = 0.1
Lmax = 2

Psig = lsu.ChooseWaveformParams(
    fmin=fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
    m1=30 * lal.MSUN_SI, m2=25 * lal.MSUN_SI,
    detector='H1', dist=200e6 * lal.PC_SI, deltaT=1. / fSample,
    tref=event_time, deltaF=1. / 4.)

data_dict = {}
for det in ("H1", "L1", "V1"):
    P = Psig.manual_copy(); P.detector = det
    data_dict[det] = lsu.non_herm_hoff(P)
psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in data_dict}

extr = lsu.ChooseWaveformParams(
    radec=True, phi=Psig.phi, theta=Psig.theta, psi=0.5, incl=0.7, phiref=0.9,
    tref=event_time, dist=300e6 * lal.PC_SI)

HARM = (-2, -1, 0, 1, 2)


def _precompute_rot(f_sidereal):
    return flwr.PrecomputeLikelihoodTermsWithRotation(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        harmonics=HARM, p_max=0, f_sidereal=f_sidereal, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=False)


def test_V1a_reduces_to_baseline():
    rint, ct, ctV, rho, meta = _precompute_rot(f_sidereal=0.0)
    lnL_rot = flwr.FactoredLogLikelihoodWithRotation(extr, rint, ct, ctV, meta, Lmax)
    rint_b, ct_b, ctV_b, rho_b, _, _ = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lnL_base = fl.FactoredLogLikelihood(extr, rho_b, rint_b, ct_b, ctV_b, Lmax)
    print("V1a: lnL_rot=%.10g  lnL_base=%.10g  |diff|=%.2e"
          % (lnL_rot, lnL_base, abs(lnL_rot - lnL_base)))
    assert abs(lnL_rot - lnL_base) < 1e-6 * (1 + abs(lnL_base)), \
        "rotation lnL does not reduce to baseline: %g vs %g" % (lnL_rot, lnL_base)


def _to_td(fs):
    npts = fs.data.length
    dt = 1.0 / (npts * fs.deltaF)
    ts = lal.CreateCOMPLEX16TimeSeries("x", fs.epoch, 0., dt, lal.DimensionlessUnit, npts)
    lal.COMPLEX16FreqTimeFFT(ts, fs, lal.CreateReverseCOMPLEX16FFTPlan(npts, 0))
    return ts


def _sample_F(det, epoch, npts, dt):
    resp = lalsim.DetectorPrefixToLALDetector(det).response
    RA, DEC, psi = extr.phi, extr.theta, extr.psi
    gmst_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time)))
    tabs = float(epoch) + np.arange(npts) * dt
    nc = 128
    tc = np.linspace(tabs[0], tabs[-1], nc)
    Fc = np.empty(nc, dtype=complex)
    for i, tt in enumerate(tc):
        gmst = gmst_ev + flwr.OMEGA_EARTH * (tt - event_time)
        fp, fx = lal.ComputeDetAMResponse(resp, RA, DEC, psi, gmst)
        Fc[i] = fp + 1j * fx
    return (np.interp(tabs, tc, Fc.real) + 1j * np.interp(tabs, tc, Fc.imag))


def _det_window(det, data, hlms):
    t_det = fl.ComputeArrivalTimeAtDetector(det, extr.phi, extr.theta, extr.tref)
    rho_epoch = data.epoch - hlms[list(hlms.keys())[0]].epoch
    t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
    N_shift = int(t_shift / Psig.deltaT + 0.5)
    N_window = int(2 * t_window / Psig.deltaT)
    t = np.arange(N_window) * Psig.deltaT + float(rho_epoch + N_shift * Psig.deltaT)
    return t_det, N_shift, N_window, t


def _pathR_lnL(modes):
    Pm = Psig.manual_copy()
    Pm.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
    Pm.deltaF = data_dict['H1'].deltaF
    hlms, hlms_conj = fl.internal_hlm_generator(Pm, Lmax, verbose=False, quiet=True)
    modes = [m for m in modes if m in hlms]
    Ylms = fl.ComputeYlms(Lmax, extr.incl, -extr.phiref, selected_modes=modes)
    distMpc = extr.dist / (lsu.lsu_PC * 1e6)
    invDistMpc = fl.distMpcRef / distMpc
    lnL = 0.
    for det in data_dict:
        data = data_dict[det]
        psd = psd_dict[det]
        npts = data.data.length
        dt = 1.0 / (npts * data.deltaF)
        fNyq = 1. / 2. / Psig.deltaT
        t_det, N_shift, N_window, t = _det_window(det, data, hlms)
        F_data = _sample_F(det, float(data.epoch), npts, dt)
        d_td = _to_td(data)
        dtld = lal.CreateCOMPLEX16TimeSeries("dF", data.epoch, 0., dt,
                                             lal.DimensionlessUnit, npts)
        dtld.data.data[:] = np.conj(F_data) * d_td.data.data
        dtld_f = lsu.DataFourier(dtld)
        rho = fl.ComputeModeIPTimeSeries(hlms, dtld_f, psd, Psig.fmin, fmax, fNyq,
                                         N_shift, N_window, True, False, 0.)
        rint = fl.InterpolateRholms(rho, t, verbose=False)
        term1 = 0.
        for m in modes:
            term1 += np.conj(Ylms[m]) * rint[m](float(t_det))
        term1 = np.real(term1) * invDistMpc
        IP = lsu.ComplexIP(Psig.fmin, fmax, fNyq, data.deltaF, psd, True, False, 0.)
        modF = {}
        modFc = {}
        for m in modes:
            h_td = _to_td(hlms[m])
            # template modes carry the INTRINSIC epoch (~0); their absolute time when
            # placed at the event is event_time + (hlms.epoch + j*dt), so sample F there.
            F_mode = _sample_F(det, event_time + float(hlms[m].epoch), hlms[m].data.length, dt)
            prod = lal.CreateCOMPLEX16TimeSeries("Fh", hlms[m].epoch, 0., dt,
                                                 lal.DimensionlessUnit, hlms[m].data.length)
            prod.data.data[:] = F_mode * h_td.data.data
            modF[m] = lsu.DataFourier(prod)
            prodc = lal.CreateCOMPLEX16TimeSeries("Fhc", hlms[m].epoch, 0., dt,
                                                  lal.DimensionlessUnit, hlms[m].data.length)
            prodc.data.data[:] = np.conj(F_mode * h_td.data.data)
            modFc[m] = lsu.DataFourier(prodc)
        term2 = 0.
        for p1 in modes:
            for p2 in modes:
                U = IP.ip(modF[p1], modF[p2])
                V = IP.ip(modFc[p1], modF[p2])
                term2 += U * np.conj(Ylms[p1]) * Ylms[p2] + V * Ylms[p1] * Ylms[p2]
        term2 = -np.real(term2) / 4. / (distMpc / fl.distMpcRef) ** 2
        lnL += term1 + term2
    return lnL


def test_V1b_matches_bruteforce_rotation():
    rint, ct, ctV, rho, meta = _precompute_rot(f_sidereal=flwr.F_SIDEREAL)
    lnL_rot = flwr.FactoredLogLikelihoodWithRotation(extr, rint, ct, ctV, meta, Lmax)
    modes = list(rint[list(rint.keys())[0]][(0, 0)].keys())
    lnL_R = _pathR_lnL(modes)
    print("V1b: lnL_rot=%.10g  lnL_pathR=%.10g  |diff|=%.2e"
          % (lnL_rot, lnL_R, abs(lnL_rot - lnL_R)))
    assert abs(lnL_rot - lnL_R) < 1e-4 * (1 + abs(lnL_R)), \
        "rotation lnL disagrees with brute force: %g vs %g" % (lnL_rot, lnL_R)


if __name__ == "__main__":
    test_V1a_reduces_to_baseline()
    test_V1b_matches_bruteforce_rotation()
    print("ALL SLOWROT V1 LIKELIHOOD CHECKS PASSED")
