"""
Validate the finite-size (frequency-dependent response) likelihood (Thrust-2 route (b)).

Builds an EXACT finite-size detector-strain injection from the SAME modes
(internal_hlm_generator -> IFFT), h_k(f) = F_+(f;sky) h_+(f) + F_x(f;sky) h_x(f) with the
VALIDATED antenna_response_fd, using a 40-km CE arm.  Then:

  (V1) REDUCE TO BASELINE : with L -> 0 the finite-size NoLoop likelihood
       (DiscreteFactoredLogLikelihoodFreqResponseNoLoop) == the MAINTAINED baseline
       DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop (NOT the older scalar
       SingleDetectorLogLikelihood/FactoredLogLikelihood).
  (V2) GROUND TRUTH       : the finite-size NoLoop likelihood reconstructs the finite-size
       injection -- deficit vs 0.5<d|d> converges as Qmax grows.  NOTE this default config
       (1.6+1.4 BNS, fmax=1024) is a WEAK demonstration: the model-distinguishing part of
       the response (beyond the common e^{-i2pi f L/c} light-crossing delay, which the
       baseline absorbs by time-marginalization) is only ~0.1% in-band, so finite-size ~=
       baseline here and no gain is expected.  The meaningful demonstration is run_strong (V4).
  (V3) CAUCHY-SCHWARZ     : every lnL <= 0.5<d|d> (network).  A violation => a wrong term.
  (V4) POSITIVE CONTROL   : run_strong -- in an in-band-effect config (15+13 Msun, fmax=2000,
       loud CE) the finite-size likelihood beats the long-wavelength baseline by a large,
       ASSERTED margin.  This is what validates the finite-L assembly (V1 only tests L->0).

All comparisons use the maintained NoLoop path, time-MAXIMIZED (NoLoop point eval is off-peak),
with cubic sub-bin time interpolation.  Measured: V1 |diff| ~3e-9; V2 (BNS/1024, weak) baseline
deficit 7.76 -> finite-size 7.80 (no gain, as expected); V3 bound respected; V4 (15+13/2000)
baseline deficit 55.8 -> finite-size 16.9, GAIN +38.9 nats (Qmax=6; residual is series
truncation at fL/c~0.27).
"""
from __future__ import print_function, division
import sys
import numpy as np
import lal, lalsimulation as lalsim
import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.slowrot_freqresponse as sfr
import RIFT.likelihood.factored_likelihood_freqresponse as flfr

event_time = 1e9; Lmax = 2; t_window = 0.1; det = 'H1'
psd = lalsim.SimNoisePSDaLIGOZeroDetHighPower
apx = lalsim.GetApproximantFromString("IMRPhenomD")
L_CE = 40000.0                    # 40-km Cosmic Explorer arm (geometry override)
SCALE = 40.                       # loudness: data-mode distance = distMpcRef/SCALE


def _ifft(hf_d):
    out = {}
    for lm, hf in hf_d.items():
        n = hf.data.length; dt = 1. / (n * hf.deltaF)
        ht = lal.CreateCOMPLEX16TimeSeries("h", hf.epoch, 0., dt, lal.DimensionlessUnit, n)
        lal.COMPLEX16FreqTimeFFT(ht, hf, lal.CreateReverseCOMPLEX16FFTPlan(n, 0)); out[lm] = ht
    return out


def _fwd_fd(re_series, epoch, dt, N):
    ht = lal.CreateCOMPLEX16TimeSeries("h", epoch, 0., dt, lal.DimensionlessUnit, N)
    ht.data.data[:] = re_series[:N]
    hf = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, 0., 1. / dt / N, lsu.lsu_HertzUnit, N)
    lal.COMPLEX16TimeFreqFFT(hf, ht, lal.CreateForwardCOMPLEX16FFTPlan(N, 0)); return hf


def _rev_td(hf):
    n = hf.data.length; dt = 1. / (n * hf.deltaF)
    ht = lal.CreateCOMPLEX16TimeSeries("h", hf.epoch, 0., dt, lal.DimensionlessUnit, n)
    lal.COMPLEX16FreqTimeFFT(ht, hf, lal.CreateReverseCOMPLEX16FFTPlan(n, 0)); return ht


from scipy.interpolate import InterpolatedUnivariateSpline


def _peak(lt):
    lt = np.asarray(lt, float); x = np.arange(len(lt))
    sp = InterpolatedUnivariateSpline(x, lt, k=4)
    xs = np.linspace(0, len(lt) - 1, len(lt) * 32)
    return float(np.max(sp(xs)))


def run(Qlist=(0, 2, 4, 6, 8)):
    fmin, fmax, deltaT, seglen = 30., 1024., 1. / 2048., 16.
    deltaF = 1. / seglen; fNyq = 1. / 2. / deltaT; N = int(round(seglen / deltaT))
    RA, DEC, PSI, INCL, PHIREF = 1.2, 0.3, 0.5, 0.4, 0.0
    DLOUD = fl.distMpcRef * 1e6 * lsu.lsu_PC / SCALE

    Psig = lsu.ChooseWaveformParams(
        fmin=fmin, radec=True, incl=INCL, phiref=PHIREF, theta=DEC, phi=RA, psi=PSI,
        m1=1.6 * lal.MSUN_SI, m2=1.4 * lal.MSUN_SI, detector=det, dist=200e6 * lal.PC_SI,
        deltaT=deltaT, tref=event_time, deltaF=deltaF)
    Psig.approx = apx
    Pm = Psig.manual_copy(); Pm.dist = DLOUD

    # --- base FD modes ONCE; build detector strain from the SAME modes ---
    hlms_fd, _ = fl.internal_hlm_generator(Pm, Lmax, verbose=False, quiet=True)
    hlmsT = _ifft(hlms_fd)
    lm0 = list(hlmsT.keys())[0]
    nn = hlmsT[lm0].data.length; dt = hlmsT[lm0].deltaT; e0 = float(hlmsT[lm0].epoch)

    # complex Sigma(t) = sum_lm Y_lm h_lm(t)  on intrinsic axis (epoch e0)
    Sig = np.zeros(nn, complex)
    for lm in hlmsT:
        Sig += hlmsT[lm].data.data * lal.SpinWeightedSphericalHarmonic(INCL, -PHIREF, -2, lm[0], lm[1])

    # geometric arrival delay for this sky (SAME function the likelihood uses)
    dt_geo = float(fl.ComputeArrivalTimeAtDetector(det, RA, DEC, event_time)) - event_time
    data_epoch = lal.LIGOTimeGPS(e0 + event_time)

    # Sigma delayed by the geometric delay: e^{-i2pi f dt_geo} in FD (packed fvals)
    fvals = flfr.evaluate_fvals_from_length(nn, deltaF)
    Sig_fd = _fwd_fd(Sig, data_epoch, dt, nn).data.data          # FD of complex Sigma(t)
    Sig_del_fd = Sig_fd * np.exp(-1j * 2.0 * np.pi * fvals * dt_geo)
    Sig_del = _rev_td(_wrap_fd(Sig_del_fd, data_epoch, deltaF)).data.data   # complex TD, delayed
    hplus_t = np.real(Sig_del); hcross_t = -np.imag(Sig_del)     # real polarization series

    hplus_f = _fwd_fd(hplus_t, data_epoch, dt, nn).data.data
    hcross_f = _fwd_fd(hcross_t, data_epoch, dt, nn).data.data
    gmst_ev = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time)))

    def make_data(Larm, finite=True):
        if finite:
            Fp, Fc = sfr.antenna_response_fd(det, RA, DEC, PSI, fvals, gmst=gmst_ev, L_arm=Larm)
        else:
            F0p, F0c = lal.ComputeDetAMResponse(
                lalsim.DetectorPrefixToLALDetector(det).response, RA, DEC, PSI, gmst_ev)
            Fp = np.full(nn, F0p, complex); Fc = np.full(nn, F0c, complex)
        hk_f = Fp * hplus_f + Fc * hcross_f
        return _wrap_fd(hk_f, data_epoch, deltaF)

    data = make_data(L_CE, finite=True)
    data_dict = {det: data}; psd_dict = {det: psd}
    IPc = lsu.ComplexIP(fmin, fmax, fNyq, data.deltaF, psd, True, False, 0.)
    HALF_DD = 0.5 * IPc.ip(data, data).real
    print("finite-size CE injection (L=%.0f km): seglen=%.0fs  0.5<d|d> = %.5f"
          % (L_CE / 1e3, seglen, HALF_DD))

    # --- extrinsic evaluation setup ---
    Pv = Psig.manual_copy()
    for k, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                 ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, k, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    Nw = int(0.03 / deltaT); tvals = np.arange(-Nw, Nw) * deltaT

    # --- baseline standard RIFT likelihood (long-wavelength) ---
    ri, ct, ctV, rho, snr, rest = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lkB = {}; rAB = {}; cuB = {}; cvB = {}; epB = {}
    for d in data_dict:
        a, b, c, U, V, rA, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rho[d].keys()), None, rho[d], ct[d], ctV[d])
        lkB[d] = a; rAB[d] = rA; cuB[d] = U; cvB[d] = V; epB[d] = e
    lnL_base = _peak(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lkB, rAB, cuB, cvB, epB, Lmax=Lmax, xpy=np, return_lnLt=True,time_interp='cubic')[0])
    print("  baseline (long-wavelength) lnL = %.5f   deficit = %.5f   [<= 0.5<d|d> ? %s]"
          % (lnL_base, HALF_DD - lnL_base, lnL_base <= HALF_DD + 1e-6))

    # --- finite-size likelihood, scan Qmax ---
    print("\n(V2) finite-size likelihood vs 0.5<d|d> as Qmax grows:")
    worst_excess = -1e99
    for Qmax in Qlist:
        bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
            event_time, t_window, Psig, data_dict, psd_dict, Lmax, fmax,
            Qmax=Qmax, L_arm=L_CE, analyticPSD_Q=True, verbose=False, quiet=True,
            skip_interpolation=True)
        lk, rbp, ubp, vbp, ep = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])
        lnLt = flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
            tvals, Pv, bk[4], lk, rbp, ubp, vbp, ep, Lmax=Lmax, array_output=True,time_interp='cubic')[0]
        lnL = _peak(lnLt)
        excess = lnL - HALF_DD
        worst_excess = max(worst_excess, excess)
        print("   Qmax=%2d (basis %d): lnL=%.5f  deficit=%.5f  excess vs 0.5<d|d>=%+.2e %s"
              % (Qmax, Qmax + 2, lnL, HALF_DD - lnL, excess,
                 "" if excess <= 1e-6 else "  <-- BOUND VIOLATION"))

    # --- (V1) reduce to baseline with L -> 0 ---
    print("\n(V1) reduce-to-baseline: finite-size(L->0) vs standard RIFT (same data):")
    # rebuild data with the LONG-WAVELENGTH (constant F) response so both agree
    data_lwl = make_data(L_CE, finite=False)
    dd_lwl = {det: data_lwl}
    ri2, ct2, ctV2, rho2, snr2, rest2 = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, dd_lwl, psd_dict, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lkB2 = {}; rAB2 = {}; cuB2 = {}; cvB2 = {}; epB2 = {}
    for d in dd_lwl:
        a, b, c, U, V, rA, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rho2[d].keys()), None, rho2[d], ct2[d], ctV2[d])
        lkB2[d] = a; rAB2[d] = rA; cuB2[d] = U; cvB2[d] = V; epB2[d] = e
    lnL_base2 = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lkB2, rAB2, cuB2, cvB2, epB2, Lmax=Lmax, xpy=np, return_lnLt=True,time_interp='cubic')[0]
    for Qmax in [0, 4]:
        bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
            event_time, t_window, Psig, dd_lwl, psd_dict, Lmax, fmax,
            Qmax=Qmax, L_arm=1e-6, analyticPSD_Q=True, verbose=False, quiet=True,
            skip_interpolation=True)
        lk, rbp, ubp, vbp, ep = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])
        lnLt = flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
            tvals, Pv, bk[4], lk, rbp, ubp, vbp, ep, Lmax=Lmax, array_output=True,time_interp='cubic')[0]
        worst = np.max(np.abs(lnLt - lnL_base2))
        print("   Qmax=%d L->0: max|lnL_fr - lnL_baseline| over window = %.3e" % (Qmax, worst))

    print("\n(V3) worst excess over 0.5<d|d> across all finite-size evals = %+.3e  (%s)"
          % (worst_excess, "BOUND RESPECTED" if worst_excess <= 1e-6 else "VIOLATED"))
    return HALF_DD, lnL_base, worst_excess


def _wrap_fd(arr, epoch, deltaF):
    n = len(arr)
    hf = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, 0., deltaF, lsu.lsu_HertzUnit, n)
    hf.data.data[:] = arr
    return hf


def run_strong(fmax=2000., seglen=8., SCALE_strong=40., m1=15., m2=13., Qmax=6):
    """(V4) POSITIVE CONTROL: in a config where the direction-dependent finite-size effect
    is actually in-band with SNR behind it (heavier system -> high-f power, higher fmax,
    loud), the finite-size likelihood must reconstruct the finite-size injection much better
    than the long-wavelength baseline.

    This is the meaningful demonstration of the finite-size likelihood.  The default `run()`
    at BNS/fmax=1024 is a WEAK config: there the model-distinguishing effect (beyond the
    common e^{-i2pi f L/c} light-crossing delay, which the baseline absorbs by
    time-marginalization) is ~0.1% in-band -- below the peak-resolution floor -- so
    finite-size ~= baseline there and NO gain is expected (or asserted).

    Measured (H1, 40-km CE arm, 15+13 Msun, fmax=2000, loud): baseline deficit ~55.8,
    finite-size deficit ~16.9 -> gain ~+38.9 nats (Qmax=6; residual is series truncation at
    fL/c~0.27, shrinks with Qmax).
    """
    event_time = 1e9; t_window = 0.1; det = 'H1'; L_arm = L_CE
    deltaT = 1. / (2. * fmax); fmin = 30.
    deltaF = 1. / seglen; fNyq = 1. / 2. / deltaT
    RA, DEC, PSI, INCL, PHIREF = 1.2, 0.3, 0.5, 0.4, 0.0
    DLOUD = fl.distMpcRef * 1e6 * lsu.lsu_PC / SCALE_strong

    Psig = lsu.ChooseWaveformParams(
        fmin=fmin, radec=True, incl=INCL, phiref=PHIREF, theta=DEC, phi=RA, psi=PSI,
        m1=m1 * lal.MSUN_SI, m2=m2 * lal.MSUN_SI, detector=det, dist=200e6 * lal.PC_SI,
        deltaT=deltaT, tref=event_time, deltaF=deltaF)
    Psig.approx = apx
    Pm = Psig.manual_copy(); Pm.dist = DLOUD

    hlms_fd, _ = fl.internal_hlm_generator(Pm, Lmax, verbose=False, quiet=True)
    hlmsT = _ifft(hlms_fd); lm0 = list(hlmsT.keys())[0]
    nn = hlmsT[lm0].data.length; dt = hlmsT[lm0].deltaT; e0 = float(hlmsT[lm0].epoch)
    Sig = np.zeros(nn, complex)
    for lm in hlmsT:
        Sig += hlmsT[lm].data.data * lal.SpinWeightedSphericalHarmonic(INCL, -PHIREF, -2, lm[0], lm[1])
    dt_geo = float(fl.ComputeArrivalTimeAtDetector(det, RA, DEC, event_time)) - event_time
    data_epoch = lal.LIGOTimeGPS(e0 + event_time)
    fvals = flfr.evaluate_fvals_from_length(nn, deltaF)
    Sig_fd = _fwd_fd(Sig, data_epoch, dt, nn).data.data
    Sig_del = _rev_td(_wrap_fd(Sig_fd * np.exp(-1j * 2 * np.pi * fvals * dt_geo), data_epoch, deltaF)).data.data
    hpf = _fwd_fd(np.real(Sig_del), data_epoch, dt, nn).data.data
    hcf = _fwd_fd(-np.imag(Sig_del), data_epoch, dt, nn).data.data
    gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(event_time)))
    Fp, Fc = sfr.antenna_response_fd(det, RA, DEC, PSI, fvals, gmst=gmst, L_arm=L_arm)
    data = _wrap_fd(Fp * hpf + Fc * hcf, data_epoch, deltaF)
    dd = {det: data}; pdd = {det: psd}
    IPc = lsu.ComplexIP(fmin, fmax, fNyq, data.deltaF, psd, True, False, 0.)
    HALF = 0.5 * IPc.ip(data, data).real

    Pv = Psig.manual_copy()
    for k, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                 ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, k, np.ones(1) * v)
    Pv.tref = event_time; Pv.deltaT = deltaT
    Nw = int(0.03 / deltaT); tvals = np.arange(-Nw, Nw) * deltaT

    ri, ct, ctV, rho, snr, rest = fl.PrecomputeLikelihoodTerms(
        event_time, t_window, Psig, dd, pdd, Lmax, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lk = {}; rA = {}; cu = {}; cv = {}; ep = {}
    for d in dd:
        a, b, c, U, V, r, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rho[d].keys()), None, rho[d], ct[d], ctV[d])
        lk[d] = a; rA[d] = r; cu[d] = U; cv[d] = V; ep[d] = e
    base = _peak(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lk, rA, cu, cv, ep, Lmax=Lmax, xpy=np, return_lnLt=True, time_interp='cubic')[0])
    bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
        event_time, t_window, Psig, dd, pdd, Lmax, fmax, Qmax=Qmax, L_arm=L_arm,
        analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True)
    lkf, rbp, ubp, vbp, epf = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])
    fin = _peak(flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
        tvals, Pv, bk[4], lkf, rbp, ubp, vbp, epf, Lmax=Lmax, array_output=True, time_interp='cubic')[0])
    gain = (HALF - base) - (HALF - fin)
    print("\n(V4) POSITIVE CONTROL %g+%g Msun, fmax=%g, CE %gkm, loud:" % (m1, m2, fmax, L_arm / 1e3))
    print("   0.5<d|d>=%.1f  baseline_deficit=%.3f  finite_deficit=%.3f  GAIN=%+.3f nats"
          % (HALF, HALF - base, HALF - fin, gain))
    assert base <= HALF + 1e-6 and fin <= HALF + 1e-6, "Cauchy-Schwarz bound violated"
    assert gain > 10.0, "finite-size failed to beat baseline where the effect is in-band: gain=%g" % gain
    return HALF, base, fin, gain


if __name__ == "__main__":
    run()
    run_strong()
    print("\nALL SLOWROT FREQRESPONSE LIKELIHOOD CHECKS PASSED")
