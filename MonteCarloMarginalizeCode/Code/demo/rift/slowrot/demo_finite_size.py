#!/usr/bin/env python
"""
slowrot demo / demo_finite_size.py -- verify-anywhere (no condor, no GPU) quick-look that the
finite-size (frequency-dependent detector response) likelihood [Path D] ADDS VALUE on a
long-armed 3G detector.

Beyond the long-wavelength (LWL) limit the detector strain is a per-frequency antenna
pattern, h_k(f) = F_+(f;sky) h_+(f) + F_x(f;sky) h_x(f), because the light-travel time across
a multi-km arm is no longer negligible vs the GW period.  We build an EXACT finite-size
injection (from the same modes the likelihood uses -> internal_hlm_generator -> IFFT ->
antenna_response_fd) and, at the true parameters, compare the recovered log-likelihood of

    lnL_LWL       : the standard RIFT likelihood (constant, frequency-independent response)
    lnL_finite    : the finite-size likelihood (--freqresponse, sky-harmonic route (b))

both TIME-MAXIMIZED on the SAME data and SAME time grid.  The gain

    gain(L) = lnL_finite(truth) - lnL_LWL(truth)

is ~0 for a LIGO 4-km arm (null control: the finite-size effect is below the floor) and grows
monotonically with the arm length -- i.e. with fL/c, the in-band light-crossing phase -- to
tens of nats for a 40-km Cosmic Explorer.  That is exactly the SNR the LWL analysis throws
away on a 3G detector, and what --freqresponse recovers.

The effect that matters is the DIRECTION-DEPENDENT part of the response; the common
e^{-i2pi f L/c} light-crossing delay is degenerate with the arrival time and is absorbed by
BOTH likelihoods' time maximization, so it does not inflate the gain.

Run:  python demo_finite_size.py      (writes outputs/finite_size_gain_vs_arm.{txt,png})

Backing code: branch rift_slowrot; --freqresponse (+ --freqresponse-qmax/-arm-length) on
integrate_likelihood_extrinsic_batchmode; RIFT/likelihood/factored_likelihood_freqresponse.py,
slowrot_freqresponse.py.  See RIFT/likelihood/SLOWROT_HANDOFF.md.
"""
from __future__ import print_function, division
import os
import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.slowrot_freqresponse as sfr
import RIFT.likelihood.factored_likelihood_freqresponse as flfr

EVENT_TIME = 1e9
LMAX = 2
DET = "H1"
PSD = lalsim.SimNoisePSDaLIGOZeroDetHighPower
# sky/pol/masses where the in-band finite-size effect is resolvable (heavier system -> high-f
# power; loud so the effect sits above the peak-resolution floor).
RA, DEC, PSI, INCL, PHIREF = 1.2, 0.3, 0.5, 0.4, 0.0
M1, M2 = 15.0, 13.0
FMAX = 2000.0
SCALE = 40.0            # loudness: data-mode distance = distMpcRef/SCALE (SNR ~ 320)
SEGLEN = 8.0
QMAX = 6               # response-basis order; higher needed as fL/c grows

# (label, arm-length[m]) -- LIGO 4 km is the null control; ET ~10 km; CE 20/40 km.
CONFIGS = [
    ("LIGO_4km",  4000.0),
    ("ET_10km",  10000.0),
    ("CE_20km",  20000.0),
    ("CE_40km",  40000.0),
]

C_SI = sfr.C_SI


# --- minimal FD helpers (self-contained; same conventions as the validation test) ---
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


def _wrap_fd(arr, epoch, deltaF):
    n = len(arr)
    hf = lal.CreateCOMPLEX16FrequencySeries("hf", epoch, 0., deltaF, lsu.lsu_HertzUnit, n)
    hf.data.data[:] = arr
    return hf


def _rev_td(hf):
    n = hf.data.length; dt = 1. / (n * hf.deltaF)
    ht = lal.CreateCOMPLEX16TimeSeries("h", hf.epoch, 0., dt, lal.DimensionlessUnit, n)
    lal.COMPLEX16FreqTimeFFT(ht, hf, lal.CreateReverseCOMPLEX16FFTPlan(n, 0)); return ht


def _peak(lt):
    """Time-maximized value via a fine spline over the sampled lnL(t) window."""
    from scipy.interpolate import InterpolatedUnivariateSpline
    lt = np.asarray(lt, float); x = np.arange(len(lt))
    sp = InterpolatedUnivariateSpline(x, lt, k=4)
    xs = np.linspace(0, len(lt) - 1, len(lt) * 32)
    return float(np.max(sp(xs)))


def run_config(label, L_arm):
    deltaT = 1.0 / (2 * FMAX); fmin = 30.0; deltaF = 1.0 / SEGLEN
    fNyq = 1.0 / 2.0 / deltaT; t_window = 0.1
    DLOUD = fl.distMpcRef * 1e6 * lsu.lsu_PC / SCALE

    Psig = lsu.ChooseWaveformParams(
        fmin=fmin, radec=True, incl=INCL, phiref=PHIREF, theta=DEC, phi=RA, psi=PSI,
        m1=M1 * lal.MSUN_SI, m2=M2 * lal.MSUN_SI, detector=DET, dist=200e6 * lal.PC_SI,
        deltaT=deltaT, tref=EVENT_TIME, deltaF=deltaF)
    Psig.approx = lalsim.GetApproximantFromString("IMRPhenomD")
    Pm = Psig.manual_copy(); Pm.dist = DLOUD

    # --- exact finite-size injection from the SAME modes ---
    hlms_fd, _ = fl.internal_hlm_generator(Pm, LMAX, verbose=False, quiet=True)
    hlmsT = _ifft(hlms_fd); lm0 = list(hlmsT.keys())[0]
    nn = hlmsT[lm0].data.length; dt = hlmsT[lm0].deltaT; e0 = float(hlmsT[lm0].epoch)
    Sig = np.zeros(nn, complex)
    for lm in hlmsT:
        Sig += hlmsT[lm].data.data * lal.SpinWeightedSphericalHarmonic(INCL, -PHIREF, -2, lm[0], lm[1])
    dt_geo = float(fl.ComputeArrivalTimeAtDetector(DET, RA, DEC, EVENT_TIME)) - EVENT_TIME
    data_epoch = lal.LIGOTimeGPS(e0 + EVENT_TIME)
    fvals = flfr.evaluate_fvals_from_length(nn, deltaF)
    Sig_fd = _fwd_fd(Sig, data_epoch, dt, nn).data.data
    Sig_del = _rev_td(_wrap_fd(Sig_fd * np.exp(-1j * 2 * np.pi * fvals * dt_geo), data_epoch, deltaF)).data.data
    hpf = _fwd_fd(np.real(Sig_del), data_epoch, dt, nn).data.data
    hcf = _fwd_fd(-np.imag(Sig_del), data_epoch, dt, nn).data.data
    gmst = float(lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(EVENT_TIME)))
    Fp, Fc = sfr.antenna_response_fd(DET, RA, DEC, PSI, fvals, gmst=gmst, L_arm=L_arm)
    data = _wrap_fd(Fp * hpf + Fc * hcf, data_epoch, deltaF)
    data_dict = {DET: data}; psd_dict = {DET: PSD}
    IP = lsu.ComplexIP(fmin, FMAX, fNyq, data.deltaF, PSD, True, False, 0.)
    snr = np.sqrt(IP.ip(data, data).real)

    Pv = Psig.manual_copy()
    for k, v in [('phi', RA), ('theta', DEC), ('incl', INCL), ('phiref', PHIREF),
                 ('psi', PSI), ('dist', DLOUD)]:
        setattr(Pv, k, np.ones(1) * v)
    Pv.tref = EVENT_TIME; Pv.deltaT = deltaT
    Nw = int(0.02 / deltaT); tvals = np.arange(-Nw, Nw) * deltaT

    # baseline (long-wavelength) NoLoop, time-maximized, at the truth
    ri, ct, ctV, rho, _, _ = fl.PrecomputeLikelihoodTerms(
        EVENT_TIME, t_window, Psig, data_dict, psd_dict, LMAX, FMAX,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lk = {}; rA = {}; cu = {}; cv = {}; ep = {}
    for d in data_dict:
        a, b, c, U, V, r, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rho[d].keys()), None, rho[d], ct[d], ctV[d])
        lk[d] = a; rA[d] = r; cu[d] = U; cv[d] = V; ep[d] = e
    lnL_lwl = _peak(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lk, rA, cu, cv, ep, Lmax=LMAX, xpy=np, return_lnLt=True, time_interp='cubic')[0])

    # finite-size (Path D) NoLoop, time-maximized, at the truth
    bk = flfr.PrecomputeLikelihoodTermsFreqResponse(
        EVENT_TIME, t_window, Psig, data_dict, psd_dict, LMAX, FMAX,
        Qmax=QMAX, L_arm=L_arm, analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True)
    lkf, rbp, ubp, vbp, epf = flfr.pack_freqresponse_arrays(bk[4], bk[3], bk[1], bk[2])
    lnL_fin = _peak(flfr.DiscreteFactoredLogLikelihoodFreqResponseNoLoop(
        tvals, Pv, bk[4], lkf, rbp, ubp, vbp, epf, Lmax=LMAX, array_output=True, time_interp='cubic')[0])

    fLc = FMAX * L_arm / C_SI
    return dict(label=label, L_arm=L_arm, fLc=fLc, snr=float(snr),
                lnL_lwl=lnL_lwl, lnL_fin=lnL_fin, gain=lnL_fin - lnL_lwl)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs"); os.makedirs(outdir, exist_ok=True)
    print("finite-size (Path D) value demo: %g+%g Msun, fmax=%g Hz, loudness SCALE=%g (SNR per row), Qmax=%d\n"
          % (M1, M2, FMAX, SCALE, QMAX))
    rows = []
    for cfg in CONFIGS:
        try:
            r = run_config(*cfg)
        except Exception as e:
            print("%-9s SKIPPED (%s)" % (cfg[0], e)); continue
        rows.append(r)
        print("%-9s L=%6.0fm  fL/c@fmax=%.3f  SNR=%6.1f  lnL_LWL=%11.3f lnL_finite=%11.3f  GAIN=%+8.3f"
              % (r['label'], r['L_arm'], r['fLc'], r['snr'], r['lnL_lwl'], r['lnL_fin'], r['gain']))
    if not rows:
        print("no configs succeeded"); return
    txt = os.path.join(outdir, "finite_size_gain_vs_arm.txt")
    with open(txt, "w") as f:
        f.write("# finite-size (Path D) demo: --freqresponse recovers SNR the long-wavelength analysis loses\n")
        f.write("# label L_arm[m] fL/c@fmax SNR lnL_LWL lnL_finite gain=lnL_finite-lnL_LWL\n")
        for r in rows:
            f.write("%s %g %g %g %g %g %g\n" % (r['label'], r['L_arm'], r['fLc'],
                    r['snr'], r['lnL_lwl'], r['lnL_fin'], r['gain']))
    print("\nwrote", txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Lkm = [r['L_arm'] / 1e3 for r in rows]; gain = [r['gain'] for r in rows]
        labs = [r['label'] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(Lkm, gain, 'o-', color="#9d0208")
        for x, y, l in zip(Lkm, gain, labs):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel(r"detector arm length $L$ [km]  ($\propto fL/c$, in-band light-crossing)")
        ax.set_ylabel(r"$\ln\mathcal{L}_{\rm finite}-\ln\mathcal{L}_{\rm LWL}$  (recovered)")
        ax.set_title("Finite-size detector response recovers SNR on 3G arms")
        ax.axhline(0, color="0.7", lw=0.8)
        fig.tight_layout()
        png = os.path.join(outdir, "finite_size_gain_vs_arm.png")
        fig.savefig(png, dpi=140)
        print("wrote", png)
    except Exception as e:
        print("(plot skipped:", e, ")")


if __name__ == "__main__":
    main()
