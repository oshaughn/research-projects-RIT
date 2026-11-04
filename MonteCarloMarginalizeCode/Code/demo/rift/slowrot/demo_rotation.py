#!/usr/bin/env python
"""
slowrot demo / demo_rotation.py -- verify-anywhere (no condor, no GPU) quick-look that the
slow-rotation likelihood [Path A/B] ADDS VALUE: on a long signal the static
(Earth-fixed-response) likelihood LOSES match with the data (which carries the true
time-varying antenna pattern, since RIFT injections go through
SimDetectorStrainREAL8TimeSeries), and --rotation-slow recovers it.  The recovered
log-likelihood at the true parameters,

    gain(config)  =  lnL_rotation(truth)  -  lnL_static(truth),      (both time-marginalized)

grows with the rotation phase Omega*T over the signal; a short signal is a null control
(gain ~ 0).  This is the likelihood-level core of the full injection-recovery PE; the headline
sky-localization / parameter-bias figure is a cluster run.

Both likelihoods are the maintained vectorized (NoLoop) path evaluated on the SAME data and
SAME time grid, so the gain is a clean difference (peak-resolution floor cancels).

Run:  python demo_rotation.py         (writes outputs/rotation_gain_vs_duration.{txt,png})

Backing code: branch rift_slowrot; --rotation-slow (Path A) and --rotation-p-max N (Path B) on
integrate_likelihood_extrinsic_batchmode.  See RIFT/likelihood/SLOWROT_HANDOFF.md.
"""
from __future__ import print_function, division
import os
import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl
import RIFT.likelihood.factored_likelihood_with_rotation as flwr

EVENT_TIME = 1e9
LMAX = 2
DET = "H1"                    # single detector: rotation's sky-info value is clearest here
PSD = lalsim.SimNoisePSDaLIGOZeroDetHighPower
# sky/pol chosen where the antenna pattern drifts appreciably over a sidereal day
RA, DEC, PSI, INCL, PHIREF = 1.0, -0.6, 0.4, 0.5, 0.0
HARM = (-2, -1, 0, 1, 2)

# (label, m1, m2, fmin, fmax, dist[Mpc]) -- increasing duration via lower mass / fmin.
# Segment length is AUTO-FIT to each waveform (next power of two).  Distances tuned for a
# comparable (loud) SNR (~30) so the gain tracks Omega*T, not loudness.
CONFIGS = [
    ("null_bbh",  30.0, 25.0, 30.0, 1024.0,  900.0),   # short BBH, ~2 s: control, gain~0
    ("bbh_8_8",    8.0,  8.0, 25.0, 1024.0,  380.0),   # ~16 s
    ("bbh_4_4",    4.0,  4.0, 22.0, 1024.0,  210.0),   # ~64 s -> larger Omega*T
]
OMEGA_EARTH = flwr.OMEGA_EARTH


def _P(m1, m2, fmin, deltaT, deltaF):
    P = lsu.ChooseWaveformParams(
        fmin=fmin, radec=True, incl=INCL, phiref=PHIREF, theta=DEC, phi=RA, psi=PSI,
        m1=m1 * lal.MSUN_SI, m2=m2 * lal.MSUN_SI, detector=DET,
        dist=200e6 * lal.PC_SI, deltaT=deltaT, tref=EVENT_TIME, deltaF=deltaF)
    P.approx = lalsim.GetApproximantFromString("IMRPhenomD")
    return P


def _P_vec(Psig, dist_SI, K=1):
    Pv = Psig.manual_copy()
    Pv.phi = np.ones(K) * RA; Pv.theta = np.ones(K) * DEC; Pv.psi = np.ones(K) * PSI
    Pv.incl = np.ones(K) * INCL; Pv.phiref = np.ones(K) * PHIREF
    Pv.dist = np.ones(K) * dist_SI
    Pv.tref = float(EVENT_TIME); Pv.deltaT = Psig.deltaT
    return Pv


def run_config(label, m1, m2, fmin, fmax, dist_Mpc):
    deltaT = 1.0 / (2 * fmax) if fmax > 1024 else 1.0 / 2048.0
    dist_SI = dist_Mpc * 1e6 * lal.PC_SI
    t_window = 0.1
    # AUTO-FIT seglen: generate with deltaF=None so non_herm_hoff pads to the next power of
    # two >= the waveform length; then read back the segment length / deltaF.
    Psig = _P(m1, m2, fmin, deltaT, None)
    Psig.dist = dist_SI
    data = lsu.non_herm_hoff(Psig)     # RIFT injection -> carries the true time-varying response
    seglen = data.data.length * deltaT
    Psig.deltaF = data.deltaF
    data_dict = {DET: data}; psd_dict = {DET: PSD}
    fNyq = 1.0 / (2 * deltaT)
    IP = lsu.ComplexIP(fmin, fmax, fNyq, data.deltaF, PSD, True, False, 0.)
    snr = np.sqrt(IP.ip(data, data).real)
    Pv = _P_vec(Psig, dist_SI)
    Nw = int(0.02 / deltaT); tvals = np.arange(-Nw, Nw) * deltaT

    # baseline (static) NoLoop, time-marginalized, at the truth
    rib, ctb, ctVb, rhob, _, _ = fl.PrecomputeLikelihoodTerms(
        EVENT_TIME, t_window, Psig, data_dict, psd_dict, LMAX, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, ignore_threshold=None)
    lk = {}; rA = {}; cu = {}; cv = {}; ep = {}
    for d in data_dict:
        a, b, c, U, V, rAr, rI, e = fl.PackLikelihoodDataStructuresAsArrays(
            list(rhob[d].keys()), None, rhob[d], ctb[d], ctVb[d])
        lk[d] = a; rA[d] = rAr; cu[d] = U; cv[d] = V; ep[d] = e
    lnL_static = float(fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, Pv, lk, rA, cu, cv, ep, Lmax=LMAX, xpy=np, time_interp='cubic')[0])

    # rotation (Path A) NoLoop at the truth
    bk = flwr.PrecomputeLikelihoodTermsWithRotation(
        EVENT_TIME, t_window, Psig, data_dict, psd_dict, LMAX, fmax,
        harmonics=HARM, p_max=0, f_sidereal=flwr.F_SIDEREAL, analyticPSD_Q=True,
        verbose=False, quiet=True, skip_interpolation=True)
    lkr, rbn, ubn, vbn, epr = flwr.pack_rotation_arrays(bk[4], bk[3], bk[1], bk[2])
    lnL_rot = float(flwr.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopWithRotation(
        tvals, Pv, bk[4], lkr, rbn, ubn, vbn, epr, Lmax=LMAX, time_interp='cubic')[0])

    OmegaT = OMEGA_EARTH * seglen
    return dict(label=label, seglen=seglen, snr=float(snr), OmegaT=OmegaT,
                lnL_static=lnL_static, lnL_rot=lnL_rot, gain=lnL_rot - lnL_static)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    outdir = os.path.join(here, "outputs"); os.makedirs(outdir, exist_ok=True)
    print("slow-rotation (Path A) value demo: single %s, SNR~30 held fixed so gain tracks Omega*T\n" % DET)
    rows = []
    for cfg in CONFIGS:
        try:
            r = run_config(*cfg)
        except Exception as e:
            print("%-9s SKIPPED (%s)" % (cfg[0], e)); continue
        rows.append(r)
        print("%-9s seglen=%5.0fs SNR=%6.1f Omega*T=%.2e  lnL_static=%12.4f lnL_rot=%12.4f  GAIN=%+.4f"
              % (r['label'], r['seglen'], r['snr'], r['OmegaT'],
                 r['lnL_static'], r['lnL_rot'], r['gain']))
    if not rows:
        print("no configs succeeded"); return
    txt = os.path.join(outdir, "rotation_gain_vs_duration.txt")
    with open(txt, "w") as f:
        f.write("# slow-rotation demo: --rotation-slow recovers SNR the static analysis loses on long signals\n")
        f.write("# label seglen[s] SNR Omega*T lnL_static lnL_rot gain=lnL_rot-lnL_static\n")
        for r in rows:
            f.write("%s %g %g %g %g %g %g\n" % (r['label'], r['seglen'], r['snr'],
                    r['OmegaT'], r['lnL_static'], r['lnL_rot'], r['gain']))
    print("\nwrote", txt)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        OmT = [r['OmegaT'] for r in rows]; gain = [r['gain'] for r in rows]
        labs = [r['label'] for r in rows]
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot(OmT, gain, 'o-', color="#2a6f97")
        for x, y, l in zip(OmT, gain, labs):
            ax.annotate(l, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
        ax.set_xlabel(r"$\Omega_\oplus T$  (rotation phase over the signal)")
        ax.set_ylabel(r"$\ln\mathcal{L}_{\rm rot}-\ln\mathcal{L}_{\rm static}$  (recovered)")
        ax.set_title("Accounting for Earth rotation recovers SNR on long signals")
        ax.axhline(0, color="0.7", lw=0.8)
        fig.tight_layout()
        png = os.path.join(outdir, "rotation_gain_vs_duration.png")
        fig.savefig(png, dpi=140)
        print("wrote", png)
    except Exception as e:
        print("(plot skipped:", e, ")")


if __name__ == "__main__":
    main()
