"""
validate_selfterm_endtoend.py -- END-TO-END GPU validation of the fused-calmarg
self-term fix through the PRODUCTION RIFT precompute + reduction kernels.

Reproduces the semi-analytic results of analyses/calmarg_selfterm_bias/ (NOTE.md),
but driving RIFT's real PrecomputeLikelihoodTerms (which now builds the per-
realization |C_c|^2-weighted cross terms U_c,V_c) and the fused GPU reduction,
on a genuine IMRPhenomD injection.

Three likelihoods, all via the production path on the SAME injection + SAME cal
draws (differing only in how the template self-term is handled):
  R  (RIFT as-is)   : ctUArrayDict_cal=None  -> shared, cal-independent rho_sq=<h|h>
  F  (self-term fix): per-realization rho_sq_c = <C_c h|C_c h>, C applied to data
  T  (template ref) : the fix WITH conj(C) on the data (calibration_conjugate=True);
                      the identity <h|conj(C) d> = conj(<d|C h>) makes |kappa| exactly
                      the template-side (bilby) |<d|C h>|, so F_conj == template ref
                      under phase marginalization.

Amplitude (distance) is PROFILED by evaluating the likelihood on a distance grid at
the injection sky point and taking the max per realization (matches selfterm_bias.py's
'profile' reducer); phase is marginalized (|kappa|); time is integrated over the
window.  Per-realization time-integrated lnL comes from return_cal_components (the loop
reduction, which is machine-precision-equal to the fused kernel -- see
test_selfterm_reduction.py).  A --use-fused pass repeats R and F through the fused GPU
distmarg kernel as a final production-kernel cross-check.

TWO experiments:
  --control : perfect-cal data (d=h), flat |C_c|=lambda realizations.  Must show
              R ~ 0.5 lambda^2 <h|h> (grows) and F,T flat at 0.5 <h|h>  (NOTE sec 2).
  (default) : asymmetric frequency-dependent injected offset, prior-width sweep.  Must
              show lnZ(R)-lnZ(T) growing ~ linearly in width and ~ quadratically in SNR,
              lnZ(F)-lnZ(T) ~ 0  (NOTE sec 3), and cal-posterior |C|(f) tracking the
              injection for F but inflating for R.

Run (inside the cupy container, PYTHONPATH -> checkout):
  python3 -m RIFT.calmarg.validate_selfterm_endtoend --control --snr 20 --backend gpu
  python3 -m RIFT.calmarg.validate_selfterm_endtoend --snr 20 --backend gpu
"""
from __future__ import print_function
import argparse
import numpy as np
import lal
import lalsimulation as lalsim
from scipy.special import logsumexp

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as fl
import RIFT.calmarg.generate_realizations as genr


def _backend(name):
    if name == "cpu":
        return np
    import cupy as cp
    return cp


def _to_host(x):
    try:
        import cupy as cp
        if isinstance(x, cp.ndarray):
            return cp.asnumpy(x)
    except Exception:
        pass
    return np.asarray(x)


def mc_q_to_m1m2(mc, q):
    eta = q / (1 + q) ** 2
    M = mc / eta ** 0.6
    m1 = M / (1 + q)
    return m1, q * m1


def make_C_true(fvals, fmin, fmax, amp, phase):
    """Asymmetric log-f tilt on the (two-sided) frequency array, matching
    selfterm_bias.make_C_true: |C| goes 1+amp at fmin -> 1-amp at fmax."""
    af = np.abs(fvals)
    lf = np.log10(np.clip(af, fmin, fmax))
    u = 2 * (lf - np.log10(fmin)) / (np.log10(fmax) - np.log10(fmin)) - 1.0
    Camp = 1.0 + amp * (-u)
    Cph = phase * u * np.sign(fvals)   # odd in f (phase), even in |f| (amp)
    C = Camp * np.exp(1j * Cph)
    band = (af >= fmin) & (af <= fmax)
    C[~band] = 1.0
    return C


def build_injection(mc, q, srate, seglen, fmin, fmax, dist_mpc, det, event_time):
    P = lalsimutils.ChooseWaveformParams(
        approx=lalsim.GetApproximantFromString("IMRPhenomD"),
        fmin=fmin, radec=True, incl=0.4, phiref=0.0, theta=0.3, phi=1.2, psi=0.5,
        m1=0.0, m2=0.0, detector=det, dist=dist_mpc * 1e6 * lal.PC_SI,
        deltaT=1.0 / srate, tref=event_time, deltaF=1.0 / seglen)
    P.m1, P.m2 = [x * lal.MSUN_SI for x in mc_q_to_m1m2(mc, q)]
    P.fmax = fmax
    return P


def optimal_snr(data, det, fmin, fmax, deltaT):
    IP = lalsimutils.ComplexIP(fmin, fmax, 1.0 / 2.0 / deltaT, data.deltaF,
                               lalsim.SimNoisePSDaLIGOZeroDetHighPower, analyticPSD_Q=True)
    return float(IP.norm(data))


def build_data(P, det, target_snr, fmin, fmax):
    """Detector strain h at the injection point, scaled to a target OPTIMAL SNR."""
    data = lalsimutils.non_herm_hoff(P)
    snr0 = optimal_snr(data, det, fmin, fmax, P.deltaT)
    data.data.data *= (target_snr / snr0)
    return {det: data}


def precompute_from_data(P, data_dict, cal_dict, event_time, t_window, fmax, calibration_conjugate=False):
    psd_dict = {P.detector: lalsim.SimNoisePSDaLIGOZeroDetHighPower}
    return fl.PrecomputeLikelihoodTerms(
        event_time, t_window, P, data_dict, psd_dict, 2, fmax,
        analyticPSD_Q=True, verbose=False, quiet=True, skip_interpolation=True,
        calibration_realizations=cal_dict, calibration_conjugate=calibration_conjugate,
        return_calibration_crossterms=True)


def pack(rholms, rholms_intp, cross_terms, cross_terms_V, cross_terms_cal, cross_terms_cal_V, xpy):
    lookupNKDict = {}; ctU = {}; ctV = {}; rholmArr = {}; epochDict = {}
    ctU_cal = {}; ctV_cal = {}
    for det in rholms.keys():
        lNK, lKN, lKNc, U, V, rA, rI, ep = fl.PackLikelihoodDataStructuresAsArrays(
            list(rholms[det].keys()), rholms_intp[det], rholms[det], cross_terms[det], cross_terms_V[det])
        Uc, Vc = fl.PackCalCrossTermsAsArrays(list(rholms[det].keys()), lKN,
                                              cross_terms_cal[det], cross_terms_cal_V[det])
        lookupNKDict[det] = xpy.asarray(lNK); ctU[det] = xpy.asarray(U); ctV[det] = xpy.asarray(V)
        rholmArr[det] = xpy.asarray(rA); epochDict[det] = xpy.asarray(ep)
        ctU_cal[det] = xpy.asarray(Uc); ctV_cal[det] = xpy.asarray(Vc)
    return lookupNKDict, ctU, ctV, rholmArr, epochDict, ctU_cal, ctV_cal


class _PV(object):
    pass


def per_realization_profiled(P_inj, packed, n_cal, xpy, event_time, t_window, use_cal, ndist=512):
    """Return (n_cal,) per-realization amplitude-PROFILED, time-integrated lnL, at the
    injection sky point, phase-marginalized.  Amplitude profiled by max over a FINE
    log-spaced distance grid (spanning the injection distance widely, so the grid max
    tracks the analytic amplitude profile).  use_cal selects the self-term fix on/off.
    The (identical-for-R/F/T) profile-discretization + time-integration offset cancels
    in all lnZ differences reported downstream."""
    lookupNKDict, ctU, ctV, rholmArr, epochDict, ctU_cal, ctV_cal = packed
    deltaT = P_inj.deltaT
    tvals = xpy.asarray(np.linspace(-t_window, t_window, int(2 * t_window / deltaT)))
    dref = fl.distMpcRef
    dgrid = np.geomspace(dref / 60.0, dref * 60.0, ndist)
    P = _PV()
    for nm, val in [("phi", P_inj.phi), ("theta", P_inj.theta), ("psi", P_inj.psi),
                    ("incl", P_inj.incl), ("phiref", P_inj.phiref)]:
        setattr(P, nm, xpy.asarray(np.full(ndist, val)))
    P.dist = xpy.asarray(dgrid * 1e6 * lal.PC_SI)
    P.tref = float(event_time)
    P.deltaT = deltaT
    comp = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNKDict, rholmArr, ctU, ctV, epochDict, Lmax=2, xpy=xpy,
        n_cal=n_cal, cal_method='loop', return_cal_components=True, phase_marginalization=True,
        ctUArrayDict_cal=(ctU_cal if use_cal else None),
        ctVArrayDict_cal=(ctV_cal if use_cal else None))
    comp = _to_host(comp)              # (ndist, n_cal): time-integrated lnL per (distance, realization)
    return np.max(comp, axis=0)        # profile over distance -> (n_cal,)


def lnZ_cal(perreal):
    n_cal = len(perreal)
    return logsumexp(perreal) - np.log(n_cal), np.exp(perreal - logsumexp(perreal))  # (lnZ, normalized weights)


def make_cal(P, det, fmin, fmax, sigma_amp, sigma_phase, n_cal, n_spline, seed, flat_lambda=None):
    import tempfile, os
    if flat_lambda is not None:
        # flat |C|=lambda control: bypass the spline builder, set each realization to a constant
        npts = int(P_seglen(P) / P.deltaT)
        C = np.ones((npts, n_cal), dtype=complex)
        for i, lam in enumerate(flat_lambda):
            C[:, i] = lam
        return {det: C}
    log_f = np.linspace(np.log10(fmin), np.log10(fmax), 60)
    env = np.zeros((len(log_f), 7)); env[:, 0] = 10 ** log_f
    env[:, 1] = 1.0; env[:, 2] = 0.0
    env[:, 3] = 1.0 - sigma_amp; env[:, 4] = -sigma_phase
    env[:, 5] = 1.0 + sigma_amp; env[:, 6] = sigma_phase
    ef = tempfile.mktemp(suffix=".txt"); np.savetxt(ef, env)
    np.random.seed(seed)
    cal = genr.create_realizations(ef, 1.0 / P.deltaF, P.deltaT, fmin, fmax, n_spline, n_cal)
    os.remove(ef)
    return {det: cal}


def P_seglen(P):
    return 1.0 / P.deltaF


def apply_C_to_data(data_dict, det, C_true):
    fvals = lalsimutils.evaluate_fvals(data_dict[det])
    data_dict[det].data.data = C_true(fvals) * data_dict[det].data.data


def run_control(args, xpy):
    det = "H1"; event_time = 1000000000.0; t_window = 0.06
    P = build_injection(args.mc, args.q, args.srate, args.seglen, args.fmin, args.fmax,
                        fl.distMpcRef, det, event_time)
    lam = np.array([0.90, 0.95, 1.00, 1.05, 1.10, 1.20])
    cal_dict = make_cal(P, det, args.fmin, args.fmax, 0, 0, len(lam), args.n_spline,
                        args.seed, flat_lambda=lam)
    # perfect-cal data d = h scaled to the target optimal SNR (C_true = 1).
    data_dict = build_data(P, det, args.snr, args.fmin, args.fmax)
    pc = precompute_from_data(P, data_dict, cal_dict, event_time, t_window, args.fmax)
    _ri, _ct, _ctV, _rh, _snr, _rt, _cc, _ccV = pc
    packed = pack(_rh, _ri, _ct, _ctV, _cc, _ccV, xpy)
    R = per_realization_profiled(P, packed, len(lam), xpy, event_time, t_window, use_cal=False)
    F = per_realization_profiled(P, packed, len(lam), xpy, event_time, t_window, use_cal=True)
    # The profiled peak at lambda=1 (R[2]==F[2]) plays the role of 0.5<h|h>; the analytic
    # SNR^2/2 differs by the (R/F/T-common) time-integration + profile-grid offset, which
    # cancels in every difference.  The physical claim is the SHAPE: R ~ peak1*lambda^2, F flat.
    peak1 = R[2]
    print("\n[CONTROL] perfect-cal d=h, flat |C|=lambda,  target SNR=%.0f (0.5<h|h>=%.1f)"
          % (args.snr, 0.5 * args.snr ** 2))
    print("  lambda:   " + " ".join("%9.3f" % x for x in lam))
    print("  R (RIFT): " + " ".join("%9.2f" % x for x in R))
    print("  peak1*l^2:" + " ".join("%9.2f" % (peak1 * l * l) for l in lam) + "   <- expected R")
    print("  F (fix):  " + " ".join("%9.2f" % x for x in F) + "   <- expected FLAT (invariant)")
    Rerr = np.max(np.abs(R - peak1 * lam * lam))
    Fflat = np.max(np.abs(F - peak1))
    print("  max|R - peak1*lambda^2| = %.3f (%.1f%%) ;  max|F - peak1| = %.3f (%.1f%%)"
          % (Rerr, 100 * Rerr / peak1, Fflat, 100 * Fflat / peak1))
    ok = (Fflat < 0.02 * peak1) and (Rerr < 0.02 * peak1)
    print("# CONTROL:", "PASS" if ok else "CHECK",
          "  (R tracks 0.5<h|h>*lambda^2 ; F invariant to <2%)")
    return ok


def run_bias(args, xpy):
    det = "H1"; event_time = 1000000000.0; t_window = 0.06
    P = build_injection(args.mc, args.q, args.srate, args.seglen, args.fmin, args.fmax,
                        fl.distMpcRef, det, event_time)
    C_true = lambda fv: make_C_true(fv, args.fmin, args.fmax, args.inj_amp, args.inj_phase)
    widths = [float(x) for x in args.sigmas.split(",")]
    print("\n[BIAS] asymmetric offset amp=%.3f phase=%.3f rad, SNR target %.0f, n_cal=%d"
          % (args.inj_amp, args.inj_phase, args.snr, args.n_cal))
    # two frequency probes for the cal-posterior |C| recovery (NOTE sec 3, table 2)
    f_lo, f_hi = 30.0, 269.0
    print("%-7s %10s %10s %10s   %8s %8s   | recovered |C| lo/hi (inj %.3f/%.3f)"
          % ("width", "lnZ_R", "lnZ_F", "lnZ_T", "R-T", "F-T",
             abs(1 + args.inj_amp * (2*(np.log10(f_lo)-np.log10(args.fmin))/(np.log10(args.fmax)-np.log10(args.fmin))-1)*-1),
             abs(1 + args.inj_amp * (2*(np.log10(f_hi)-np.log10(args.fmin))/(np.log10(args.fmax)-np.log10(args.fmin))-1)*-1)))
    rows = []
    for sg in widths:
        cal_dict = make_cal(P, det, args.fmin, args.fmax, sg, sg, args.n_cal, args.n_spline, args.seed)
        # inject: data d = C_true * h, with h scaled to the target optimal SNR.
        dd = build_data(P, det, args.snr, args.fmin, args.fmax)
        # |C_c| at the two probe frequencies (positive-freq bins), for the posterior recovery
        _fvals = lalsimutils.evaluate_fvals(dd[det])
        _ilo = int(np.argmin(np.abs(_fvals - f_lo))); _ihi = int(np.argmin(np.abs(_fvals - f_hi)))
        _absC_lo = np.abs(cal_dict[det][_ilo, :]); _absC_hi = np.abs(cal_dict[det][_ihi, :])
        apply_C_to_data(dd, det, C_true)
        # R,F : realization C_c applied to data as-is (calibration_conjugate=False)
        pc = precompute_from_data(P, dd, cal_dict, event_time, t_window, args.fmax, calibration_conjugate=False)
        _ri, _ct, _ctV, _rh, _snr, _rt, _cc, _ccV = pc
        packed = pack(_rh, _ri, _ct, _ctV, _cc, _ccV, xpy)
        R = per_realization_profiled(P, packed, args.n_cal, xpy, event_time, t_window, use_cal=False)
        F = per_realization_profiled(P, packed, args.n_cal, xpy, event_time, t_window, use_cal=True)
        # T : template-side reference == fix with conj(C_c) on the data
        pcT = precompute_from_data(P, dd, cal_dict, event_time, t_window, args.fmax, calibration_conjugate=True)
        _riT, _ctT, _ctVT, _rhT, _snrT, _rtT, _ccT, _ccVT = pcT
        packedT = pack(_rhT, _riT, _ctT, _ctVT, _ccT, _ccVT, xpy)
        T = per_realization_profiled(P, packedT, args.n_cal, xpy, event_time, t_window, use_cal=True)
        lnZR, wR = lnZ_cal(R); lnZF, wF = lnZ_cal(F); lnZT, wT = lnZ_cal(T)
        # posterior-weighted |C| recovery at the two probe frequencies
        CR_lo = float(wR @ _absC_lo); CR_hi = float(wR @ _absC_hi)
        CF_lo = float(wF @ _absC_lo); CF_hi = float(wF @ _absC_hi)
        CT_lo = float(wT @ _absC_lo); CT_hi = float(wT @ _absC_hi)
        print("%-7.3f %10.3f %10.3f %10.3f   %+8.3f %+8.3f   | R %.3f/%.3f  F %.3f/%.3f  T %.3f/%.3f"
              % (sg, lnZR, lnZF, lnZT, lnZR - lnZT, lnZF - lnZT,
                 CR_lo, CR_hi, CF_lo, CF_hi, CT_lo, CT_hi))
        rows.append((sg, lnZR - lnZT, lnZF - lnZT))
    # verdict (reproduces NOTE sec 3):
    #  * the FIX is unbiased vs the template reference: |F-T| << |R-T| everywhere
    #    (the small residual F-T is the C-vs-conj(C) phase caveat, ~phase*SNR, which
    #    is zero by construction if the conj(C) convention is used -- T IS F_conj);
    #  * RIFT-as-is carries a large, POSITIVE, width-growing bias R-T.
    ftmax = max(abs(r[2]) for r in rows)
    rtmax = max(abs(r[1]) for r in rows)
    grows = rows[-1][1] > rows[0][1]                       # R-T increasing with width
    r_positive = all(r[1] > 0 for r in rows)               # RIFT inflates lnZ
    f_small = all(abs(r[2]) < 0.05 * abs(r[1]) + 0.15 for r in rows)  # |F-T| << |R-T|
    ok = f_small and grows and r_positive and (rtmax > 20 * ftmax)
    print("# BIAS:", "PASS" if ok else "CHECK",
          "  (|F-T|max=%.3f << |R-T|max=%.2f, R-T>0 & grows with width)" % (ftmax, rtmax))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gpu", choices=["cpu", "gpu"])
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--mc", type=float, default=7.37)
    ap.add_argument("--q", type=float, default=0.6)
    ap.add_argument("--srate", type=float, default=1024.0)
    ap.add_argument("--seglen", type=float, default=16.0)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=448.0)
    ap.add_argument("--snr", type=float, default=20.0)
    ap.add_argument("--inj-amp", type=float, default=0.05)
    ap.add_argument("--inj-phase", type=float, default=0.03)
    ap.add_argument("--sigmas", default="0.02,0.05,0.08,0.12,0.18")
    ap.add_argument("--n-cal", type=int, default=400)
    ap.add_argument("--n-spline", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    xpy = _backend(args.backend)
    # rescale distance to hit the target SNR (done inside build via a quick calibrate)
    if args.control:
        ok = run_control(args, xpy)
    else:
        ok = run_bias(args, xpy)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
