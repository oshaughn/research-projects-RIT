#!/usr/bin/env python
"""study_stencil_lnL_sensitivity.py

DOES THE Q_lm SUB-SAMPLE TIME-INTERPOLATION STENCIL MOVE lnL AND lnZ?

Measurement, using the real RIFT likelihood machinery (no toy signals):

  * Build a ChooseWaveformParams signal, a zero-noise data_dict over H1/L1/V1, an analytic
    aLIGO ZDHP PSD, and run fl.PrecomputeLikelihoodTerms + PackLikelihoodDataStructuresAsArrays
    exactly as test_slowrot_noloop.py / test_slowrot_gpu.py do.
  * Draw a FIXED set of K extrinsic points from a FIXED seed.  Every stencil sees the SAME
    points, so this is a paired comparison and the stencil is the only thing that varies.
  * Evaluate fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop with return_lnLt=True for
    time_interp in {'nearest','cubic','sinc'} on a common coarse time grid.
  * REFERENCE ("infinite sinc"): Q_lm(t) as produced by ComputeModeIPTimeSeries is the inverse
    FFT of a spectrum that is identically zero outside [fmin,fMax], so it is band-limited.
    Zero-padding its FFT by an integer factor M and inverse-transforming is therefore an
    essentially exact interpolation onto an M-times finer time grid.  We then evaluate the
    likelihood by NEAREST lookup on that fine grid, which is what the reference is.

    WHERE THE REFERENCE IS NOT EXACT (stated up front, and measured below):
      (a) residual quantization: nearest lookup on the fine grid still has up to 1/(2M) of a
          COARSE sample of timing error.  Checked by re-running the reference at 2M and
          demanding the reference move by much less than the smallest stencil-vs-reference
          difference.
      (b) periodic wrap: PrecomputeLikelihoodTerms stores a CUT of the full-length rho(t)
          series, and zero-pad-FFT interpolation of a cut treats the cut as periodic.  The
          resulting Gibbs ringing is an error in the reference itself, which (a) cannot see
          because both M and 2M share it.  Checked independently by rebuilding the reference
          from a Q window HALF as long (edges twice as close, wrap artifact ~2x larger) and
          comparing; the evaluation window is kept far from the stored-window edges.
  * Reduce each lnL_t(K,npts) to one lnL per extrinsic point by Simpson time integration with
    IDENTICAL weights for all four methods (this is what the production code does internally
    with dx=deltaT; doing it here keeps the quadrature out of the comparison).
  * Evidence: lnZ = log(mean(exp(lnL - max))) + max over the fixed point set; repeated over
    several seeds so the seed-to-seed SPREAD of lnZ - lnZ_ref is reported alongside the mean.

Run (CPU only, off the session host):
  OMP_NUM_THREADS=1 PYTHONPATH=/home/richard.oshaughnessy/rift_wt_sinc/MonteCarloMarginalizeCode/Code \
    /home/richard.oshaughnessy/RIFT_develUWM/bin/python \
    RIFT/likelihood/study_stencil_lnL_sensitivity.py 2>/dev/null
"""
from __future__ import print_function, division

import sys
import time
import argparse

import numpy as np
import lal
import lalsimulation as lalsim

import RIFT.lalsimutils as lsu
import RIFT.likelihood.factored_likelihood as fl

# Same environment workaround the existing slowrot tests use: when numba's @vectorize
# decoration fails at import (RIFT_LOWLATENCY set in this venv), factored_likelihood falls
# back to a scalar lalylm that cannot take array arguments.  Rebind it locally, for this
# process only.  Does not touch factored_likelihood.py on disk.
if not getattr(fl, "numba_on", True):
    fl.lalylm = np.vectorize(lal.SpinWeightedSphericalHarmonic, otypes=[complex])

EVENT_TIME = 1e9
LMAX = 2
REF_STENCIL = 'cubic'   # lookup used on the FFT-upsampled fine grid; see eval_reference
DELTA_F = 1. / 4.


# ---------------------------------------------------------------------------
# configuration / precompute
# ---------------------------------------------------------------------------
# The model behind the shipped guidance (RIFT/likelihood/DESIGN_q_window_stencil.md).  Named once
# so the banner, the skip message, the argparse help and Setup cannot disagree -- they did.
DEFAULT_APPROX = 'SEOBNRv4'


class Setup(object):
    """Everything the likelihood needs for one (sample rate, fmax, source) combination."""

    def __init__(self, label, fSample, fmax, m1, m2, fmin, t_window, dist_mpc=200.,
                 deltaF=DELTA_F, approx=None, quiet=True):
        self.label = label
        self.fSample = float(fSample)
        self.fmax = float(fmax)
        self.deltaT = 1. / self.fSample
        self.fmin = float(fmin)
        self.t_window = float(t_window)
        self.oversampling = (self.fSample / 2.) / self.fmax
        self.dist_mpc = float(dist_mpc)
        self.deltaF = float(deltaF)

        self.Psig = lsu.ChooseWaveformParams(
            fmin=self.fmin, radec=True, incl=0.3, phiref=0.0, theta=0.2, phi=1.0, psi=0.4,
            m1=m1 * lal.MSUN_SI, m2=m2 * lal.MSUN_SI,
            detector='H1', dist=self.dist_mpc * 1e6 * lal.PC_SI, deltaT=self.deltaT,
            tref=EVENT_TIME, deltaF=self.deltaF)
        # Approximant.  Default (None) leaves ChooseWaveformParams' own default, TaylorT4,
        # which is what the existing slowrot tests use.  SEOBNRv4 is a TD IMR model and is
        # the reason this is an argument: TaylorT4 terminates at ISCO and has NO merger or
        # ringdown, so every feature above f_ISCO in a TaylorT4 Q spectrum is termination
        # ringing from the approximant rather than physics.
        # DEFAULT IS THE IMR MODEL, deliberately.  This script produced the guidance in
        # RIFT/likelihood/DESIGN_q_window_stencil.md, and that guidance rests on SEOBNRv4.
        # Defaulting to TaylorT4 meant an ordinary reproduction run regenerated inspiral-only
        # numbers -- which are not merely less precise: they NAMED THE WRONG STENCIL at M = 9,
        # 10 and 20, because TaylorT4 terminates at ISCO and carries no merger-ringdown.  A
        # script whose default output contradicts the recommendation it supports is a trap.
        self.approx_name = approx or DEFAULT_APPROX
        self.Psig.approx = getattr(lalsim, self.approx_name)
        self._unreachable_hint = (
            "%s cannot be generated at srate %g for M = %.4g Msun (its ringdown exceeds Nyquist). "
            "Raise --mass-ladder-srate / the config's srate to 16384, or pass an inspiral-only "
            "model with --approx TaylorT4 -- but note inspiral-only results named the WRONG "
            "stencil at M = 9, 10 and 20, so do not use them to support stencil guidance."
            % (self.approx_name, fSample, m1 + m2))
        if 'Taylor' in self.approx_name:
            print("  ** WARNING: %s is INSPIRAL-ONLY (terminates at ISCO, no merger-ringdown).\n"
                  "     It understates the Q bandwidth by 2-3.7x and named the WRONG stencil at\n"
                  "     M = 9, 10 and 20. Do not use these numbers to support stencil guidance;\n"
                  "     see RIFT/likelihood/DESIGN_q_window_stencil.md." % self.approx_name)
        self.data_dict = {}
        for det in ("H1", "L1", "V1"):
            P = self.Psig.manual_copy()
            P.detector = det
            self.data_dict[det] = lsu.non_herm_hoff(P)
        self.psd_dict = {det: lalsim.SimNoisePSDaLIGOZeroDetHighPower for det in self.data_dict}
        # SEOBNRv4's TD path will not truncate: if the signal does not fit the segment it
        # WRAPS, silently and catastrophically.  Check the actual strain the likelihood will
        # see -- inverse-transform the non_herm_hoff series (packed [-fNyq .. fNyq-df], hence
        # the ifftshift) and test whether it is still live at the segment edges, which is
        # exactly what wrapping produces.  A signal that fits is tapered to ~0 at both ends.
        self.seg_duration = 1.0 / self.deltaF
        _ht = np.fft.ifft(np.fft.ifftshift(self.data_dict['H1'].data.data))
        _a = np.abs(_ht)
        _peak = float(np.max(_a))
        assert _peak > 0 and np.all(np.isfinite(_a)), \
            "%s at M=%.4g produced empty or non-finite strain" % (self.approx_name, m1 + m2)
        _n_edge = max(16, int(0.001 * len(_a)))
        _edge = max(float(np.max(_a[:_n_edge])), float(np.max(_a[-_n_edge:]))) / _peak
        _live = np.nonzero(_a > 1e-4 * _peak)[0]
        self.wf_duration = float(len(_live)) / self.fSample
        self.edge_fraction = _edge
        assert _edge < 1e-2, (
            "%s at M=%.4g, srate %g, segment %.4g s: strain is still at %.2e of peak at the "
            "segment edge -- the waveform does not fit and has WRAPPED"
            % (self.approx_name, m1 + m2, self.fSample, self.seg_duration, _edge))

        self.packs = self._precompute(self.t_window, quiet)

    def _precompute(self, t_window, quiet=True):
        # NOTE: PrecomputeLikelihoodTerms RESETS P.dist to the fiducial reference distance
        # in place, so hand it a copy.
        Ptmpl = self.Psig.manual_copy()
        out = fl.PrecomputeLikelihoodTerms(
            EVENT_TIME, t_window, Ptmpl, self.data_dict, self.psd_dict, LMAX, self.fmax,
            analyticPSD_Q=True, verbose=False, quiet=quiet, ignore_threshold=None,
            skip_interpolation=True)
        rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, _rest = out
        packs = dict(lookupNK={}, rho={}, ctU={}, ctV={}, epoch={}, snr=guess_snr)
        for det in self.data_dict:
            pairKeys = list(rholms[det].keys())
            (lookupNK, _keys2n, _conj, ctU, ctV, rholmArray, _intp, epoch) = \
                fl.PackLikelihoodDataStructuresAsArrays(
                    pairKeys, None, rholms[det], crossTerms[det], crossTermsV[det])
            packs['lookupNK'][det] = lookupNK
            packs['rho'][det] = rholmArray          # (n_lms, n_time)
            packs['ctU'][det] = ctU
            packs['ctV'][det] = ctV
            packs['epoch'][det] = epoch
        return packs

    def alternate_window_packs(self, t_window):  # noqa: D401
        """Second precompute with a different stored-Q window (reference wrap-artifact test)."""
        return self._precompute(t_window)


# ---------------------------------------------------------------------------
# extrinsic points
# ---------------------------------------------------------------------------
def draw_points(K, seed, dist_mpc):
    """Isotropic sky/orientation, distance uniform over [0.5, 4] x the injected distance --
    the same shape as test_slowrot_gpu._P_vec (100-800 Mpc about a 200 Mpc injection), scaled
    so that every configuration is probed over the same range of lnL."""
    rng = np.random.RandomState(seed)
    return dict(
        phi=rng.uniform(0, 2 * np.pi, K),                 # RA
        theta=np.arcsin(rng.uniform(-1, 1, K)),           # DEC
        psi=rng.uniform(0, np.pi, K),
        incl=np.arccos(rng.uniform(-1, 1, K)),
        phiref=rng.uniform(0, 2 * np.pi, K),
        dist=rng.uniform(0.5 * dist_mpc, 4.0 * dist_mpc, K) * 1e6 * lsu.lsu_PC,
    )



RELEVANT_BAND = 30.0   # nats below the peak; points fainter than this carry exp(-30) of the
                       # posterior weight and cannot move any inference


def draw_points_near_truth(K, seed, setup, rho, rho0=100.0, base=0.05, s_max=0.1):
    """Cloud AROUND the injection, with every offset scaled as 1/SNR.

    Why this set exists.  The isotropic set above is drawn over the whole sky with distance
    down to 0.5 x the injected distance, so it contains points whose lnL is enormous and
    NEGATIVE (rho_sq ~ 1/d^2 with a mismatched sky).  Those points have |kappa| large, hence
    |d lnL| large, but weight exp(lnL - lnL_max) ~ 0: a max| | over the isotropic set is
    therefore dominated by samples that cannot influence any inference.  Here the offsets
    scale as 1/rho, which is how the posterior width scales, so the cloud spans a comparable
    band of lnL at EVERY rung and the error statistics over it are directly comparable across
    the SNR ladder.

    Two guards, both necessary and both learned the hard way:
      * the distance offset is LOGNORMAL (d -> d exp(s z)), not d(1 + s z).  The linear form
        drives d towards zero for s of order 1, and rho_sq ~ 1/d^2 then produces lnL of order
        -1e10, which swamps every statistic computed over the cloud.
      * s is capped at s_max.  1/rho scaling keeps the lnL span of the cloud constant, but only
        while the quadratic expansion of lnL about the peak holds; the cap keeps the low-SNR
        rungs inside it.  Below the cap the cloud is simply TIGHTER than scale-invariant, which
        is harmless.  The realised lnL span is printed for every rung -- check it.
    """
    rng = np.random.RandomState(seed + 777)
    s = min(float(s_max), base * rho0 / float(rho))
    P = setup.Psig
    eps = 1e-6
    return dict(
        phi=float(P.phi) + s * rng.randn(K),
        theta=np.clip(float(P.theta) + s * rng.randn(K), -np.pi / 2 + eps, np.pi / 2 - eps),
        psi=float(P.psi) + s * rng.randn(K),
        incl=np.clip(float(P.incl) + s * rng.randn(K), eps, np.pi - eps),
        phiref=float(P.phiref) + s * rng.randn(K),
        dist=setup.dist_mpc * np.exp(s * rng.randn(K)) * 1e6 * lsu.lsu_PC,
    )


def err_stats(lnL, lnL_ref):
    """Paired error statistics, reported BOTH over all points and over the inference-relevant
    band lnL_ref > max(lnL_ref) - RELEVANT_BAND."""
    assert_finite('lnL', lnL)
    assert_finite('lnL_ref', lnL_ref)
    d = lnL - lnL_ref
    band = lnL_ref > (np.max(lnL_ref) - RELEVANT_BAND)
    out = dict(maxabs=float(np.max(np.abs(d))), rms=float(np.sqrt(np.mean(d ** 2))),
               mean=float(np.mean(d)), lnL_max=float(np.max(lnL)), lnL_min=float(np.min(lnL)),
               n_band=int(np.sum(band)))
    if out['n_band'] > 0:
        db = d[band]
        out.update(maxabs_band=float(np.max(np.abs(db))),
                   rms_band=float(np.sqrt(np.mean(db ** 2))),
                   mean_band=float(np.mean(db)))
    else:
        out.update(maxabs_band=np.nan, rms_band=np.nan, mean_band=np.nan)
    return out


def make_Pvec(setup, pts, sl, deltaT):
    Pv = setup.Psig.manual_copy()
    for key in ('phi', 'theta', 'psi', 'incl', 'phiref', 'dist'):
        setattr(Pv, key, np.asarray(pts[key][sl]))
    Pv.tref = float(EVENT_TIME)
    Pv.deltaT = float(deltaT)
    return Pv


# ---------------------------------------------------------------------------
# band-limited (zero-pad FFT) upsampling
# ---------------------------------------------------------------------------
def bandlimited_upsample(x, M):
    """Interpolate complex x (..., N) onto an M-times finer grid by FFT zero padding.

    Exact for a periodic band-limited signal; y[..., ::M] reproduces x identically.
    The Nyquist bin (N even) is split symmetrically between +fNyq and -fNyq, which is the
    choice that preserves y[..., ::M] == x.  For a genuinely band-limited Q that bin is
    numerically zero anyway; the returned nyq_frac lets the caller check that.
    """
    x = np.asarray(x)
    N = x.shape[-1]
    X = np.fft.fft(x, axis=-1)
    Nf = N * M
    Y = np.zeros(x.shape[:-1] + (Nf,), dtype=np.complex128)
    h = N // 2
    Y[..., :h] = X[..., :h]
    Y[..., Nf - (N - h):] = X[..., h:]
    if N % 2 == 0:
        v = Y[..., Nf - h].copy()
        Y[..., Nf - h] = 0.5 * v
        Y[..., h] = 0.5 * v
    y = np.fft.ifft(Y, axis=-1) * M
    nyq_frac = float(np.max(np.abs(X[..., h])) / np.max(np.abs(X)))
    return y, nyq_frac


# ---------------------------------------------------------------------------
# lnL evaluation
# ---------------------------------------------------------------------------
def eval_lnL_t(setup, packs, pts, tvals, deltaT, time_interp, rho_arrays, chunk):
    """lnL_t of shape (K, len(tvals)), evaluated in chunks over extrinsic points."""
    K = len(pts['phi'])
    out = np.empty((K, len(tvals)), dtype=np.float64)
    for lo in range(0, K, chunk):
        sl = slice(lo, min(lo + chunk, K))
        Pv = make_Pvec(setup, pts, sl, deltaT)
        out[sl] = fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals, Pv, packs['lookupNK'], rho_arrays, packs['ctU'], packs['ctV'],
            packs['epoch'], Lmax=LMAX, xpy=np, return_lnLt=True, time_interp=time_interp)
    return out


def eval_reference(setup, packs, pts, tvals, M, chunk, rho_fine=None, stencil='cubic'):
    """Reference lnL_t on the coarse tvals grid, from an Mx finer (FFT zero-padded) Q grid.

    ``stencil`` is the lookup used ON THE FINE GRID.  'nearest' is the literal prescription
    (no interpolating stencil at all), but its residual error is only O(1/M) -- at M=32 that
    is still ~1/32 of the coarse 'nearest' error, which is NOT small compared to what we are
    trying to resolve.  'cubic' on the fine grid is O((1/M)^4) ~ 1e-6 of the coarse cubic
    error at M=32, i.e. six orders of magnitude below the differences being measured, so it
    is the default; the two are shown to agree by ref_convergence_ladder() below, which walks
    'nearest' up in M until it lands on the 'cubic' reference.
    """
    deltaT_f = setup.deltaT / M
    npts = len(tvals)
    npts_f = (npts - 1) * M + 1
    tvals_f = tvals[0] + np.arange(npts_f) * deltaT_f
    if rho_fine is None:
        rho_fine, _ = build_fine_rho(packs, M)
    lnL_t_f = eval_lnL_t(setup, packs, pts, tvals_f, deltaT_f, stencil, rho_fine,
                         max(1, chunk // 4))
    return lnL_t_f[:, ::M]


def build_fine_rho(packs, M):
    rho_fine = {}
    worst_roundtrip = 0.0
    worst_nyq = 0.0
    for det, arr in packs['rho'].items():
        y, nyq = bandlimited_upsample(arr, M)
        worst_roundtrip = max(worst_roundtrip,
                              float(np.max(np.abs(y[..., ::M] - arr)) / np.max(np.abs(arr))))
        worst_nyq = max(worst_nyq, nyq)
        rho_fine[det] = y
    return rho_fine, (worst_roundtrip, worst_nyq)


def time_marginalize(lnL_t, deltaT):
    """One lnL per extrinsic point: log int dt exp(lnL_t), Simpson weights, dx=deltaT.

    Uses fl.my_simps (the same quadrature the production reduction uses) applied here so
    every method gets bit-identical weights and the quadrature drops out of the comparison.
    """
    m = np.max(lnL_t, axis=-1, keepdims=True)
    return m[:, 0] + np.log(fl.my_simps(np.exp(lnL_t - m), dx=deltaT, axis=-1))


def ln_evidence(lnL):
    m = np.max(lnL)
    return m + np.log(np.mean(np.exp(lnL - m)))


# ---------------------------------------------------------------------------
# Q spectrum diagnostic
# ---------------------------------------------------------------------------
def q_spectrum_report(setup, packs):
    """How much of Q_lm's power actually lives near Nyquist?

    fNyq/fmax is only a proxy for the stencil's difficulty: Q(t) = <h_lm(t)|d> is band-limited
    by BOTH fMax and the template's own high-frequency cutoff, whichever is lower, and its
    power is further shaped by |h|^2/S.  A Tukey-windowed FFT of the stored Q window (windowed
    to suppress the leakage from the cut) gives the honest picture.
    """
    det = 'H1'
    arr = packs['rho'][det]
    N = arr.shape[1]
    w = lal.CreateTukeyREAL8Window(N, 0.2).data.data
    X = np.fft.fft(arr * w[None, :], axis=-1)
    f = np.fft.fftfreq(N, d=setup.deltaT)
    p = np.sum(np.abs(X) ** 2, axis=0)
    order = np.argsort(np.abs(f))
    fa = np.abs(f)[order]
    cum = np.cumsum(p[order]) / np.sum(p)
    out = {}
    for q in (0.99, 0.999, 0.9999):
        out['f%g' % q] = float(fa[np.searchsorted(cum, q)])
    # fraction of power above 1/2 and 3/4 of the *stencil-relevant* Nyquist
    fNyq = setup.fSample / 2.
    for frac in (0.25, 0.5, 0.75):
        thr = frac * fNyq
        out['pow>%.2ffNyq' % frac] = float(np.sum(p[np.abs(f) > thr]) / np.sum(p))
    return out



# ---------------------------------------------------------------------------
# achieved network SNR
# ---------------------------------------------------------------------------
def true_point_lnL_t(setup, packs, tvals, chunk, rho_fine=None, M=32):
    """lnL(t) at the TRUE extrinsic parameters (true sky/orientation/distance).

    The data are noiseless and the template is the injection, so max_t lnL_t = rho_net^2/2
    exactly.  Measuring the SNR this way uses the very machinery under test, so the SNR that
    labels each rung is the one that actually sets the lnL scale (not a nominal number).
    """
    P = setup.Psig
    pts = dict(phi=np.array([float(P.phi)]), theta=np.array([float(P.theta)]),
               psi=np.array([float(P.psi)]), incl=np.array([float(P.incl)]),
               phiref=np.array([float(P.phiref)]),
               dist=np.array([setup.dist_mpc * 1e6 * lsu.lsu_PC]))
    return eval_reference(setup, packs, pts, tvals, M, chunk, rho_fine=rho_fine,
                          stencil=REF_STENCIL)


def network_snr(setup, packs, tvals, chunk, rho_fine=None, n_phiref=32, n_psi=8):
    """SNR_lik = sqrt(2 max_t max_{phiref,psi} lnL) at the true sky, inclination and distance.

    MAXIMISED over the phase/polarization pair rather than evaluated at the nominal injected
    values, and that is not a nicety.  RIFT's SEOBNR mode decomposition (hlmoft ->
    SimIMRSpinAlignedEOBModes) carries a phase convention that differs from the one
    non_herm_hoff uses to build the injection, so at the NOMINAL true point SEOBNRv4 scores
    lnL = -43 while the same data have an optimal <d|d> SNR of 172.  Maximising over the two
    degenerate angles recovers SNR_lik/SNR_direct = 0.99 for SEOBNRv4 and 0.96 for TaylorT4:
    the offset is purely a convention, the template is not corrupted, and nothing about the
    PAIRED stencil comparison depends on it (same Q, same points, only the stencil varies).
    Without this the SEOBNRv4 distance normalisation is nonsense (sqrt of a negative number).
    """
    P = setup.Psig
    ph = np.repeat(np.linspace(0, 2 * np.pi, n_phiref, endpoint=False), n_psi)
    ps = np.tile(np.linspace(0, np.pi, n_psi, endpoint=False), n_phiref)
    n = n_phiref * n_psi
    pts = dict(phi=np.full(n, float(P.phi)), theta=np.full(n, float(P.theta)),
               psi=ps, incl=np.full(n, float(P.incl)), phiref=ph,
               dist=np.full(n, setup.dist_mpc * 1e6 * lsu.lsu_PC))
    lnL_t = eval_reference(setup, packs, pts, tvals, 32, chunk, rho_fine=rho_fine,
                           stencil=REF_STENCIL)
    peak = float(np.max(lnL_t))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("peak lnL over the (phiref,psi) grid is %r -- cannot define an SNR"
                           % peak)
    return float(np.sqrt(2.0 * peak))


def network_snr_direct(setup):
    """Independent cross-check of the network SNR: sqrt(sum_det <d|d>) from lsu.ComplexIP
    on the same (noiseless) data and analytic PSD, with no likelihood machinery involved."""
    tot = 0.0
    for det, d in setup.data_dict.items():
        IP = lsu.ComplexIP(setup.fmin, setup.fmax, 1. / 2. / setup.deltaT, d.deltaF,
                           setup.psd_dict[det], True, False, 0.)
        tot += float(np.abs(IP.ip(d, d)))
    return float(np.sqrt(tot))


def assert_finite(name, x):
    bad = int(np.sum(~np.isfinite(x)))
    if bad:
        raise RuntimeError("%s: %d non-finite lnL values -- refusing to report a max| | over "
                           "them" % (name, bad))
    return bad


def ess_fraction(lnL):
    """Effective sample fraction of the lnZ estimator, so the reader can see when lnZ is
    dominated by a single point (which it always is at very high SNR)."""
    w = np.exp(lnL - np.max(lnL))
    return float(np.sum(w) ** 2 / np.sum(w ** 2) / len(w))


# ---------------------------------------------------------------------------
# SNR ladder (near-Nyquist configuration A)
# ---------------------------------------------------------------------------
def run_snr_ladder(label, fSample, fmax, m1, m2, fmin, dist0, snr_targets, K, seeds,
                   t_half, M_ref, M_check, t_window, chunk, approx=None):
    """Configuration A across an SNR ladder.

    A stencil makes a fixed RELATIVE error in Q(t).  lnL ~ SNR^2, so the ABSOLUTE lnL error
    is predicted to grow as SNR^2 -- a difference that is invisible at demo SNRs need not be
    invisible at 3G SNRs.  SNR is varied by the injected distance only (same waveform, same
    stencil geometry); the extrinsic draw is dist = x_i * d_inj with x_i FIXED across rungs,
    so a clean SNR^2 scaling is what the null hypothesis predicts.
    """
    t0 = time.time()
    print("=" * 100)
    print("SNR LADDER  %s : fSample=%g fmax=%g  fNyq/fmax=%.3g  m1=%g m2=%g fmin=%g"
          % (label, fSample, fmax, (fSample / 2.) / fmax, m1, m2, fmin))
    sys.stdout.flush()

    probe = Setup(label, fSample, fmax, m1, m2, fmin, t_window, dist_mpc=dist0, approx=approx)
    npts_half = int(round(t_half * fSample))
    npts = 2 * npts_half + 1
    tvals = (np.arange(npts) - npts_half) * probe.deltaT
    rho_probe = network_snr(probe, probe.packs, tvals, chunk)
    print("  SNR CONVENTION: rungs are labelled by SNR_lik = sqrt(2 x peak lnL at the true")
    print("    extrinsic point), i.e. the SNR the LIKELIHOOD actually attains -- that is the")
    print("    quantity that sets the lnL scale, so it is what translates these nats to a real")
    print("    event.  The optimal <d|d> network SNR of the same noiseless data is also shown;")
    print("    it is larger, because the Lmax=2 template the likelihood uses does not recover")
    print("    100%% of the injected strain (a pre-existing property of this test setup, not of")
    print("    the stencils, and it cancels in the paired stencil comparison).")
    print("  probe: d=%g Mpc -> SNR_lik %.4g  (optimal <d|d> SNR: %.4g)"
          % (dist0, rho_probe, network_snr_direct(probe)))
    del probe

    rows = []
    for target in snr_targets:
        d_inj = dist0 * rho_probe / float(target)
        setup = Setup(label, fSample, fmax, m1, m2, fmin, t_window, dist_mpc=d_inj, approx=approx)
        packs = setup.packs
        rho_fine, _ = build_fine_rho(packs, M_ref)
        rho = network_snr(setup, packs, tvals, chunk, rho_fine=rho_fine)
        rho_dir = network_snr_direct(setup)
        lnL_peak_true = 0.5 * rho ** 2

        acc = dict((st, []) for st in ('nearest', 'cubic', 'sinc'))
        accN = dict((st, []) for st in ('nearest', 'cubic', 'sinc'))
        lnZ = dict(ref=[], nearest=[], cubic=[], sinc=[])
        ess = []
        cloud_span = []
        for seed in seeds:
            for tag, pts, store in (('iso', draw_points(K, seed, d_inj), acc),
                                    ('near', draw_points_near_truth(K, seed, setup, rho),
                                     accN)):
                lnL_t_ref = eval_reference(setup, packs, pts, tvals, M_ref, chunk,
                                           rho_fine=rho_fine, stencil=REF_STENCIL)
                lnL_ref = time_marginalize(lnL_t_ref, setup.deltaT)
                assert_finite('reference', lnL_ref)
                if tag == 'iso':
                    lnZ['ref'].append(ln_evidence(lnL_ref))
                    ess.append(ess_fraction(lnL_ref))
                else:
                    cloud_span.append(float(np.max(lnL_ref) - np.min(lnL_ref)))
                for stencil in ('nearest', 'cubic', 'sinc'):
                    lnL = time_marginalize(
                        eval_lnL_t(setup, packs, pts, tvals, setup.deltaT, stencil,
                                   packs['rho'], chunk), setup.deltaT)
                    store[stencil].append(err_stats(lnL, lnL_ref))
                    if tag == 'iso':
                        lnZ[stencil].append(ln_evidence(lnL))
        if target == snr_targets[0]:
            lnL_t_r2 = eval_reference(setup, packs, pts, tvals, M_check, chunk,
                                      stencil=REF_STENCIL)
            print("  reference check at this rung: moves %.3g nats going M=%d->%d"
                  % (float(np.max(np.abs(time_marginalize(lnL_t_r2, setup.deltaT) - lnL_ref))),
                     M_ref, M_check))
        rows.append(dict(target=target, d_inj=d_inj, rho=rho, acc=acc, accN=accN, lnZ=lnZ,
                         ess=float(np.mean(ess)), cloud_span=float(np.mean(cloud_span))))
        print("  rung target SNR %5g -> d=%.4g Mpc, achieved SNR_lik %.5g "
              "(peak lnL at truth %.6g; optimal <d|d> SNR %.5g)  (%.0fs)"
              % (target, d_inj, rho, lnL_peak_true, rho_dir, time.time() - t0))
        sys.stdout.flush()
        del rho_fine, packs, setup

    # ---- tables ----
    print("")
    for tag, key, blurb in (
            ('ISOTROPIC', 'acc',
             'whole sky, dist in [0.5,4]x d_inj -- includes huge-negative-lnL samples'),
            ('NEAR-TRUTH', 'accN',
             'cloud about the injection with all offsets scaled as 1/SNR')):
        print("")
        print("  SNR LADDER, %s point set (%s)" % (tag, blurb))
        print("  %d points x %d seeds per rung, paired across stencils" % (K, len(seeds)))
        print("  %-8s %8s %11s %11s %11s %11s %12s %12s" %
              ("stencil", "SNR_lik", "max|dlnL|", "RMS dlnL", "max/SNR^2", "RMS/SNR^2",
               "max(lnL)", "min(lnL)"))
        for r in rows:
            for stencil in ('nearest', 'cubic', 'sinc'):
                A = r[key][stencil]
                mx = max(x['maxabs'] for x in A)
                rms = float(np.mean([x['rms'] for x in A]))
                print("  %-8s %8.4g %11.4g %11.4g %11.4g %11.4g %12.6g %12.6g" %
                      (stencil, r['rho'], mx, rms, mx / r['rho'] ** 2, rms / r['rho'] ** 2,
                       max(x['lnL_max'] for x in A), min(x['lnL_min'] for x in A)))
            print("      (lnZ ESS fraction %.3g ; near-truth cloud lnL span %.4g nats)"
                  % (r['ess'], r['cloud_span']))
        print("")

    print("  EVIDENCE across the ladder: mean and seed-spread of lnZ - lnZ_ref (nats)")
    print("  %-8s %8s %14s %14s" % ("stencil", "SNR", "mean dlnZ", "spread"))
    for r in rows:
        for stencil in ('nearest', 'cubic', 'sinc'):
            d = np.array(r['lnZ'][stencil]) - np.array(r['lnZ']['ref'])
            print("  %-8s %8.4g %14.5g %14.4g" %
                  (stencil, r['rho'], float(np.mean(d)), float(np.max(d) - np.min(d))))
        print("")

    # ---- power-law fit and the threshold SNRs ----
    print("  SCALING AND THRESHOLDS  (power-law fit err = C * SNR_lik^p over the ladder;")
    print("   a threshold below the lowest rung is an EXTRAPOLATION under the fitted law)")
    rho_arr = np.array([r['rho'] for r in rows])

    def _fit(y, name):
        p_fit, logC = np.polyfit(np.log(rho_arr), np.log(y), 1)
        C = np.exp(logC)
        print("    %-42s : p = %.3f  ->  0.1 nat at SNR %.4g, 1 nat at SNR %.4g"
              % (name, p_fit, (0.1 / C) ** (1. / p_fit), (1.0 / C) ** (1. / p_fit)))

    for stencil in ('nearest', 'cubic', 'sinc'):
        for key, lab in (('acc', 'isotropic'), ('accN', 'near-truth')):
            _fit(np.array([max(x['maxabs'] for x in r[key][stencil]) for r in rows]),
                 "%s  max|dlnL|  (%s)" % (stencil, lab))
            _fit(np.array([float(np.mean([x['rms'] for x in r[key][stencil]]))
                           for r in rows]), "%s  RMS dlnL   (%s)" % (stencil, lab))
        _fit(np.array([max(1e-300, abs(float(np.mean(np.array(r['lnZ'][stencil]) -
                                                     np.array(r['lnZ']['ref'])))))
                       for r in rows]), "%s  |mean d lnZ| (isotropic)" % stencil)
    print("  total %.0f s" % (time.time() - t0))
    sys.stdout.flush()
    return rows



# ---------------------------------------------------------------------------
# mass ladder: does the f_ISCO bandwidth rule pick the right stencil?
# ---------------------------------------------------------------------------
# GW frequency at ISCO for total mass M (solar masses).  Kept here rather than imported:
# time_interp_choice used to export it, then stopped, and this script must not break when the
# module under study is edited.
F_ISCO_1MSUN_HZ = 4397.0


def chirp_time_s(m1_msun, m2_msun, f_low):
    """Leading-order (0PN) inspiral duration from f_low to coalescence, seconds."""
    m1 = m1_msun * lal.MTSUN_SI
    m2 = m2_msun * lal.MTSUN_SI
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    return (5. / 256.) * mc ** (-5. / 3.) * (np.pi * f_low) ** (-8. / 3.)


def segment_deltaF(m1, m2, fmin, base_T=4.0):
    """Segment length (as a deltaF) long enough to hold the whole signal from fmin.

    fmin is held FIXED across the mass ladder -- every mass is analysed in the same
    [fmin, fmax] band, so the only thing varying is the source.  That forces the segment to
    grow at low mass (a 2.6 Msun binary sweeps for ~90 s from 30 Hz), which is why deltaF is
    a per-mass quantity here and a constant everywhere else in this file.
    """
    need = 2.0 * chirp_time_s(m1, m2, fmin) + 4.0
    T = base_T
    while T < need:
        T *= 2.0
    return 1.0 / T, T


def run_mass_ladder(fSample, fmax, fmin, masses, target_snr, K, seeds, t_half, M_ref, M_check,
                    t_window, t_window_short, chunk, on_gpu_variants=(False, True),
                    approx=None):
    """Sweep total mass at FIXED srate/fmax and ask, per mass, which stencil actually wins and
    whether time_interp_choice predicts it.

    Every mass is normalised to the same SNR_lik (via the injected distance) so the nats are
    comparable down the ladder; the SNR^2 scaling needed to do that was measured, not assumed.
    """
    import RIFT.likelihood.time_interp_choice as tic
    t0 = time.time()
    print("=" * 110)
    print("MASS LADDER : approximant=%s  fSample=%g  fmax=%g  fmin=%g  "
          "(fNyq/fmax = %.3g for every mass)"
          % (approx or DEFAULT_APPROX, fSample, fmax, fmin, (fSample / 2.) / fmax))
    print("  every mass normalised to SNR_lik = %g so the nats are comparable down the ladder"
          % target_snr)
    print("  selector under test: %s" % tic.__file__)
    sys.stdout.flush()

    rows = []
    for m_total in masses:
        if abs(m_total - 2.6) < 1e-9:
            m1, m2 = 1.3, 1.3
        else:
            m1 = m2 = m_total / 2.0
        dF, T_seg = segment_deltaF(m1, m2, fmin)
        tau = chirp_time_s(m1, m2, fmin)

        try:
            probe = Setup('probe', fSample, fmax, m1, m2, fmin, t_window, dist_mpc=200.,
                          deltaF=dF, approx=approx)
        except Exception as exc:
            print("  M=%6.1f : SKIPPED -- %s cannot be generated at srate %g: %s"
                  % (m_total, approx or DEFAULT_APPROX, fSample, str(exc)[:120]))
            sys.stdout.flush()
            continue
        npts_half = int(round(t_half * fSample))
        npts = 2 * npts_half + 1
        tvals = (np.arange(npts) - npts_half) * probe.deltaT
        rho_probe = network_snr(probe, probe.packs, tvals, chunk)
        del probe
        d_inj = 200. * rho_probe / float(target_snr)

        setup = Setup('M%g' % m_total, fSample, fmax, m1, m2, fmin, t_window,
                      dist_mpc=d_inj, deltaF=dF, approx=approx)
        packs = setup.packs
        spec = q_spectrum_report(setup, packs)
        rho_fine, (rt, nyq) = build_fine_rho(packs, M_ref)
        rho = network_snr(setup, packs, tvals, chunk, rho_fine=rho_fine)
        rho_dir = network_snr_direct(setup)
        check_bounds(setup, packs, seeds[:1], K, tvals, npts, M_check, d_inj)

        acc = dict((st, []) for st in ('nearest', 'cubic', 'sinc'))
        floor = dict((st, np.nan) for st in ('nearest', 'cubic', 'sinc'))
        lnZ = dict(ref=[], nearest=[], cubic=[], sinc=[])
        for seed in seeds:
            pts = draw_points(K, seed, d_inj)
            lnL_ref = time_marginalize(
                eval_reference(setup, packs, pts, tvals, M_ref, chunk, rho_fine=rho_fine,
                               stencil=REF_STENCIL), setup.deltaT)
            lnZ['ref'].append(ln_evidence(lnL_ref))
            lnL_by_stencil = {}
            for stencil in ('nearest', 'cubic', 'sinc'):
                lnL = time_marginalize(
                    eval_lnL_t(setup, packs, pts, tvals, setup.deltaT, stencil, packs['rho'],
                               chunk), setup.deltaT)
                lnL_by_stencil[stencil] = lnL
                acc[stencil].append(err_stats(lnL, lnL_ref))
                lnZ[stencil].append(ln_evidence(lnL))
            if seed == seeds[0]:
                ref2 = time_marginalize(
                    eval_reference(setup, packs, pts, tvals, M_check, chunk,
                                   stencil=REF_STENCIL), setup.deltaT)
                ref_move = float(np.max(np.abs(ref2 - lnL_ref)))
                packs_s = setup.alternate_window_packs(t_window_short)
                ref_s = time_marginalize(
                    eval_reference(setup, packs_s, pts, tvals, M_ref, chunk,
                                   stencil=REF_STENCIL), setup.deltaT)
                wrap_move = float(np.max(np.abs(ref_s - lnL_ref)))
                del packs_s
                # PER-STENCIL REFERENCE FLOOR.  The reference is built by zero-pad-FFT
                # interpolating a CUT of rho(t), which treats the cut as periodic; the
                # resulting wrap (Gibbs) error is a property of the REFERENCE and cannot be
                # seen by the M -> 2M check, which shares it.  Re-scoring the SAME stencil
                # lnL values against a reference built from a shorter stored window changes
                # only that artifact, so the shift is a direct per-stencil error floor.  This
                # costs nothing: the stencil lnL values are already in hand.  Any entry in
                # column B at or below its floor is an UPPER BOUND, not a measurement.
                for stencil in ('nearest', 'cubic', 'sinc'):
                    d_long = lnL_by_stencil[stencil] - lnL_ref
                    d_short = lnL_by_stencil[stencil] - ref_s
                    floor[stencil] = abs(float(np.max(np.abs(d_short)))
                                         - float(np.max(np.abs(d_long))))

        # The shipped selector API is in flux (automatic selection was removed after the
        # TaylorT4 ladder).  Query it if it is still there; otherwise report no prediction
        # rather than inventing one.
        preds = {}
        for on_gpu in on_gpu_variants:
            chooser = getattr(tic, 'choose_time_interp_stencil', None)
            if chooser is None:
                preds[on_gpu] = (None, None, None)
            else:
                preds[on_gpu] = chooser(fSample, fmax, on_gpu=on_gpu, m_total_msun=m_total)
        # PSD-based bandwidth estimator (RIFT.misc.psd_bandwidth), evaluated on the SAME
        # analytic ZDHP PSD and the same [fmin, fmax] this measurement uses, at each of the
        # quantiles its calibration table quotes.  This is the estimator that is meant to
        # replace f_ISCO, and its calibration currently rests on TaylorT4 bandwidths.
        psd_est = {}
        try:
            import RIFT.misc.psd_bandwidth as pbw
            _f = np.arange(1, int(fSample / 2)) * 1.0
            _p = np.array([lalsim.SimNoisePSDaLIGOZeroDetHighPower(x) for x in _f])
            for q in (0.95, 0.99, 0.9999):
                psd_est[q] = pbw.bandwidth_from_psd(_f, _p, fmin, fmax,
                                                    m_total_msun=m_total, quantile=q)
        except Exception as exc:
            psd_est = {'error': str(exc)[:80]}
        rows.append(dict(M=m_total, m1=m1, m2=m2, T_seg=T_seg, tau=tau, d_inj=d_inj, rho=rho,
                         spec=spec, acc=acc, floor=floor, lnZ=lnZ, preds=preds,
                         f_isco=F_ISCO_1MSUN_HZ / m_total,
                         f_q_rule=(tic.q_bandwidth_hz(fmax, m_total)
                                   if hasattr(tic, 'q_bandwidth_hz')
                                   else min(fmax, F_ISCO_1MSUN_HZ / m_total)),
                         psd_est=psd_est,
                         ref_move=ref_move, wrap_move=wrap_move, upsample_rt=rt))
        print("  M=%6.1f  (%g+%g)  T_seg=%gs tau=%.3gs wf=%.3gs edge=%.1e  d=%.4g Mpc  SNR_lik=%.4g (direct %.4g, ratio %.3f)  "
              "f_Q(99.99%%)=%.1f Hz  f_RD~%.0f Hz  (%.0fs)"
              % (m_total, m1, m2, T_seg, tau, setup.wf_duration, setup.edge_fraction,
                 d_inj, rho, rho_dir, rho / rho_dir,
                 spec['f0.9999'], 16000. / m_total, time.time() - t0))
        sys.stdout.flush()
        del rho_fine, packs, setup

    # ---------------- report ----------------
    print("")
    print("  A. MEASURED Q BANDWIDTH vs THE f_ISCO BOUND USED BY THE RULE")
    print("  %6s %10s %10s %10s %12s %12s %11s %11s" %
          ("M/Msun", "f 99%", "f 99.9%", "f 99.99%", "f_ISCO=4397/M", "f_Q(rule)",
           "meas/f_ISCO", "fNyq/f_Q"))
    for r in rows:
        print("  %6.1f %10.1f %10.1f %10.1f %12.1f %12.1f %11.3g %11.4g" %
              (r['M'], r['spec']['f0.99'], r['spec']['f0.999'], r['spec']['f0.9999'],
               r['f_isco'], r['f_q_rule'], r['spec']['f0.9999'] / r['f_isco'],
               (fSample / 2.) / r['f_q_rule']))

    print("")
    print("  B. PAIRED STENCIL ERROR vs THE EXACT REFERENCE (nats; %d points x %d seeds; "
          "all masses at SNR_lik=%g)" % (K, len(seeds), target_snr))
    print("  %6s | %19s | %19s | %19s | %9s" %
          ("M/Msun", "nearest max / RMS", "cubic   max / RMS", "sinc    max / RMS",
           "cubic/sinc"))
    for r in rows:
        cells = []
        mx = {}
        rms = {}
        for st in ('nearest', 'cubic', 'sinc'):
            mx[st] = max(x['maxabs'] for x in r['acc'][st])
            rms[st] = float(np.mean([x['rms'] for x in r['acc'][st]]))
            flag = '<' if mx[st] <= 3.0 * r['floor'][st] else ' '
            cells.append("%s%8.4g /%9.4g" % (flag, mx[st], rms[st]))
        print("  %6.1f | %s | %s | %s | %9.4g" %
              (r['M'], cells[0], cells[1], cells[2], mx['cubic'] / mx['sinc']))
    print("  '<' marks an entry within 3x of its own reference floor (column E): an UPPER "
          "BOUND, not a measurement.")

    print("")
    print("  C. WHO WINS, WHAT THE RULE PREDICTS, AND WHETHER IT AGREES")
    print("  %6s %8s %10s %12s %10s %10s %8s | %10s %8s" %
          ("M/Msun", "fNyq/f_Q", "winner", "margin(max)", "margin(RMS)", "rule CPU",
           "agree", "rule GPU", "agree"))
    disagreements = []
    for r in rows:
        mx = dict((st, max(x['maxabs'] for x in r['acc'][st]))
                  for st in ('nearest', 'cubic', 'sinc'))
        rms = dict((st, float(np.mean([x['rms'] for x in r['acc'][st]])))
                   for st in ('nearest', 'cubic', 'sinc'))
        winner = 'cubic' if mx['cubic'] < mx['sinc'] else 'sinc'
        loser = 'sinc' if winner == 'cubic' else 'cubic'
        margin_mx = mx[loser] / mx[winner]
        margin_rms = rms[loser] / rms[winner]
        line = []
        for on_gpu in on_gpu_variants:
            pred, ov, thr = r['preds'][on_gpu]
            if pred is None:
                line.append(('n/a', True))
                continue
            ok = (pred == winner)
            line.append((pred, ok))
            if not ok:
                disagreements.append((r['M'], 'GPU' if on_gpu else 'CPU', pred, winner,
                                      margin_mx, margin_rms))
        print("  %6.1f %8.4g %10s %12.4g %10.4g %10s %8s | %10s %8s" %
              (r['M'], (fSample / 2.) / r['f_q_rule'], winner, margin_mx, margin_rms,
               line[0][0], "yes" if line[0][1] else "NO", line[1][0],
               "yes" if line[1][1] else "NO"))

    print("")
    print("  C2. PSD-BASED BANDWIDTH ESTIMATOR (RIFT.misc.psd_bandwidth) vs THIS MEASUREMENT")
    print("  estimate/measured at each quantile, and fNyq/measured with the winner")
    print("  %6s %10s | %9s %9s %9s | %9s %9s %9s | %10s %8s" %
          ("M/Msun", "meas f_Q", "est q.95", "est q.99", "est q1e-4",
           "rat .95", "rat .99", "rat 1e-4", "fNyq/meas", "winner"))
    for r in rows:
        meas = r['spec']['f0.9999']
        e = r.get('psd_est', {})
        vals = [e.get(q) for q in (0.95, 0.99, 0.9999)]
        mx = dict((st, max(x['maxabs'] for x in r['acc'][st])) for st in ('cubic', 'sinc'))
        win = 'cubic' if mx['cubic'] < mx['sinc'] else 'sinc'
        def _f(v):
            return ("%9.1f" % v) if isinstance(v, float) else "      n/a"
        def _r(v):
            return ("%9.3g" % (v / meas)) if isinstance(v, float) else "      n/a"
        print("  %6.1f %10.1f | %s %s %s | %s %s %s | %10.4g %8s" %
              (r['M'], meas, _f(vals[0]), _f(vals[1]), _f(vals[2]),
               _r(vals[0]), _r(vals[1]), _r(vals[2]), (fSample / 2.) / meas, win))

    print("")
    print("  D. EVIDENCE: mean +- seed-spread of lnZ - lnZ_ref (nats)")
    print("  %6s %22s %22s %22s" % ("M/Msun", "nearest", "cubic", "sinc"))
    for r in rows:
        cells = []
        for st in ('nearest', 'cubic', 'sinc'):
            d = np.array(r['lnZ'][st]) - np.array(r['lnZ']['ref'])
            cells.append("%11.4g +-%9.3g" % (float(np.mean(d)),
                                             float(np.max(d) - np.min(d))))
        print("  %6.1f %22s %22s %22s" % (r['M'], cells[0], cells[1], cells[2]))

    print("")
    print("  E. REFERENCE VALIDITY PER MASS (must stay far below column B)")
    print("  %6s %14s %12s | %11s %11s %11s" %
          ("M/Msun", "M32->64", "wrap(ref)", "floor:nearest", "floor:cubic", "floor:sinc"))
    for r in rows:
        print("  %6.1f %14.4g %12.4g | %11.4g %11.4g %11.4g" %
              (r['M'], r['ref_move'], r['wrap_move'], r['floor']['nearest'],
               r['floor']['cubic'], r['floor']['sinc']))

    print("")
    sinc_ov = [(fSample / 2.) / r['spec']['f0.9999'] for r in rows
               if max(x['maxabs'] for x in r['acc']['sinc'])
               < max(x['maxabs'] for x in r['acc']['cubic'])]
    cub_ov = [(fSample / 2.) / r['spec']['f0.9999'] for r in rows
              if max(x['maxabs'] for x in r['acc']['sinc'])
              >= max(x['maxabs'] for x in r['acc']['cubic'])]
    print("  THRESHOLD BRACKET on fNyq / measured-99.99%%-bandwidth, from THIS ladder:")
    print("    sinc wins up to  %s" % ("%.3f" % max(sinc_ov) if sinc_ov else "(no sinc wins)"))
    print("    cubic wins from  %s" % ("%.3f" % min(cub_ov) if cub_ov else "(no cubic wins)"))
    print("")
    if disagreements:
        print("  ** RULE MIS-SELECTS at %d (mass, backend) points:" % len(disagreements))
        for (M, be, pred, win, mmx, mrms) in disagreements:
            print("     M=%g Msun %s: rule says %s, measurement says %s "
                  "(penalty %.3gx on max, %.3gx on RMS)" % (M, be, pred, win, mmx, mrms))
    else:
        print("  ** RULE AGREES WITH THE MEASUREMENT AT EVERY MASS AND BOTH BACKENDS.")
    print("  total %.0f s" % (time.time() - t0))
    sys.stdout.flush()
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def ref_convergence_ladder(setup, packs, pts, tvals, lnL_ref, Ms, chunk, K_sub):
    """Walk the LITERAL prescription ('nearest' on an Mx fine grid) up in M and show it
    converging onto the primary reference.  Done on a subset of points to keep it cheap."""
    sub = {k: v[:K_sub] for k, v in pts.items()}
    out = []
    for M in Ms:
        lnL_t = eval_reference(setup, packs, sub, tvals, M, chunk, stencil='nearest')
        d = time_marginalize(lnL_t, setup.deltaT) - lnL_ref[:K_sub]
        out.append((M, float(np.max(np.abs(d))), float(np.sqrt(np.mean(d ** 2)))))
    return out


def run_config(label, fSample, fmax, m1, m2, fmin, dist_mpc, K, seeds, t_half, M_ref,
               M_check, t_window, t_window_short, chunk, ladder_Ms=(32, 64, 128, 256),
               approx=None):
    t0 = time.time()
    print("=" * 100)
    print("CONFIG %s : fSample=%g fmax=%g  fNyq/fmax=%.3g   source m1=%g m2=%g fmin=%g "
          "dist=%g Mpc" % (label, fSample, fmax, (fSample / 2.) / fmax, m1, m2, fmin, dist_mpc))
    sys.stdout.flush()

    try:
        setup = Setup(label, fSample, fmax, m1, m2, fmin, t_window, dist_mpc=dist_mpc,
                      approx=approx)
    except Exception as exc:
        # Almost always: an IMR model asked for below the mass where its ringdown fits under
        # Nyquist.  Say what to do instead of emitting a raw lal domain error, and skip this
        # configuration rather than aborting the remaining ones.
        print("\n  CONFIG %s SKIPPED -- %s could not be generated at srate %g for M = %.4g Msun."
              % (label, approx or DEFAULT_APPROX, fSample, m1 + m2))
        print("    %s" % (str(exc)[:160],))
        print("    Raise this config's srate to 16384, or re-run with --approx TaylorT4 -- but "
              "note\n    inspiral-only results named the WRONG stencil at M = 9, 10 and 20, so do "
              "not use\n    them to support stencil guidance. See "
              "RIFT/likelihood/DESIGN_q_window_stencil.md.")
        return None
    packs = setup.packs
    n_time = packs['rho']['H1'].shape[1]
    print("  precompute: %.1fs   n_time(stored Q window)=%d  (=%.4g s)  SNR guess=%.4g"
          % (time.time() - t0, n_time, n_time * setup.deltaT, packs['snr']))

    spec = q_spectrum_report(setup, packs)
    print("  Q(t) spectrum (Tukey-windowed, H1, all modes):  f(99%%)=%.1f Hz  f(99.9%%)=%.1f Hz "
          " f(99.99%%)=%.1f Hz   frac power >0.25fNyq=%.2e  >0.5fNyq=%.2e  >0.75fNyq=%.2e"
          % (spec['f0.99'], spec['f0.999'], spec['f0.9999'],
             spec['pow>0.25fNyq'], spec['pow>0.50fNyq'], spec['pow>0.75fNyq']))

    npts_half = int(round(t_half * setup.fSample))
    npts = 2 * npts_half + 1
    tvals = (np.arange(npts) - npts_half) * setup.deltaT
    print("  eval time grid: npts=%d, +-%.4g s about tref" % (npts, npts_half * setup.deltaT))

    # ---- window bounds: make sure no stencil (or the fine reference) ever runs off the
    # stored Q window, which would silently zero-fill.
    check_bounds(setup, packs, seeds, K, tvals, npts, M_check, dist_mpc)

    results = {}
    lnZ = {}
    rho_fine, (rt, nyq) = build_fine_rho(packs, M_ref)
    print("  upsample check (M=%d): max|y[::M]-x|/max|x| = %.2e ; |X[Nyq]|/max|X| = %.2e"
          % (M_ref, rt, nyq))
    for seed in seeds:
        pts = draw_points(K, seed, dist_mpc)

        lnL_t_ref = eval_reference(setup, packs, pts, tvals, M_ref, chunk,
                                   rho_fine=rho_fine, stencil=REF_STENCIL)
        lnL_ref = time_marginalize(lnL_t_ref, setup.deltaT)
        lnZ.setdefault('ref', []).append(ln_evidence(lnL_ref))

        for stencil in ('nearest', 'cubic', 'sinc'):
            lnL_t = eval_lnL_t(setup, packs, pts, tvals, setup.deltaT, stencil,
                               packs['rho'], chunk)
            lnL = time_marginalize(lnL_t, setup.deltaT)
            st = err_stats(lnL, lnL_ref)
            st['maxabs_lnLt'] = float(np.max(np.abs(lnL_t - lnL_t_ref)))
            results.setdefault(stencil, []).append(st)
            lnZ.setdefault(stencil, []).append(ln_evidence(lnL))
        print("  seed %d done (%.0fs elapsed)" % (seed, time.time() - t0))
        sys.stdout.flush()

        if seed == seeds[0]:
            # ---- reference validity (a): does the reference move when M -> M_check?
            lnL_t_ref2 = eval_reference(setup, packs, pts, tvals, M_check, chunk,
                                        stencil=REF_STENCIL)
            lnL_ref2 = time_marginalize(lnL_t_ref2, setup.deltaT)
            ref_move = float(np.max(np.abs(lnL_ref2 - lnL_ref)))
            ref_move_lnZ = abs(ln_evidence(lnL_ref2) - ln_evidence(lnL_ref))
            # ---- reference validity (b): wrap artifact, from a HALF-length stored Q window
            packs_short = setup.alternate_window_packs(t_window_short)
            lnL_t_ref_s = eval_reference(setup, packs_short, pts, tvals, M_ref, chunk,
                                         stencil=REF_STENCIL)
            lnL_ref_s = time_marginalize(lnL_t_ref_s, setup.deltaT)
            wrap_move = float(np.max(np.abs(lnL_ref_s - lnL_ref)))
            del packs_short
            ladder = ref_convergence_ladder(setup, packs, pts, tvals, lnL_ref,
                                            ladder_Ms, chunk, min(K, 200))

    print("")
    print("  RESULTS  (differences in nats; lnL is the time-marginalized log likelihood)")
    print("  ALL %d points per seed.  NOTE max(lnL) vs min(lnL): the isotropic draw contains "
          "points with" % K)
    print("  huge NEGATIVE lnL (small distance, mismatched sky); they carry no posterior "
          "weight but do")
    print("  carry a large |kappa|, so the all-points max| | is a pessimistic bound, not an "
          "inference-relevant one.")
    print("  %-8s %12s %12s %12s %12s %13s %13s" %
          ("stencil", "max|dlnL|", "RMS dlnL", "mean dlnL", "max|dlnL_t|", "max(lnL)",
           "min(lnL)"))
    for stencil in ('nearest', 'cubic', 'sinc'):
        r = results[stencil]
        print("  %-8s %12.4g %12.4g %12.4g %12.4g %13.6g %13.6g" %
              (stencil, max(x['maxabs'] for x in r),
               float(np.mean([x['rms'] for x in r])),
               float(np.mean([x['mean'] for x in r])),
               max(x['maxabs_lnLt'] for x in r),
               max(x['lnL_max'] for x in r), min(x['lnL_min'] for x in r)))
    print("")
    print("  RESTRICTED to the inference-relevant band lnL_ref > max(lnL_ref) - %g "
          "(%s points/seed)" % (RELEVANT_BAND,
                                "/".join(str(x['n_band']) for x in results['cubic'])))
    print("  %-8s %12s %12s %12s" % ("stencil", "max|dlnL|", "RMS dlnL", "mean dlnL"))
    for stencil in ('nearest', 'cubic', 'sinc'):
        r = results[stencil]
        print("  %-8s %12.4g %12.4g %12.4g" %
              (stencil, max(x['maxabs_band'] for x in r),
               float(np.mean([x['rms_band'] for x in r])),
               float(np.mean([x['mean_band'] for x in r]))))

    print("")
    print("  REFERENCE VALIDITY  (primary reference = '%s' lookup on an M=%dx FFT-upsampled Q)"
          % (REF_STENCIL, M_ref))
    smallest = min(max(x['maxabs'] for x in results[s]) for s in ('nearest', 'cubic', 'sinc'))
    print("    reference moves by max %.4g nats going M=%d -> M=%d  "
          "(smallest stencil-vs-reference max|dlnL| = %.4g -> ratio %.3g)"
          % (ref_move, M_ref, M_check, smallest, ref_move / smallest if smallest else np.nan))
    print("    reference lnZ moves by %.4g nats going M=%d -> M=%d" % (ref_move_lnZ, M_ref, M_check))
    print("    reference moves by max %.4g nats when the stored Q window is halved "
          "(%.4g s -> %.4g s): this bounds the periodic-wrap (Gibbs) artifact"
          % (wrap_move, 2 * t_window, 2 * t_window_short))
    print("    literal prescription ('nearest' on the fine grid) vs this reference, on %d points:"
          % min(K, 200))
    for (M, mx, rms) in ladder:
        print("        M=%4d : max|dlnL| = %10.4g   RMS = %10.4g" % (M, mx, rms))

    print("")
    print("  EVIDENCE  lnZ = log(mean(exp(lnL-max)))+max over the SAME %d fixed points, "
          "%d seeds" % (K, len(seeds)))
    print("    reference lnZ per seed: %s" % np.array2string(np.array(lnZ['ref']), precision=6))
    print("  %-8s %14s %14s %14s" % ("stencil", "mean lnZ", "mean d lnZ", "spread(d lnZ)"))
    for stencil in ('nearest', 'cubic', 'sinc'):
        d = np.array(lnZ[stencil]) - np.array(lnZ['ref'])
        print("  %-8s %14.6f %14.4g %14.4g" %
              (stencil, float(np.mean(lnZ[stencil])), float(np.mean(d)),
               float(np.max(d) - np.min(d))))
    print("  total %.0f s" % (time.time() - t0))
    sys.stdout.flush()
    return results, lnZ


def check_bounds(setup, packs, seeds, K, tvals, npts, M_check, dist_mpc):
    """Assert every stencil window (incl. sinc's 8 taps/side and the finest reference grid)
    lies strictly inside the stored Q series -- otherwise the builders zero-fill silently."""
    a = fl.SINC_HALFWIDTH_DEFAULT
    gmst = float(lal.GreenwichMeanSiderealTime(EVENT_TIME))
    worst_lo, worst_hi = np.inf, np.inf
    for seed in seeds:
        pts = draw_points(K, seed, dist_mpc)
        for det in packs['rho']:
            loc = np.asarray(lalsim.DetectorPrefixToLALDetector(det).location)
            dt = fl.TimeDelayFromEarthCenter(loc, pts['phi'], pts['theta'], gmst, xpy=np)
            t_det = float(EVENT_TIME - float(packs['epoch'][det])) + dt
            n_time = packs['rho'][det].shape[1]
            for M in (1, M_check):
                s0 = (t_det + tvals[0]) / (setup.deltaT / M)
                i0 = np.floor(s0)
                worst_lo = min(worst_lo, float(np.min(i0)) - a + 1)
                worst_hi = min(worst_hi, n_time * M - float(np.max(i0)) - (npts - 1) * M - a)
    print("  window bounds: min margin below start = %.0f samples, above end = %.0f samples "
          "(both must be > 0)" % (worst_lo, worst_hi))
    assert worst_lo > 0 and worst_hi > 0, "evaluation window runs off the stored Q series"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=2000)
    ap.add_argument("--seeds", type=int, nargs='+', default=[101, 202, 303])
    ap.add_argument("--t-half", type=float, default=0.01,
                    help="half width of the lnL(t) evaluation window, seconds")
    ap.add_argument("--M-ref", type=int, default=32)
    ap.add_argument("--M-check", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--ladder-Ms", type=int, nargs='+', default=[32, 64, 128, 256])
    ap.add_argument("--dist-scale", type=float, default=1.0,
                    help="multiply every injected distance by this (lnL and dlnL both scale "
                         "as SNR^2, so this is the knob that rescales the whole table)")
    ap.add_argument("--only", type=str, default=None, help="run only this config label")
    ap.add_argument("--mode", choices=('grid', 'snr-ladder', 'mass-ladder'), default='grid')
    ap.add_argument("--masses", type=float, nargs='+',
                    default=[2.6, 5., 10., 20., 35., 55., 80., 120.])
    ap.add_argument("--mass-ladder-snr", type=float, default=100.)
    ap.add_argument("--mass-ladder-fmin", type=float, default=30.)
    ap.add_argument("--mass-ladder-srate", type=float, default=4096.)
    ap.add_argument("--approx", type=str, default=None,
                    help="lalsimulation approximant name. DEFAULT %s -- the IMR model behind the "
                         "guidance in RIFT/likelihood/DESIGN_q_window_stencil.md. Inspiral-only "
                         "models (TaylorT4) terminate at ISCO, understate the Q bandwidth by "
                         "2-3.7x and NAMED THE WRONG STENCIL at M = 9, 10 and 20; passing one "
                         "prints a warning. Note %s cannot be generated below M ~ 8 at srate "
                         "4096 -- use srate 16384 there, or accept the inspiral-only caveat."
                         % (DEFAULT_APPROX, DEFAULT_APPROX))
    ap.add_argument("--t-window", type=float, default=0.4,
                    help="half width of the STORED Q window (the reference is built from it)")
    ap.add_argument("--t-window-short", type=float, default=0.2,
                    help="shorter stored Q window used to bound the periodic-wrap artifact")
    ap.add_argument("--snr-targets", type=float, nargs='+',
                    default=[10., 30., 100., 300., 1000.])
    args = ap.parse_args()

    # (label, fSample, fmax, m1, m2, fmin, t_window, t_window_short)
    #
    # Two SOURCES are run through each sample-rate/fmax configuration on purpose.  fNyq/fmax
    # is the number the stencil chooser uses, but the quantity that actually sets the stencil's
    # difficulty is the bandwidth of Q(t) = <h_lm(t)|d>, which is limited by the TEMPLATE as
    # well as by fMax.  The 30+25 Msun system used by the existing slowrot tests has its ISCO
    # near 80 Hz, so at fmax=1700 its Q is nowhere near Nyquist no matter what fNyq/fmax says.
    # The 1.3+1.3 Msun system has ISCO near 1690 Hz, so it genuinely fills the band.  Both are
    # reported; neither is chosen after seeing the answer.
    configs = [
        ("A-heavy", 4096., 1700., 30., 25., 30., 200., 0.4, 0.2),
        ("B-heavy", 16384., 512., 30., 25., 30., 200., 0.4, 0.2),
        ("A-light", 4096., 1700., 1.3, 1.3, 150., 12., 0.4, 0.2),
        ("B-light", 16384., 512., 1.3, 1.3, 150., 12., 0.4, 0.2),
    ]
    if args.mode == 'mass-ladder':
        run_mass_ladder(args.mass_ladder_srate, 1700., args.mass_ladder_fmin, args.masses,
                        args.mass_ladder_snr, args.K, args.seeds, args.t_half, args.M_ref,
                        args.M_check, args.t_window, args.t_window_short, args.chunk,
                        approx=args.approx)
        return

    if args.mode == 'snr-ladder':
        # Near-Nyquist configuration A only (fNyq/fmax = 1.2), both sources.
        for (label, fS, fmax, m1, m2, fmin, dmpc, tw, tws) in configs:
            if not label.startswith('A'):
                continue
            if args.only and args.only not in label:
                continue
            run_snr_ladder(label, fS, fmax, m1, m2, fmin, dmpc, args.snr_targets, args.K,
                           args.seeds, args.t_half, args.M_ref, args.M_check, tw, args.chunk, approx=args.approx)
        return

    for (label, fS, fmax, m1, m2, fmin, dmpc, tw, tws) in configs:
        if args.only and args.only not in label:
            continue
        run_config(label, fS, fmax, m1, m2, fmin, dmpc * args.dist_scale, args.K, args.seeds,
                   args.t_half, args.M_ref, args.M_check, tw, tws, args.chunk,
                   ladder_Ms=args.ladder_Ms, approx=args.approx)


if __name__ == "__main__":
    main()
