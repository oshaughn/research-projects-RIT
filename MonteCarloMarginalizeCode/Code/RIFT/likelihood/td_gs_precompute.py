#
# td_gs_precompute.py
#
# Time-domain Precompute code path for ILE (--internal-ile-inner-product td-gs).
#
# Everything here stays in the time domain: the data d is loaded as a time series (no
# Tukey window), the template modes h_lm(t) come from lalsimutils.hlmoft with
# conditioning off (no taper), and neither is ever put through a windowed/periodic
# frequency-domain inner product.  The noise model is a Toeplitz covariance
# C = Toeplitz(ACF) on each contiguous analysis piece (tdanalysis style: a square window
# == hard sample selection, pieces independent, so C is block-diagonal over pieces).  The
# ACF comes from a tdanalysis [tau, R] .npy file or from the PSD through tdanalysis'
# PSDToAutocorr (walled outside the band, explicit condition number).
#
# The Toeplitz algebra is tdanalysis' (the names of the final tdanalysis 1.1 release,
# docs/toeplitz_api.md there):
#     tdanalysis.forms.toeplitz_solver.ToeplitzSolver   C_p^{-1} on one piece: a Cholesky
#                                                       factor for short pieces, the
#                                                       Gohberg-Semencul (GS) form for
#                                                       long ones (generators by refined
#                                                       Levinson solves of the ACF)
#     tdanalysis.forms.sliding.sliding_data_term        Q of templates sliding through a piece
#     tdanalysis.forms.sliding.sliding_self_terms       U, V of templates sliding through a piece
#     tdanalysis.forms.sliding.is_turn_on_inside_window the template turn-on check
#     ToeplitzSolver.apply_Cinv                         C_p^{-1} d_p, once per piece and job
#     ToeplitzSolver.whiten                             L^{-1} d_p for the whiteness check
#     tdanalysis.utils.filters.highpass_zero_phase      the optional data high-pass
#                                                       (--td-highpass; re-exported here)
#     tdanalysis.utils.filters.fir_*                    calibration realizations as exact FIR
#                                                       filters on the template
#     tdanalysis.utils.subsample_align.fractional_shift the sub-sample template placement
#                                                       (--td-subsample)
# This module adds what is RIFT's: lal series and epochs, the (l,m) mode bookkeeping, the
# analysis pieces (in units of M_ref on the ILE command line), RIFT's storage grid of
# shifts, the sum over pieces, the turn-on refusal and its message, the calibration
# conventions (RIFT's packing, correction type, conjugate phase), the interleaving of the
# sub-sample phases onto the fine grid, and the output contract below.
#
# Per detector and ILE job (built ONCE, TDGSDetectorContext):
#     w_p = C_p^{-1} d_p                 for every piece p (ToeplitzSolver of piece p)
# Per intrinsic point (TDGSDetectorContext.precompute), for every shift k on RIFT's
# storage grid (k = N_shift + i, i = 0..N_window-1; template sample j sits at data
# sample j + k):
#     Q_lm(k)        = sum_p  conj(h_lm,k |_p)^T  w_p
#     U_lm,l'm'(k)   = sum_p  conj(h_lm,k |_p)^T  C_p^{-1}  h_l'm',k |_p
#     V_lm,l'm'(k)   = sum_p        (h_lm,k |_p)^T  C_p^{-1}  h_l'm',k |_p
# where h|_p is the shifted template cut to piece p.  U follows lsu.ComplexIP's
# conj(h1) h2 convention; V is RIFT's crossTermsV = <conj(h_lm) | h_l'm'>.  With
# tdanalysis' ACF convention conj(u)^T C^{-1} v IS the matched-filter inner product
# (ip_normalization = 1, see ToeplitzInnerProductContext.ip_normalization).
#
# Each piece p is the window [i0, i1) of the data: sliding_data_term and
# sliding_self_terms run on it with offset = i0 and the data-frame shifts, so their
# convention (template sample j at window sample j + k) is RIFT's, and the per-piece
# results are summed here.
#
# U(t), V(t) have four exact methods in tdanalysis (no truncation, interpolation or
# scalar-U shortcut; they differ only in cost and memory), chosen with --td-u-method:
#
#   "prefix"     windowed sums of one Gohberg-Semencul kernel correlation per mode, valid
#                when the shifted template ends inside the piece for every grid shift (or,
#                mirrored, starts inside it); four FFT correlations per mode, then
#                O(n + N_window) per mode pair.  Refused (ValueError) on a piece where the
#                condition fails.
#   "recursion"  the kernel correlations slid one sample at a time: O(n N_window) per
#                mode, any cut pattern, no per-shift solve.
#   "batched"    direct reference: the cut shifted templates stacked as columns, C^{-1}
#                applied once per mode (memory n x N_window per mode).
#   "chunked"    "batched" in column blocks under a memory bound (uv_block_bytes).
#
# "auto" (default): pieces on the Cholesky solver use "chunked" (exact there); GS pieces
# use "prefix" when its condition holds (the full segment in an ILE run, ringdown
# pieces) and "recursion" otherwise (a piece the template overruns at both ends, where
# the per-column FFT solves of the batched methods cost far more than the recursion).
# The method used on each piece is kept in TDGSDetectorContext.uv_routes and printed
# with verbose output.  "prefix" and "recursion" use the GS generators of the piece's
# solver, which tdanalysis computes from the ACF for Cholesky pieces too; on a badly
# conditioned ACF they agree with the Cholesky solve to ~cond(C) eps (~1e-9 at cond 1e8).
# FFTs appear only as exact zero-padded linear-correlation engines inside this Toeplitz
# algebra; nothing is windowed.
#
# Output contract (what the likelihood consumes):
#     rholms[mode]            lal.COMPLEX16TimeSeries, length N_window, deltaT, epoch =
#                             data.epoch - hlm.epoch + N_shift*deltaT -- exactly the series
#                             factored_likelihood.ComputeModeIPTimeSeries returns after its
#                             DataRollBins/Cut (the fractional part of (data.epoch -
#                             hlm.epoch)/deltaT is carried in the epoch, as on the FD code path;
#                             the template is only ever shifted by whole samples).
#     crossTerms[(p1,p2)]     complex ndarray of length N_window, aligned sample-by-sample
#     crossTermsV[(p1,p2)]    with rholms (entry i <-> shift k = N_shift + i).
#
# Calibration marginalization (TDCalibrationFilters, precompute(..., calibration=...)):
# each RIFT realization eta_c(f) (RIFT.calmarg.generate_realizations, RIFT's reversed
# two-sided packing, unity outside the band) becomes a real FIR filter K_c on the TEMPLATE,
# "calibrate then cut": (K_c h)[j] = sum_i c_c[i] h[j-i], a linear convolution of the
# zero-extended template, then cut to the pieces.  The data are untouched.  K_c is the
# N-tap filter whose response ON THE REALIZATION'S GRID f_m = m/(N deltaT) is eta_c exactly:
# its taps are the inverse DFT of eta_c (real, since eta_c(-f) = conj eta_c(f)), i = -N/2+1 ..
# N/2 (tdanalysis.utils.filters.fir_taps_from_response).  The FD code path multiplies by eta_c on that grid, i.e. convolves CIRCULARLY with the
# same taps; the TD code path convolves linearly, so the two agree exactly where the FD code path
# does not wrap around.  Because the template has compact support, a tap c_c[i] contributes
# to a piece sample only if |i| is at most the reach R = max over pieces and grid shifts of
# the distance from a piece sample to the template's support; the filter truncated to
# half-length L >= R gives the SAME values on the pieces as the N-tap filter (to rounding),
# so per intrinsic point L = min(R, N/2) is used (the full N taps when R >= N/2), with no
# tolerance anywhere (tdanalysis' fir_max_lag and fir_truncate_at_max_lag).
# --td-calibration-fir-half-length forces a shorter L (then an approximation; measured
# errors in docs/td_gs_route.md).  Everything is exact to rounding:
#     Q_c(t)        = (K_c h_t)|_pieces^H w = h_t^H (K_c^T w)       (w zero outside the pieces,
#                     so W_c = K_c^T w with the N-tap filter is one exact linear correlation per
#                     realization, formed once per realization set; Q_c(t) is then the exact
#                     correlation of the template's support with W_c, for every template)
#     U_c(t), V_c(t) = the exact U/V methods above on the filtered template g_c = K_c h, whose
#                     support is the template's extended by the taps' reach on each side
#                     (passed to the sliding forms as support=: the method selection uses
#                     it, and the rounding residue the convolution leaves outside it is ignored).
# Convention (td_calibration.py, test/td_calibration_study.md): RIFT's FD default -- the
# realization multiplying the DATA in RIFT's reversed packing -- equals, in the standard
# e^{-2 pi i f t} convention, the TEMPLATE filtered with eta_c at face value; this is the
# default here (correction_type "rift": C_c = eta_c for every detector).  RIFT's
# --calibration-conjugate-phase is C_c = conj(eta_c).  correction_type "bilby" uses
# bilby_pipe's per-detector semantics: C_c = eta_c for V1 ("template" type) and C_c =
# 1/eta_c for H1, L1, K1 ("data" type: eta maps calibrated data to theoretical strain).
#
# Sub-sample placement (subsample = K > 1, ILE --td-subsample K):
#     The series are returned on a grid of spacing deltaT/K (rholms deltaT = deltaT/K,
#     length K*N_window, same epoch): entry r <-> the template placed at shift
#     N_shift + r/K.  Phase j = r mod K is the template moved by a fraction of a sample
#     with tdanalysis' Lanczos resampler (tdanalysis.utils.subsample_align.fractional_shift,
#     half-width SHIFT_STENCIL = 16), by s_j = j/K for j/K <= 1/2 and s_j = j/K - 1 placed
#     one sample later otherwise -- the same residual-in-[-1/2, 1/2] decomposition
#     GWTransientTD._get_peak_and_target_indices uses, so where a sample's fractional
#     offset falls on the 1/K grid the two codes evaluate the identical resampled template.
#     Every phase
#     goes through the same exact Q/U/V methods as phase 0 (which is the K = 1 result,
#     bit for bit); Precompute costs K times the K = 1 cost.  Q is linear in the template,
#     so a fine-grid Q equals the Lanczos interpolation of the coarse Q; U and V are
#     quadratic, so interpolating them would not be equivalent -- they are recomputed from
#     the shifted templates.  The likelihood (factored_likelihood.
#     DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop, td_subsample=K) finds each
#     extrinsic sample's arrival time on the fine grid and gathers its time bins with
#     stride K, so the time marginalization keeps its coarse bins and Simpson spacing.
#     With calibration, each phase j runs the calibrated body above on its shifted template:
#     the Lanczos shift first (the shifted template as RIFT and GWTransientTD place it, same
#     length as the input), then the calibration FIR as a full linear convolution of that
#     template, then the cut and the exact U/V methods.  Both are linear time-invariant
#     filters, so the order matters only through the shift's cut at the template's ends;
#     this order is "calibration applied to the physically shifted signal", and identity
#     realizations give the uncalibrated fine grid to rounding.  The reach R and the exact
#     tap truncation are those of each shifted template, so every phase is exact by the
#     argument above.  The realization blocks of Q_c and the per-realization U_c(t), V_c(t)
#     carry the fine grid (n_cal blocks of K*N_window entries, r = K*i + j within a block),
#     and the likelihood's calibration loop gathers them with the same stride K.
#

import numpy as np
import scipy.signal

import lal

import RIFT.lalsimutils as lsu


def _tdanalysis_location():
    try:
        import tdanalysis
        return "%s %s" % (getattr(tdanalysis, "__version__", "?"), getattr(tdanalysis, "__file__", "?"))
    except Exception as e:   # not importable at all
        return "not importable: %r" % (e,)


_TDANALYSIS_NEEDED = "tdanalysis 1.1 (the final release)"

try:
    import tdanalysis.forms.toeplitz_solver as _toeplitz_solver
    import tdanalysis.forms.sliding as _sliding
    import tdanalysis.forms.gs as _gs
    import tdanalysis.covariance.psd as _psd
    import tdanalysis.utils.filters as _filters
except ImportError as _exc:
    raise ImportError(
        "the td-gs code path (RIFT.likelihood.td_gs_precompute) needs %s ("
        "tdanalysis.forms.toeplitz_solver, tdanalysis.forms.sliding, tdanalysis.utils.filters). "
        "The tdanalysis on this path (%s) does not have them: %s"
        % (_TDANALYSIS_NEEDED, _tdanalysis_location(), _exc))


def _missing_tdanalysis_api():
    """Names of the tdanalysis 1.1 API this module uses that the tdanalysis on the path lacks.

    Earlier tdanalysis 1.1 development trees carry the same version string with other
    names (ToeplitzSolver.apply, turn_on_inside_window, want_V, ...), so every name is
    checked here, before anything is imported from them."""
    import dataclasses
    import inspect
    wanted = [
        (_toeplitz_solver, ("ToeplitzSolver",)),
        (_sliding, ("sliding_data_term", "sliding_self_terms", "is_turn_on_inside_window",
                    "SLIDING_METHODS", "DEFAULT_BLOCK_BYTES", "PrefixNotApplicable")),
        (_gs, ("gs_generators_from_acorr",)),
        (_psd, ("PSDToAutocorr", "PSDToAutocorrConfig")),
        (_filters, ("highpass_zero_phase", "fir_taps_from_response", "fir_truncate",
                    "fir_max_lag", "fir_truncate_at_max_lag")),
    ]
    missing = ["%s.%s" % (mod.__name__, name) for mod, names in wanted for name in names
               if not hasattr(mod, name)]
    solver = getattr(_toeplitz_solver, "ToeplitzSolver", None)
    if solver is not None:
        missing += ["tdanalysis.forms.toeplitz_solver.ToeplitzSolver.%s" % name
                    for name in ("apply_Cinv", "whiten") if not hasattr(solver, name)]
    if hasattr(_sliding, "sliding_self_terms"):
        params = inspect.signature(_sliding.sliding_self_terms).parameters
        missing += ["the %s= argument of tdanalysis.forms.sliding.sliding_self_terms" % name
                    for name in ("offset", "compute_V", "support") if name not in params]
    config = getattr(_psd, "PSDToAutocorrConfig", None)
    if config is not None and dataclasses.is_dataclass(config) and \
            "sampling_rate" not in {f.name for f in dataclasses.fields(config)}:
        missing.append("the sampling_rate field of tdanalysis.covariance.psd.PSDToAutocorrConfig")
    return missing


_missing = _missing_tdanalysis_api()
if _missing:
    raise ImportError(
        "the td-gs code path (RIFT.likelihood.td_gs_precompute) needs %s. The tdanalysis "
        "on this path (%s) lacks %s; earlier 1.1 development trees carry the same version "
        "string with other names. Install tdanalysis 1.1 (pip install 'tdanalysis>=1.1,<2') "
        "or update the checkout."
        % (_TDANALYSIS_NEEDED, _tdanalysis_location(), ", ".join(_missing)))

from tdanalysis.forms.toeplitz_solver import ToeplitzSolver  # noqa: E402
from tdanalysis.forms.sliding import (sliding_data_term, sliding_self_terms,  # noqa: E402
                                      is_turn_on_inside_window, SLIDING_METHODS,
                                      DEFAULT_BLOCK_BYTES, PrefixNotApplicable)
from tdanalysis.utils.filters import highpass_zero_phase  # noqa: E402,F401  (ILE --td-highpass)
from tdanalysis.utils.filters import (fir_taps_from_response, fir_truncate,  # noqa: E402
                                      fir_max_lag, fir_truncate_at_max_lag)

DEFAULT_ACF_CONDITION_NUMBER = 1.0e8   # as the tdanalysis real-event / XG ACF files
U_METHODS = tuple(SLIDING_METHODS)     # ("auto", "prefix", "recursion", "batched", "chunked")
DEFAULT_UV_BLOCK_BYTES = DEFAULT_BLOCK_BYTES   # memory bound of one column block of "chunked"
CALIBRATION_CORRECTION_TYPES = ("rift", "bilby")


# --------------------------------------------------------------------------- helpers

def _dimensionless_unit():
    try:
        return lal.DimensionlessUnit
    except AttributeError:
        return lsu.lsu_DimensionlessUnit


def load_acf_file(path, deltaT, rel_tol=1e-6):
    """Load a tdanalysis autocorrelation file and check its sampling.

    Format: float64 array of shape (2, N) with rows [tau, R(tau)], tau[k] = k/fs (as
    tdanalysis writes them).  The row holding R is chosen by the same rule
    as tdanalysis' loaders: ``arr[1] if abs(arr[1,0]) >= abs(arr[0,0]) else arr[0]``
    (tau[0] = 0, R(0) > 0).  A bare 1-D array is taken as R(tau) at spacing ``deltaT``.

    Raises ValueError if the lag spacing differs from ``deltaT`` (relative tolerance
    ``rel_tol``; the ILE analysis rate must match the ACF's, e.g. --srate 4096 for the
    4096 Hz real-event files) or the array is malformed.  Returns R(tau) as float64.
    """
    arr = np.load(path)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        acf = arr
    elif arr.ndim == 2 and arr.shape[0] == 2:
        if abs(arr[1, 0]) >= abs(arr[0, 0]):
            tau, acf = arr[0], arr[1]
        else:
            tau, acf = arr[1], arr[0]
        if tau.size < 2:
            raise ValueError("ACF file %s: fewer than two lags" % path)
        dtau = np.diff(tau)
        if not np.allclose(dtau, dtau[0], rtol=rel_tol, atol=0):
            raise ValueError("ACF file %s: lag grid is not uniform" % path)
        if abs(dtau[0] - deltaT) > rel_tol * deltaT:
            raise ValueError(
                "ACF file %s: lag spacing %.9g s (ACF sample rate %.6g Hz) does not match "
                "the analysis deltaT %.9g s (%.6g Hz). Run ILE at the ACF's rate, e.g. "
                "--srate %d (ILE defaults to 16384), or supply an ACF at %.6g Hz."
                % (path, dtau[0], 1. / dtau[0], deltaT, 1. / deltaT, int(round(1. / dtau[0])),
                   1. / deltaT))
    else:
        raise ValueError("ACF file %s: expected shape (2, N) [tau, R] or (N,), got %s"
                         % (path, arr.shape))
    if not np.all(np.isfinite(acf)):
        raise ValueError("ACF file %s: non-finite entries" % path)
    if acf[0] <= 0:
        raise ValueError("ACF file %s: R(0) = %g is not positive" % (path, acf[0]))
    return np.array(acf, dtype=np.float64)


def acf_from_psd(psd, deltaF, fs, n_lags, fLow, fMax, condition_number=DEFAULT_ACF_CONDITION_NUMBER):
    """ACF lags 0..n_lags-1 from a one-sided PSD with tdanalysis' PSDToAutocorr.

    Same construction as ToeplitzInnerProductContext.gs_acf (in-band PSD samples, band
    patching: constant wall outside [fLow, fMax], re-gridded onto an FFT long enough that
    the kept lags do not wrap), with the condition number passed explicitly.  Zero /
    non-finite PSD bins (e.g. below a per-detector --fmin-ifo cut) are left out of the
    band.  Returns (acf, info) where info is PSDToAutocorr.patch_info (wall value,
    n_floored = in-band bins clipped up to S_max/cond, ...).
    """
    from tdanalysis.covariance.psd import PSDToAutocorr, PSDToAutocorrConfig
    if isinstance(psd, lal.REAL8FrequencySeries):
        S = np.asarray(psd.data.data, dtype=np.float64)
    else:
        S = np.asarray(psd, dtype=np.float64)
    f = np.arange(S.size) * float(deltaF)
    band = np.isfinite(S) & (S > 0) & (f >= fLow) & (f < fMax)
    if not band.any():
        raise ValueError("acf_from_psd: no positive finite PSD bins in [%g, %g) Hz" % (fLow, fMax))
    fs_int = int(round(fs))
    duration = (n_lags - 1) / float(fs)
    conv = PSDToAutocorr(PSDToAutocorrConfig(
        sampling_rate=fs_int, acorr_duration=duration, f_min=float(fLow), f_max=float(fMax),
        patch_band=True, run_diagnostics=False,
        target_condition_number=float(condition_number)))
    _, acf = conv._psd_to_autocorr(f[band], S[band], fs_int, duration)
    info = dict(getattr(conv, "patch_info", {}) or {})
    info["condition_number"] = float(condition_number)
    return np.asarray(acf[:n_lags], dtype=np.float64), info


def import_fractional_shift():
    """tdanalysis' Lanczos fractional-shift resampler, (fractional_shift, SHIFT_STENCIL).

    Imported, not copied: the td-gs sub-sample placement must move templates with the
    same kernel as GWTransientTD.  It is tdanalysis.utils.subsample_align (tdanalysis >= 1.1).
    """
    try:
        from tdanalysis.utils.subsample_align import fractional_shift, SHIFT_STENCIL
    except ImportError as e:
        raise ImportError(
            "td-gs --td-subsample needs tdanalysis >= 1.1 (tdanalysis.utils.subsample_align."
            "fractional_shift, SHIFT_STENCIL). The tdanalysis on this path (%s) does not have "
            "it: %s" % (_tdanalysis_location(), e))
    return fractional_shift, int(SHIFT_STENCIL)


def subsample_phases(K):
    """Fractional shift s_j and extra whole-sample offset e_j of each phase j = 0..K-1 of a
    1/K grid: phase j is the template moved right by s_j samples with the Lanczos kernel
    and placed e_j samples later, s_j + e_j = j/K, with s_j in (-1/2, 1/2] (the residual
    range GWTransientTD._get_peak_and_target_indices uses; |s| <= 1/2 also halves the
    resampler's worst-case error).  Returns a list of (s_j, e_j)."""
    K = int(K)
    if K < 1:
        raise ValueError("subsample must be a positive integer, got %r" % (K,))
    out = []
    for j in range(K):
        if 2 * j <= K:
            out.append((j / float(K), 0))
        else:
            out.append((j / float(K) - 1.0, 1))
    return out


def shifted_modes(harr, s, fractional_shift=None):
    """{mode: template moved right by s samples} with tdanalysis' fractional_shift (the
    output has the input's length, as in GWTransientTD; s = 0 returns the inputs)."""
    if fractional_shift is None:
        fractional_shift = import_fractional_shift()[0]
    if s == 0.0:
        return {m: np.asarray(harr[m], dtype=np.complex128) for m in harr}
    return {m: np.asarray(fractional_shift(np.asarray(harr[m], dtype=np.complex128), float(s)),
                          dtype=np.complex128) for m in harr}


def parse_td_pieces(piece_strings, mref_msun):
    """Convert repeatable --td-piece 'START,END' (units of M_ref) to seconds.

    Returns a list of (start_s, end_s) relative to the reference time, sorted, with each
    end > start.  Overlaps are refused (pieces must be disjoint to be independent blocks).
    """
    if not piece_strings:
        return []
    if mref_msun is None or mref_msun <= 0:
        raise ValueError("--td-piece needs --td-mref MSUN (> 0) to convert M to seconds")
    msec = float(mref_msun) * lal.MTSUN_SI
    pieces = []
    for s in piece_strings:
        parts = [p for p in str(s).replace(";", ",").split(",") if p.strip() != ""]
        if len(parts) != 2:
            raise ValueError("--td-piece %r: expected 'START,END' in units of M_ref" % s)
        a, b = float(parts[0]) * msec, float(parts[1]) * msec
        if not b > a:
            raise ValueError("--td-piece %r: END must exceed START" % s)
        pieces.append((a, b))
    pieces.sort()
    for (a0, b0), (a1, b1) in zip(pieces[:-1], pieces[1:]):
        if a1 < b0:
            raise ValueError("--td-piece intervals overlap: [%g,%g] and [%g,%g] s"
                             % (a0, b0, a1, b1))
    return pieces


def template_route_is_fd(approx):
    """True if lalsimutils.hlmoft would build these modes from a frequency-domain
    generator (ChooseFDModes, SimInspiralTD-based IMRPhenomP reconstruction, or
    TDModesFromPolarizations of an FD-implemented approximant).  Those carry lalsuite's
    own conditioning that cannot be switched off, so the td-gs code path refuses them.
    Mirrors the branch order of hlmoft."""
    import lalsimulation as lalsim
    fd_modes = [getattr(lsu, n, None) for n in
                ("lalIMRPhenomHM", "lalIMRPhenomXHM", "lalIMRPhenomXPHM", "lalIMRPhenomXO4a",
                 "lalSEOBNRv4HM_ROM", "lalIMRPhenomXPNR")]
    if approx in [a for a in fd_modes if a is not None and a >= 0]:
        return True
    seob = [getattr(lsu, n, None) for n in ("lalSEOBv4", "lalTEOBv2", "lalTEOBv4", "lalSEOBNRv4HM")]
    seob += [getattr(lalsim, n, None) for n in
             ("SEOBNRv2", "SEOBNRv1", "SEOBNRv4_opt", "EOBNRv2", "SEOBNRv3", "SEOBNRv3_opt")]
    if approx in [a for a in seob if a is not None and a >= 0]:
        return False
    name = approx if isinstance(approx, str) else lalsim.GetStringFromApproximant(approx)
    if 'IMRPhenomP' in name or approx == getattr(lsu, "lalIMRPhenomXP", None) \
            or lsu.check_FD_pending(approx):
        return True
    if lalsim.SimInspiralImplementedFDApproximants(approx) == 1:
        return True
    return False


# --------------------------------------------------------------------------- calibration filters

def calibration_template_factor(cal2side, ifo, correction_type="rift", conjugate=False):
    """The per-bin factor C_c(f) that multiplies the TEMPLATE (RIFT packing, same shape as the
    input) from RIFT realizations eta_c(f) (create_realizations output, (npts, n_cal)).

    correction_type "rift"  : C = eta for every detector.  This reproduces RIFT's FD code path
                              (eta applied to the data in RIFT's reversed packing equals, in
                              the standard convention, eta on the template at face value).
                     "bilby" : bilby_pipe's per-detector semantics
                              (RIFT.calmarg.calibration.correction_type_for_ifo): C = eta for
                              V1 ("template" type), C = 1/eta for H1, L1, K1 ("data" type).
    conjugate : RIFT's --calibration-conjugate-phase, C -> conj(C).
    Returns (C, bilby_type) with bilby_type the "template"/"data" label used.
    """
    from RIFT.likelihood.td_calibration import template_factor
    if correction_type == "rift":
        ctype = "template"
    elif correction_type == "bilby":
        from RIFT.calmarg.calibration import correction_type_for_ifo
        ctype = correction_type_for_ifo(None, ifo)
    else:
        raise ValueError("correction_type must be one of %s, got %r"
                         % (CALIBRATION_CORRECTION_TYPES, correction_type))
    return template_factor(cal2side, ctype, conjugate), ctype


class TDCalibrationFilters(object):
    r"""One detector's calibration realizations as FIR filters on the template, plus the
    adjoint-filtered data weights, built once per realization set (TDGSDetectorContext.
    calibration_filters caches it).  The filters are tdanalysis' (tdanalysis.utils.filters:
    fir_taps_from_response, fir_truncate, fir_max_lag, fir_truncate_at_max_lag); this
    class holds RIFT's conventions (packing, correction type, conjugate phase) and the data
    weights.

    Attributes
    ----------
    n_cal, N           : realizations; segment length (= taps per filter)
    taps_full          : (n_cal, N) real, periodic index (tap i at column i mod N)
    W                  : (n_cal, 2N-1) real, W[c, j + N/2] = (K_c^T w)[j] with w the piecewise
                         C^{-1} d, zero outside the pieces, K_c the N-tap filter: Q_c(t) =
                         h_t^H K_c^T w is a correlation of the unfiltered template with W[c].
    half_length        : None (exact: L from the template's reach per intrinsic point) or a
                         forced symmetric half-length (an approximation when shorter than
                         the reach).
    correction_type, bilby_type, conjugate : the convention used (module header).
    """

    def __init__(self, ctx, cal2side, correction_type="rift", conjugate=False, half_length=None,
                 verbose=True):
        from RIFT.likelihood.td_calibration import realizations_to_positive, is_hermitian_rift
        cal = np.asarray(cal2side)
        if cal.ndim != 2 or cal.shape[0] != ctx.N:
            raise ValueError(
                "%s: calibration realizations have shape %s; expected (%d, n_cal) -- RIFT's two-sided "
                "packing on the data segment's grid (T = N deltaT = %g s)"
                % (ctx.ifo, cal.shape, ctx.N, ctx.N * ctx.deltaT))
        self.n_cal = int(cal.shape[1])
        self.N = int(ctx.N)
        self.deltaT = ctx.deltaT
        self.correction_type = correction_type
        self.conjugate = bool(conjugate)
        C2, self.bilby_type = calibration_template_factor(cal, ctx.ifo, correction_type, conjugate)
        if not is_hermitian_rift(C2):
            raise ValueError("%s: calibration realizations are not Hermitian (C(-f) != conj C(f)); "
                             "no real filter exists" % ctx.ifo)
        C_pos = realizations_to_positive(C2)                               # (N/2+1, n_cal)
        # the N taps whose response on the realization's grid is C_c exactly
        self.taps_full = np.ascontiguousarray(fir_taps_from_response(C_pos).T)   # (n_cal, N)
        half = self.N // 2
        if half_length is not None:
            half_length = int(half_length)
            if half_length < 0 or half_length > half - 1:
                raise ValueError("%s: calibration FIR half-length %d must lie in [0, %d]" % (ctx.ifo, half_length, half - 1))
        self.half_length = half_length
        # adjoint-filtered data weights with the N-tap filter: W_c[j + N/2] = sum_m c_c[m - j] w[m]
        # (exact linear correlation: fftconvolve of the zero-extended w with the reversed taps)
        w_full = np.zeros(self.N, dtype=np.float64)
        for (i0, i1), w in zip(ctx.runs, ctx.w):
            w_full[i0:i1] = w
        taps_all, _, self._W_offset = fir_truncate(self.taps_full.T)               # (N, n_cal)
        self.W = np.array([scipy.signal.fftconvolve(w_full, taps_all[::-1, c]) for c in range(self.n_cal)])
        # magnitude of the taps far from the centre (how fast the realization's impulse response
        # decays): |c[i]| for |i| = 1, 2, 4, ... relative to c[0]
        if verbose:
            idx = [2 ** k for k in range(0, int(np.log2(half)) + 1) if 2 ** k <= half]
            tail = ["%d:%.1e" % (i, np.max(np.abs(self.taps_full[:, i])) / np.max(np.abs(self.taps_full[:, 0]))) for i in idx[::3]]
            print("  [td-gs] %s: calibration: %d realizations as N-tap FIR filters on the template "
                  "(N = %d, %.4g s); convention %s (bilby type '%s'%s); %s; max |c[i]|/|c[0]| at i = %s"
                  % (ctx.ifo, self.n_cal, self.N, self.N * ctx.deltaT, correction_type, self.bilby_type,
                     ", conjugated" if conjugate else "",
                     "L from the template's reach per intrinsic point (exact)" if half_length is None
                     else "FORCED half-length L = %d (an approximation when shorter than the reach)" % half_length,
                     " ".join(tail)))

    def taps_for(self, R):
        """(taps (n_cal, n_taps), L_neg, L_pos) exact on the pieces for reach R
        (tdanalysis.utils.filters.fir_max_lag): tdanalysis' fir_truncate_at_max_lag, the
        symmetric truncation at L = R when R < N/2 (identical values on the pieces as the N-tap
        filter), else all N taps; a forced half_length overrides (approximate when < R)."""
        if self.half_length is not None:
            taps, L_neg, L_pos = fir_truncate(self.taps_full.T, self.half_length)
        else:
            taps, L_neg, L_pos = fir_truncate_at_max_lag(self.taps_full.T, R)
        return np.ascontiguousarray(taps.T), L_neg, L_pos

    def filtered_template(self, h, taps_c, L_neg):
        """g_c = K_c h on the whole template line as a 'full' linear convolution: sample j' of
        the result is template index j' - L_neg (the template origin moves by L_neg)."""
        return scipy.signal.fftconvolve(h, taps_c)

    def data_term(self, h, c, j_first, j_last, N_shift, T):
        """Q_c(t) = sum_j conj(h[j]) W_c[j + N_shift + t + N/2], t = 0..T-1: the exact linear
        correlation of the template's support with the matching window of W[c]."""
        h_sup = h[j_first:j_last + 1]
        a = N_shift + self._W_offset + j_first
        b = N_shift + T - 1 + self._W_offset + j_last + 1
        win = np.zeros(b - a, dtype=np.float64)
        lo, hi = max(a, 0), min(b, self.W.shape[1])
        if hi > lo:
            win[lo - a:hi - a] = self.W[c, lo:hi]
        out = scipy.signal.fftconvolve(win, np.conj(h_sup)[::-1], mode="valid")
        return out


# --------------------------------------------------------------------------- per-detector context

class TDGSDetectorContext(object):
    r"""Per-detector state for the td-gs Precompute (built once per ILE job).

    Parameters
    ----------
    data_td : lal.REAL8TimeSeries (or COMPLEX16TimeSeries)
        The detector strain, unwindowed.  Samples at index >= ``n_valid`` are zero
        padding (ILE pads to a power of two) and may not be part of any piece.
    psd : lal.REAL8FrequencySeries or ndarray
        One-sided PSD on the rfft grid of the full series (deltaF = 1/(N deltaT)); used
        only to build the ACF when ``acf`` is None (tdanalysis PSDToAutocorr, walled
        outside [fLow, fMax)).
    fLow, fMax : float
        Analysis band [Hz] for the PSD->ACF conversion.
    pieces : list of (t0_gps, t1_gps) or None
        Analysis pieces as GPS seconds, fixed for the whole run.  None = the full valid
        data segment (one piece).
    acf : ndarray or None
        R(tau) at lags 0, deltaT, 2 deltaT, ... (e.g. from load_acf_file).  Must cover
        the longest piece.
    n_valid : int or None
        Number of real (unpadded) samples; default = full length.
    dense_max_n : int
        Pieces with at most this many samples use the Cholesky method of tdanalysis'
        ToeplitzSolver; longer ones its Gohberg-Semencul method.  Both are exact.
    u_stride : int
        Compute U(t), V(t) every ``u_stride`` samples of the storage grid and fill the rest
        by cubic interpolation (1 = exact at every sample).
    u_method : {"auto", "prefix", "recursion", "batched", "chunked"}
        How U(t), V(t) are evaluated (tdanalysis.forms.sliding.sliding_self_terms, see the
        module header).  All methods are exact.
        "prefix" (windowed sums of one correlation per mode and kernel) needs the
        shifted template to end inside -- or, mirrored, to start inside -- every piece
        for every grid shift and raises otherwise; "recursion" is exact for every cut
        pattern with no per-shift solve; "batched" is the direct reference (stacked cut
        templates, C^{-1} applied once; memory n x N_window per mode); "chunked" is
        "batched" in column blocks of at most ``uv_block_bytes``.  "auto" (default) uses
        "chunked" on Cholesky pieces, and on GS pieces "prefix" where its condition holds,
        "recursion" elsewhere.  The choice made for each piece is in ``self.uv_routes``
        after precompute, as (method, orientation or None).
    uv_block_bytes : int
        Memory bound of one column block (stacked templates of all modes, C^{-1} applied
        to one mode, GS work arrays) by the "chunked" method.
    acf_condition_number : float
        target_condition_number for the PSD->ACF wall (ignored when ``acf`` is given).
    whiteness_check : bool
        Also whiten every piece, z = L^{-1} d_p with C_p = L L^T, by the piece solver's
        ``whiten`` (its Cholesky factor on Cholesky pieces; on GS pieces a Levinson-Durbin
        recursion, O(n_p^2) once per job with nothing n_p x n_p formed, whose rounding error
        at cond 1e8 is ~4e-7 relative, far below what the diagnostic resolves), and report
        the variance and lag-1 autocorrelation of z.  The cheap d_p^T C_p^{-1} d_p / n_p is
        always computed (``self.whiteness``).
    calibration_correction_type : {"rift", "bilby"}
        How a calibration realization eta_c becomes the template factor C_c (module header):
        "rift" (default) C_c = eta_c, RIFT's FD convention; "bilby" per-detector bilby_pipe
        semantics (1/eta for H1/L1/K1, eta for V1).
    calibration_fir_half_length : int or None
        None (default): the calibration filters are exact (module header: per intrinsic
        point the taps are truncated at the template's reach, or all N are used).  An
        integer forces a symmetric half-length L, an approximation when shorter than the
        reach (for speed studies; measured errors in docs/td_gs_route.md).
    subsample : int
        K >= 1.  With K > 1, ``precompute`` returns Q(t), U(t), V(t) on a grid of spacing
        deltaT/K (module header, "Sub-sample placement"): K - 1 extra template sets moved
        by a fraction of a sample with tdanalysis' Lanczos resampler, each through the
        same exact methods (calibration included: each phase's shifted template is filtered
        and its realization blocks carry the fine grid).  Costs K times the K = 1
        Precompute.  Needs ``tdanalysis.utils.subsample_align`` (tdanalysis >= 1.1; checked
        here, at construction).
    """

    def __init__(self, data_td, psd, fLow, fMax, pieces=None, acf=None, n_valid=None,
                 dense_max_n=4096, u_stride=1, u_method="auto",
                 acf_condition_number=DEFAULT_ACF_CONDITION_NUMBER, whiteness_check=False,
                 ifo=None, verbose=True, uv_block_bytes=DEFAULT_UV_BLOCK_BYTES,
                 calibration_correction_type="rift", calibration_fir_half_length=None, subsample=1):
        self.ifo = ifo
        self.subsample = int(subsample)
        if self.subsample < 1:
            raise ValueError("subsample must be a positive integer, got %r" % (subsample,))
        self._fractional_shift = None
        self.shift_stencil = 0
        if self.subsample > 1:
            self._fractional_shift, self.shift_stencil = import_fractional_shift()
        self.deltaT = float(data_td.deltaT)
        self.epoch = data_td.epoch
        self.N = int(data_td.data.length)
        if calibration_correction_type not in CALIBRATION_CORRECTION_TYPES:
            raise ValueError("calibration_correction_type must be one of %s, got %r"
                             % (CALIBRATION_CORRECTION_TYPES, calibration_correction_type))
        self.calibration_correction_type = calibration_correction_type
        self.calibration_fir_half_length = calibration_fir_half_length
        self._calibration_cache = None
        self.uv_routes_cal = []           # per piece, the method of the last calibrated precompute (phase 0)
        self.uv_routes_cal_by_phase = []  # the same per sub-sample phase (one entry when K = 1)
        self.calibration_last = {}      # diagnostics of the last calibrated precompute
        self.n_valid = self.N if n_valid is None else int(n_valid)
        self.d = np.asarray(data_td.data.data)
        if np.iscomplexobj(self.d):
            if np.max(np.abs(self.d.imag)) > 0:
                raise ValueError("td-gs data must be real strain")
            self.d = self.d.real
        self.d = np.array(self.d, dtype=np.float64)
        self.u_stride = max(1, int(u_stride))
        if u_method not in U_METHODS:
            raise ValueError("u_method must be one of %s" % (U_METHODS,))
        self.u_method = u_method
        self.uv_block_bytes = int(uv_block_bytes)
        self.uv_routes = []           # per piece, the U/V method precompute used last (phase 0)
        self.uv_routes_by_phase = []  # the same per sub-sample phase (one entry when K = 1)
        self.dense_max_n = int(dense_max_n)
        self.fLow, self.fMax = float(fLow), float(fMax)
        self._warned_support = False

        # ---- ACF: supplied, or from the PSD (tdanalysis PSDToAutocorr) ----
        self.acf_supplied = acf is not None
        self.acf_info = {}
        if self.acf_supplied:
            self.acf = np.asarray(acf, dtype=np.float64)
        else:
            self.acf, self.acf_info = acf_from_psd(
                psd, 1.0 / (self.N * self.deltaT), 1.0 / self.deltaT, self.N,
                self.fLow, self.fMax, condition_number=acf_condition_number)

        # ---- pieces -> sample ranges [i0, i1) in the data frame ----
        self.explicit_pieces = pieces is not None and len(pieces) > 0
        if self.explicit_pieces:
            runs = []
            for (t0, t1) in pieces:
                i0 = int(round((float(t0) - float(self.epoch)) / self.deltaT))
                i1 = int(round((float(t1) - float(self.epoch)) / self.deltaT))
                if i0 < 0 or i1 > self.n_valid or i1 <= i0:
                    raise ValueError(
                        "%s: piece [%.6f, %.6f] GPS -> samples [%d, %d) does not lie inside "
                        "the valid data [0, %d) (data start %.6f, %.6g s long)"
                        % (ifo, t0, t1, i0, i1, self.n_valid, float(self.epoch),
                           self.n_valid * self.deltaT))
                runs.append((i0, i1))
            runs.sort()
            for (a0, b0), (a1, b1) in zip(runs[:-1], runs[1:]):
                if a1 < b0:
                    raise ValueError("%s: pieces overlap in samples: %s" % (ifo, runs))
            self.runs = runs
        else:
            self.runs = [(0, self.n_valid)]

        # ---- per piece: tdanalysis ToeplitzSolver (validates the covariance), w_p =
        #      C_p^{-1} d_p (once per job), whiteness ----
        self.solvers = []
        self.w = []
        self.whiteness = []
        for (i0, i1) in self.runs:
            n = i1 - i0
            if self.acf.size < n:
                raise ValueError("%s: ACF too short for piece [%d, %d): %d lags supplied, %d needed"
                                 % (ifo, i0, i1, self.acf.size, n))
            sol = ToeplitzSolver(self.acf[:n], dense_max_n=self.dense_max_n)
            self.solvers.append(sol)
            w = sol.apply_Cinv(self.d[i0:i1])
            self.w.append(w)
            chi2_per_n = float(self.d[i0:i1] @ w) / n
            entry = dict(i0=int(i0), i1=int(i1), n=int(n), chi2_per_n=chi2_per_n,
                         white_var=None, white_lag1=None)
            if whiteness_check:
                z = sol.whiten(self.d[i0:i1])
                entry["white_var"] = float(np.mean(z * z))
                entry["white_lag1"] = float(np.mean(z[1:] * z[:-1]) / np.mean(z * z))
            self.whiteness.append(entry)
        if verbose:
            self.report()

    def acf_lags(self, n):
        """ACF lags 0..n-1 (the covariance of an n-sample piece is Toeplitz of this)."""
        if self.acf.size < n:
            raise ValueError("ACF has %d lags; %d needed" % (self.acf.size, n))
        return self.acf[:n]

    # ------------------------------------------------------------------ calibration

    def calibration_filters(self, cal2side, conjugate=False, verbose=True):
        """TDCalibrationFilters for this detector's realization set (RIFT packing, (N, n_cal)),
        with the context's convention and FIR settings; cached on the array's identity, shape
        and a content digest so the per-point Precompute does not rebuild it."""
        import hashlib
        cal = np.asarray(cal2side)
        digest = hashlib.blake2b(np.ascontiguousarray(cal[::max(1, cal.shape[0] // 64)]).tobytes(),
                                 digest_size=16).digest()
        key = (id(cal2side), tuple(cal.shape), str(cal.dtype), bool(conjugate), digest)
        if self._calibration_cache is not None and self._calibration_cache[0] == key:
            return self._calibration_cache[1]
        filters = TDCalibrationFilters(self, cal, correction_type=self.calibration_correction_type,
                                       conjugate=conjugate, half_length=self.calibration_fir_half_length,
                                       verbose=verbose)
        self._calibration_cache = (key, filters)
        return filters

    def report(self):
        """Print the one-line summary and the per-piece whiteness diagnostic."""
        print("  [td-gs] %s: %d piece(s) %s samples, solver(s) %s, ACF %s, U/V by %s%s"
              % (self.ifo, len(self.runs), [(int(a), int(b)) for (a, b) in self.runs],
                 [s.method for s in self.solvers],
                 "file (%d lags)" % self.acf.size if self.acf_supplied
                 else "PSD->tdanalysis (cond %.1e)" % self.acf_info.get("condition_number", np.nan),
                 self.u_method,
                 "" if self.subsample == 1 else
                 ", sub-sample placement 1/%d (tdanalysis Lanczos shift, half-width %d; "
                 "Precompute cost x%d)" % (self.subsample, self.shift_stencil, self.subsample)))
        if not self.acf_supplied:
            n_floored = int(self.acf_info.get("n_floored", 0) or 0)
            keys = [k for k in ("n_floored", "n_walled", "wall", "s_floor", "s_lo", "s_hi", "nfft", "df")
                    if k in self.acf_info]
            print("  [td-gs] %s: PSD->ACF patch info: %s"
                  % (self.ifo, ", ".join("%s=%s" % (k, ("%.4g" % self.acf_info[k]) if isinstance(
                      self.acf_info[k], float) else self.acf_info[k]) for k in keys)))
            if n_floored > 0:
                print("  [td-gs] %s: WARNING: %d in-band PSD bins were FLOORED at S_max/cond by the "
                      "ACF conditioning; the TD noise model differs from the PSD there. Raise "
                      "--td-acf-cond or narrow the band." % (self.ifo, n_floored))
        for e in self.whiteness:
            flag = "" if 0.7 <= e["chi2_per_n"] <= 1.4 else "   <-- WARNING: far from 1"
            extra = ""
            if e["white_var"] is not None:
                extra = ", whitened var %.3f, lag-1 corr %.3f" % (e["white_var"], e["white_lag1"])
            print("  [td-gs] %s: piece [%d, %d) d^T C^-1 d / n = %.3f%s%s"
                  % (self.ifo, e["i0"], e["i1"], e["chi2_per_n"], extra, flag))

    # ------------------------------------------------------------------ template placement

    @staticmethod
    def _stack_modes(harr, modes):
        """The template modes as the rows of one (n_modes, M) complex array, zero-extended
        to the longest mode (the layout tdanalysis' sliding forms take)."""
        M = max(harr[m].size for m in modes)
        H = np.zeros((len(modes), M), dtype=np.complex128)
        for ia, m in enumerate(modes):
            H[ia, :harr[m].size] = harr[m]
        return H

    @staticmethod
    def _template_support(H):
        """(first, last) non-zero sample over the rows of H; (0, -1) when all are zero."""
        nz = np.flatnonzero(np.any(np.abs(H) > 0, axis=0))
        if nz.size == 0:
            return 0, -1
        return int(nz[0]), int(nz[-1])

    def _check_template_support(self, H, j_first, N_shift, N_window):
        """Refuse (explicit pieces) or warn (implicit full segment) when the template's
        first non-zero sample would fall inside a piece for some shift on the storage grid.

        With no taper the template switches on abruptly at its first sample; a piece that
        extends before that turn-on contains a step the data does not have, which biases
        real-data fits.  Shift k puts the turn-on at data sample k + j_first; the grid is
        k in [N_shift, N_shift + N_window - 1] (tdanalysis.forms.sliding.
        is_turn_on_inside_window on each piece).  (With calibration the same check applies
        to the unfiltered template; the FIR then smears the turn-on step into the piece
        with the taps' magnitude at the turn-on distance, which precompute reports as the
        turn-on leakage -- the FD code path's circulant multiplication smears it the same way.)
        """
        shifts = N_shift + np.arange(N_window)
        lo = N_shift + j_first
        hi = N_shift + N_window - 1 + j_first
        for (i0, i1) in self.runs:
            # some shift puts the turn-on strictly after the piece start and before its end
            if is_turn_on_inside_window(H, shifts, i1 - i0, offset=i0):
                msg = ("%s: the template turn-on (first non-zero sample) lands inside piece "
                       "[%d, %d) for some storage-grid shifts (turn-on at data samples %d..%d, "
                       "i.e. %.3f..%.3f s after the piece start). The template must already be "
                       "on at the piece start for EVERY shift of the +-storage window: lower "
                       "--fmin-template (longer template), start the piece later, or shrink "
                       "--internal-data-storage-window-half."
                       % (self.ifo, i0, i1, lo, hi, (lo - i0) * self.deltaT, (hi - i0) * self.deltaT))
                if self.explicit_pieces:
                    raise ValueError(msg)
                if not self._warned_support:
                    print("  [td-gs] WARNING (full segment, no --td-piece): " + msg)
                    self._warned_support = True

    # ------------------------------------------------------------------ Q, U, V over the pieces

    def _q_all(self, H, N_shift, T):
        """Q(t) of every row of H at the shifts N_shift + t, t = 0..T-1, summed over the
        pieces: tdanalysis' sliding_data_term on each piece (offset = piece start) against
        w_p = C_p^{-1} d_p.  Returns (n_modes, T) complex."""
        shifts = N_shift + np.arange(T)
        Q = np.zeros((H.shape[0], T), dtype=np.complex128)
        for (i0, i1), w in zip(self.runs, self.w):
            Q += sliding_data_term(H, w, shifts, offset=i0)
        return Q

    @staticmethod
    def _method_label(label):
        """tdanalysis' method label ("prefix(end)", "recursion", ...) as RIFT's
        (method, orientation or None) pair."""
        if label.startswith("prefix(") and label.endswith(")"):
            return ("prefix", label[len("prefix("):-1])
        return (label, None)

    def _uv_all(self, H, N_shift, T, tsel, j_first, j_last, support=None):
        """U(t), V(t) of every pair of rows of H at the shifts N_shift + tsel, summed over
        the pieces: tdanalysis' sliding_self_terms on each piece (offset = piece start)
        with that piece's solver and the context's u_method.  Returns (U, V, methods):
        U, V of shape (len(tsel), n_modes, n_modes), Hermitian / symmetric in the modes,
        and per piece the (method, orientation or None) used.  u_method="prefix" on a
        piece where the prefix condition fails raises ValueError naming every such piece
        (template support j_first..j_last).  ``support`` = (first, last), if given, goes to
        sliding_self_terms: the template samples used, whatever H holds outside them."""
        nm = H.shape[0]
        shifts = N_shift + np.asarray(tsel)
        U = np.zeros((tsel.size, nm, nm), dtype=np.complex128)
        V = np.zeros((tsel.size, nm, nm), dtype=np.complex128)
        methods, bad = [], []
        for (i0, i1), sol in zip(self.runs, self.solvers):
            try:
                Up, Vp, label = sliding_self_terms(H, sol, shifts, method=self.u_method,
                                                   block_bytes=self.uv_block_bytes, offset=i0,
                                                   return_method=True, support=support)
            except PrefixNotApplicable:
                # only raised for u_method="prefix"; collect every failing piece
                bad.append((i0, i1))
                methods.append(None)
                continue
            U += Up
            V += Vp
            methods.append(self._method_label(label))
        if bad:
            raise ValueError(
                "%s: u_method='prefix' needs the shifted template to end inside (or "
                "start inside) the piece for every grid shift; pieces %s fail "
                "(template support %d..%d, N_shift %d, N_window %d). Use 'auto'."
                % (self.ifo, bad, j_first, j_last, N_shift, T))
        return U, V, methods

    def _complete_pairs(self, U, V, modes, tsel, T):
        """crossTerms, crossTermsV dicts {(mode_a, mode_b): length-T array} from the
        (len(tsel), n_modes, n_modes) U, V: the upper triangle, filled between the stride
        points by cubic interpolation, and its Hermitian (U) / symmetric (V) completion."""
        crossTerms, crossTermsV = {}, {}
        for ia, ma in enumerate(modes):
            for ib in range(ia, len(modes)):
                mb = modes[ib]
                u, v = self._expand(U[:, ia, ib], tsel, T), self._expand(V[:, ia, ib], tsel, T)
                crossTerms[(ma, mb)] = u
                crossTermsV[(ma, mb)] = v
                if mb != ma:
                    crossTerms[(mb, ma)] = np.conj(u)
                    crossTermsV[(mb, ma)] = v.copy()
        return crossTerms, crossTermsV

    def _rholm_series(self, Q, hlms, modes, N_shift, deltaT):
        """lal series of Q ({mode: array}) at spacing ``deltaT`` with the epoch bookkeeping
        of ComputeModeIPTimeSeries: rhoTS.epoch = data.epoch - hlm.epoch, then DataRollBins
        adds N_shift*deltaT."""
        rholms = {}
        for m in modes:
            ts = lal.CreateCOMPLEX16TimeSeries("Complex overlap", lal.LIGOTimeGPS(0.), 0.,
                                               deltaT, _dimensionless_unit(), Q[m].size)
            ts.epoch = self.epoch - hlms[m].epoch
            ts.epoch += N_shift * self.deltaT
            ts.data.data = Q[m]
            rholms[m] = ts
        return rholms

    def precompute(self, hlms, N_shift, N_window, verbose=False, calibration=None,
                   calibration_self_term=True):
        """Q_lm(t), U(t), V(t) on the storage grid for TD modes ``hlms`` ({(l,m): lal
        COMPLEX16TimeSeries}, all with the same epoch/deltaT).

        Returns (rholms, crossTerms, crossTermsV); see the module header for the contract.

        calibration : TDCalibrationFilters (from ``calibration_filters``) or None.  With
            filters, the return value is (rholms, crossTerms, crossTermsV, crossTermsCal,
            crossTermsCalV): rholms[mode] holds the n_cal realizations' Q_c(t) as contiguous
            blocks of length N_window (block 0's epoch, as ComputeModeIPTimeSeries does);
            crossTerms/crossTermsV are the UNCALIBRATED U(t), V(t) (one block); crossTermsCal
            and crossTermsCalV are lists of n_cal dicts with the per-realization U_c(t),
            V_c(t) of the filtered templates, or None when ``calibration_self_term`` is
            False (RIFT's --calibration-global-norm: the calibration-independent self term).

        With ``self.subsample`` = K > 1 every series has K entries per storage sample at
        spacing deltaT/K (rholms deltaT = deltaT/K): entry r of a block <-> shift
        N_shift + r/K, so the uncalibrated series are K*N_window long and each realization
        block of the calibrated ones as well (n_cal*K*N_window in total).
        """
        modes = list(hlms.keys())
        h0 = hlms[modes[0]]
        if abs(float(h0.deltaT) - self.deltaT) > 1e-12 * self.deltaT:
            raise ValueError("template deltaT %g != data deltaT %g" % (h0.deltaT, self.deltaT))
        N_shift, T = int(N_shift), int(N_window)
        harr = {m: np.asarray(hlms[m].data.data, dtype=np.complex128) for m in modes}
        K = self.subsample
        # Phase j = 0..K-1 (subsample_phases): the template moved by s_j of a sample with
        # tdanalysis' kernel and placed e_j samples later, each through the whole-sample
        # Precompute body (calibration included).  Phase 0 is the unshifted template, the
        # K = 1 result bit for bit.  Fine entry r = K*i + j <-> shift N_shift + i + j/K.
        phases = []
        for j, (s_j, e_j) in enumerate(subsample_phases(K)):
            harr_j = harr if j == 0 else shifted_modes(harr, s_j, self._fractional_shift)
            try:
                phases.append(self._precompute_phase(harr_j, modes, N_shift + e_j, T,
                                                     calibration, calibration_self_term))
            except ValueError as e:
                if j == 0:
                    raise
                raise ValueError(
                    "%s (template moved by %+.4f sample for sub-sample phase %d/%d: the "
                    "Lanczos shift kernel, half-width %d, spreads the template's first "
                    "sample over the %d samples before it, so the piece must also clear "
                    "that margin)" % (e, s_j, j, K, self.shift_stencil, self.shift_stencil - 1))
        ph0 = phases[0]
        self.uv_routes = ph0["uv_routes"]
        self.uv_routes_by_phase = [ph["uv_routes"] for ph in phases]
        dt_out = self.deltaT / K
        crossTerms = {k: self._interleave([ph["U"][k] for ph in phases]) for k in ph0["U"]}
        crossTermsV = {k: self._interleave([ph["V"][k] for ph in phases]) for k in ph0["V"]}
        if verbose:
            u00 = crossTerms[(modes[0], modes[0])]
            print("  [td-gs] %s: N_shift=%d N_window=%d template support %d..%d; U/V methods %s; "
                  "U(2,2) centre %.6g, |U| range %.3g..%.3g%s"
                  % (self.ifo, N_shift, T, ph0["j_first"], ph0["j_last"],
                     [r if o is None else "%s(%s)" % (r, o) for (r, o) in self.uv_routes],
                     np.real(u00[u00.size // 2]), np.min(np.abs(u00)), np.max(np.abs(u00)),
                     "" if K == 1 else "; sub-sample 1/%d: %d entries at deltaT/%d, methods per phase %s"
                     % (K, K * T, K, [[r for (r, o) in rp] for rp in self.uv_routes_by_phase])))
        if calibration is None:
            Q = {m: self._interleave([ph["Q"][m] for ph in phases]) for m in modes}
            rholms = self._rholm_series(Q, hlms, modes, N_shift, dt_out)
            return rholms, crossTerms, crossTermsV

        # ---- calibration: the per-phase realization blocks interleaved onto the fine grid ----
        n_cal = calibration.n_cal
        self.uv_routes_cal = ph0["uv_routes_cal"]
        self.uv_routes_cal_by_phase = [ph["uv_routes_cal"] for ph in phases]
        self.calibration_last = dict(ph0["calibration_last"])
        self.calibration_last["phases"] = [ph["calibration_last"] for ph in phases]
        Qcal = {m: self._interleave([ph["Qcal"][m] for ph in phases], n_cal) for m in modes}
        rholms_cal = self._rholm_series(Qcal, hlms, modes, N_shift, dt_out)
        crossTermsCal = crossTermsCalV = None
        if calibration_self_term:
            crossTermsCal = [{k: self._interleave([ph["Ucal"][c][k] for ph in phases]) for k in ph0["Ucal"][c]}
                             for c in range(n_cal)]
            crossTermsCalV = [{k: self._interleave([ph["Vcal"][c][k] for ph in phases]) for k in ph0["Vcal"][c]}
                              for c in range(n_cal)]
        if verbose:
            cl = ph0["calibration_last"]
            msg = ("  [td-gs] %s: calibration: %d realizations; taps reach %d, used %d..%d (%s); turn-on "
                   "leakage %.1e; Q_c(t) %.2f s" % (self.ifo, n_cal, cl["reach"], -cl["L_neg"], cl["L_pos"],
                                                     "exact" if cl["exact"] else "FORCED, approximate",
                                                     cl["turn_on_leakage"], sum(ph["t_q"] for ph in phases)))
            if calibration_self_term:
                u0 = crossTermsCal[0][(modes[0], modes[0])]
                u00 = crossTerms[(modes[0], modes[0])]
                msg += ("; U_c(t), V_c(t) %.2f s with methods %s; U_0(2,2) centre %.6g (uncalibrated %.6g)"
                        % (sum(ph["t_uv"] for ph in phases),
                           [r if o is None else "%s(%s)" % (r, o) for (r, o) in self.uv_routes_cal],
                           np.real(u0[u0.size // 2]), np.real(u00[u00.size // 2])))
            else:
                msg += "; self term: calibration-independent U(t), V(t)"
            if K > 1:
                msg += "; sub-sample 1/%d: %d entries per realization block, reach per phase %s" % (
                    K, K * T, [ph["calibration_last"]["reach"] for ph in phases])
            print(msg)
        return rholms_cal, crossTerms, crossTermsV, crossTermsCal, crossTermsCalV

    @staticmethod
    def _interleave(per_phase, n_blocks=1):
        """One fine-grid array from the K per-phase arrays, each made of ``n_blocks``
        contiguous blocks of T entries: out[b*K*T + K*i + j] = per_phase[j][b*T + i].
        K = 1 returns the array itself (no copy)."""
        K = len(per_phase)
        if K == 1:
            return per_phase[0]
        T = per_phase[0].size // n_blocks
        out = np.empty(n_blocks * K * T, dtype=np.complex128)
        view = out.reshape(n_blocks, T, K)
        for j, a in enumerate(per_phase):
            view[:, :, j] = np.asarray(a).reshape(n_blocks, T)
        return out

    def _precompute_phase(self, harr, modes, N_shift, T, calibration=None, calibration_self_term=True):
        """The whole-sample Precompute body for the template arrays ``harr`` placed at
        shifts N_shift + i, i = 0..T-1: Q, U, V (dicts of length-T arrays, U/V Hermitian/
        symmetric completed; Q only without calibration) and, with ``calibration``, the
        per-realization Q_c (dict of n_cal*T arrays, contiguous realization blocks) and U_c,
        V_c (lists of n_cal dicts, or None without the self term).  Sets self.uv_routes (and
        self.uv_routes_cal, self.calibration_last).  Returns a dict with keys U, V, j_first,
        j_last, uv_routes, and Q without calibration or Qcal, Ucal, Vcal, uv_routes_cal,
        calibration_last, t_q, t_uv with it."""
        H = self._stack_modes(harr, modes)
        j_first, j_last = self._template_support(H)
        self._check_template_support(H, j_first, N_shift, T)

        # U/V evaluation points on the grid (stride) -- always include both ends.
        tsel = np.arange(0, T, self.u_stride)
        if tsel[-1] != T - 1:
            tsel = np.append(tsel, T - 1)

        U, V, self.uv_routes = self._uv_all(H, N_shift, T, tsel, j_first, j_last)
        crossTerms, crossTermsV = self._complete_pairs(U, V, modes, tsel, T)
        out = dict(U=crossTerms, V=crossTermsV, j_first=j_first, j_last=j_last, uv_routes=self.uv_routes)
        if calibration is None:
            Q = self._q_all(H, N_shift, T)
            out["Q"] = {m: Q[ia] for ia, m in enumerate(modes)}
            return out

        # ---- calibration: per realization, Q_c by the adjoint method, U_c/V_c on the
        #      filtered template (module header) ----
        import time as _time
        n_cal = calibration.n_cal
        R = fir_max_lag(j_first, j_last, (N_shift, N_shift + T - 1), self.runs)
        taps, L_neg, L_pos = calibration.taps_for(R)
        exact = calibration.half_length is None or calibration.half_length >= R
        # turn-on leakage: the taps' largest magnitude at or beyond the distance between the
        # template's turn-on (latest grid shift) and each piece start, relative to c[0]
        leak = 0.0
        for (i0, i1) in self.runs:
            d = i0 - (N_shift + T - 1 + j_first)
            if d >= 1 and d <= L_pos:
                leak = max(leak, float(np.max(np.abs(taps[:, L_neg + d:])) / np.max(np.abs(taps[:, L_neg]))))
        self.calibration_last = dict(reach=int(R), L_neg=int(L_neg), L_pos=int(L_pos), exact=bool(exact),
                                     turn_on_leakage=leak, n_cal=n_cal)
        t0 = _time.time()
        # Q_c(t): the exact correlation of the template's support with W_c = K_c^T w (N-tap filter)
        Qcal = {m: np.empty(n_cal * T, dtype=np.complex128) for m in modes}
        for c in range(n_cal):
            for ia, m in enumerate(modes):
                Qcal[m][c * T:(c + 1) * T] = calibration.data_term(H[ia], c, j_first, j_last, N_shift, T)
        t_q = _time.time() - t0
        crossTermsCal = crossTermsCalV = None
        t_uv = 0.0
        if calibration_self_term:
            # g_c = K_c h on the whole template line ('full' linear convolution): sample j' of
            # g_c is template index j' - L_neg, so it sits at data sample j' - L_neg + k: the
            # template origin moves by L_neg, N_shift -> N_shift - L_neg, and the support
            # becomes [j_first, j_last + L_neg + L_pos] in the new indexing.  That extended
            # support goes to sliding_self_terms as support=, so the method selection (prefix
            # validity) uses it and the rounding residue outside it is ignored.
            t0 = _time.time()
            N_shift_c = N_shift - L_neg
            j_last_c = j_last + L_neg + L_pos
            crossTermsCal, crossTermsCalV = [], []
            for c in range(n_cal):
                Hc = self._filtered_modes(calibration, H, taps[c], L_neg)
                Uc, Vc, self.uv_routes_cal = self._uv_all(Hc, N_shift_c, T, tsel, j_first, j_last_c,
                                                          support=(j_first, j_last_c))
                ct, ctV = self._complete_pairs(Uc, Vc, modes, tsel, T)
                crossTermsCal.append(ct)
                crossTermsCalV.append(ctV)
            t_uv = _time.time() - t0
        out.update(Qcal=Qcal, Ucal=crossTermsCal, Vcal=crossTermsCalV, uv_routes_cal=self.uv_routes_cal,
                   calibration_last=self.calibration_last, t_q=t_q, t_uv=t_uv)
        return out

    @staticmethod
    def _filtered_modes(calibration, H, taps_c, L_neg):
        """K_c applied to every row of H (calibration.filtered_template, a 'full' linear
        convolution).  Outside the filtered support the convolution leaves only rounding
        residue; the caller passes that support to sliding_self_terms, which ignores it."""
        return np.array([calibration.filtered_template(H[a], taps_c, L_neg) for a in range(H.shape[0])])

    def _expand(self, y, tsel, T):
        if tsel.size == T:
            return np.array(y, dtype=np.complex128)
        from scipy.interpolate import CubicSpline
        t = np.arange(T)
        return CubicSpline(tsel, y.real)(t) + 1j * CubicSpline(tsel, y.imag)(t)


# --------------------------------------------------------------------------- Precompute driver

def identify_effective_modes_td(crossTermsOneDetector, fac):
    """td-gs analogue of factored_likelihood.IdentifyEffectiveModesForDetector for
    time-dependent cross terms: a mode p is effective when max_t |U_{p,p2}(t)| exceeds
    fac * |U_{22,22}| at the grid centre for some p2.  Returns the set of kept modes."""
    modes = set(k[1] for k in crossTermsOneDetector)
    ref = abs(crossterm_centre(crossTermsOneDetector[((2, 2), (2, 2))])) * fac
    keep = set()
    for p in modes:
        for p2 in modes:
            if np.max(np.abs(np.asarray(crossTermsOneDetector[(p, p2)]))) > ref:
                keep.add(p)
                break
    return keep


def crossterm_centre(value):
    """Scalar view of a cross term for diagnostics: the array's centre sample when U is
    time-dependent, the value itself otherwise."""
    if np.ndim(value) == 0:
        return value
    v = np.asarray(value)
    return v[v.size // 2]


def generate_td_modes(P, Lmax, extra_waveform_kwargs=None, quiet=True):
    """TD modes for the td-gs code path: lalsimutils.hlmoft with conditioning off.

    Refuses frequency-domain generators (see template_route_is_fd).  Sets P.taper to
    TAPER_NONE so the SEOB (complex_hoft) and TEOB branches apply no taper either, and
    passes no_condition=True so hlmoft's own start tapers are skipped.
    """
    import lalsimulation as lalsim
    if template_route_is_fd(P.approx):
        name = P.approx if isinstance(P.approx, str) else lalsim.GetStringFromApproximant(P.approx)
        raise ValueError(
            "td-gs: approximant %s is generated in the frequency domain (lalsuite "
            "conditioning cannot be switched off); use a time-domain-native model "
            "(e.g. IMRPhenomTPHM, SEOBNRv4PHM, NRSur7dq4, SEOBNRv4)." % name)
    P.taper = lsu.lsu_TAPER_NONE
    kw = dict(extra_waveform_kwargs or {})
    kw['no_condition'] = True
    kw.pop('fd_alignment_postevent_time', None)   # ChooseFDModes-only knob
    hlms = lsu.hlmoft(P, Lmax, **kw)
    if not isinstance(hlms, dict):
        hlms = lsu.SphHarmTimeSeries_to_dict(hlms, Lmax)
    if not quiet:
        m0 = list(hlms.keys())[0]
        print("  [td-gs] %d TD modes, length %d, epoch %.6f s, deltaT %g"
              % (len(hlms), hlms[m0].data.length, float(hlms[m0].epoch), hlms[m0].deltaT))
    return hlms


def precompute_likelihood_terms_td_gs(event_time_geo, t_window, P, data_dict, td_gs_contexts,
                                      Lmax, verbose=True, quiet=False,
                                      extra_waveform_kwargs=None, skip_interpolation=False,
                                      return_calibration_crossterms=False,
                                      calibration_realizations=None, calibration_conjugate=False,
                                      calibration_self_term=True, **refused):
    """td-gs branch of PrecomputeLikelihoodTerms.  Same return tuple.

    calibration_realizations : {det: (N, n_cal) complex, RIFT packing} or None.  With
        realizations each detector's context builds (and caches) its TDCalibrationFilters
        (module header: template filtered with C_c, the context's convention, RIFT's
        --calibration-conjugate-phase as ``calibration_conjugate``) and returns rholms with
        n_cal contiguous blocks of Q_c(t), as the FD code path does.  With
        ``calibration_self_term`` (default) the per-realization time-indexed U_c(t), V_c(t)
        come back as the two trailing dicts of the eight-value return (each det -> list of
        n_cal cross-term dicts of length-N_window arrays); with it False (RIFT's
        --calibration-global-norm) they are None and the likelihood uses the calibration-
        independent U(t), V(t).

    ``refused`` carries the keyword arguments of PrecomputeLikelihoodTerms that this code path
    does not support (glitch/TD-Toeplitz factory, ROM/NR/gwsignal/external EOB generators);
    any of them set to a non-default value is an error.
    """
    from RIFT.likelihood import factored_likelihood as fl
    _not_supported = {
        'IP_factory': None, 'ROM_group': None,
        'ROM_param': None, 'NR_group': None, 'NR_param': None, 'nr_lookup': False,
        'use_external_EOB': False, 'use_gwsignal': False, 'hybrid_use': False,
        'ROM_use_basis': False, 'use_provided_strain': False,
    }
    ignore_threshold = refused.pop('ignore_threshold', None)
    for key, default in _not_supported.items():
        if key in refused and refused[key] not in (default, 0, False, None):
            raise ValueError("td-gs: %s=%r is not supported on the time-domain code path "
                             "(planned for a later stage)" % (key, refused[key]))
    have_cal = calibration_realizations is not None
    if have_cal and not isinstance(calibration_realizations, dict):
        raise ValueError("td-gs: calibration_realizations must be a per-detector dict")

    detectors = list(data_dict.keys())
    first_data = data_dict[detectors[0]]
    P.dist = fl.distMpcRef * 1e6 * lsu.lsu_PC
    P.deltaT = float(first_data.deltaT)
    P.deltaF = 1.0 / (first_data.data.length * P.deltaT)
    if not quiet:
        print("  ++++ Template data being computed for the following binary (td-gs) +++ ")
        P.print_params()
    hlms = generate_td_modes(P, Lmax, extra_waveform_kwargs, quiet=quiet)

    rholms, rholms_intp, crossTerms, crossTermsV = {}, {}, {}, {}
    crossTermsCal = {} if (have_cal and calibration_self_term) else None
    crossTermsCalV = {} if (have_cal and calibration_self_term) else None
    for det in detectors:
        ctx = td_gs_contexts[det]
        data = data_dict[det]
        if abs(float(data.deltaT) - P.deltaT) > 1e-12 * P.deltaT or data.data.length != first_data.data.length:
            raise ValueError("td-gs: detector %s data grid differs from %s" % (det, detectors[0]))
        t_det = fl.ComputeArrivalTimeAtDetector(det, P.phi, P.theta, event_time_geo)
        # Identical bookkeeping to the FD code path (PrecomputeLikelihoodTerms).
        rho_epoch = data.epoch - hlms[list(hlms.keys())[0]].epoch
        t_shift = float(float(t_det) - float(t_window) - float(rho_epoch))
        N_shift = int(t_shift / P.deltaT + 0.5)
        N_window = int(2 * t_window / P.deltaT)
        if have_cal:
            if det not in calibration_realizations:
                raise ValueError("td-gs: no calibration realizations for %s" % det)
            filters = ctx.calibration_filters(calibration_realizations[det], conjugate=calibration_conjugate,
                                              verbose=verbose or not quiet)
            rholms[det], crossTerms[det], crossTermsV[det], ct_cal, ctV_cal = ctx.precompute(
                hlms, N_shift, N_window, verbose=verbose, calibration=filters,
                calibration_self_term=calibration_self_term)
            if calibration_self_term:
                crossTermsCal[det], crossTermsCalV[det] = ct_cal, ctV_cal
        else:
            rholms[det], crossTerms[det], crossTermsV[det] = ctx.precompute(
                hlms, N_shift, N_window, verbose=verbose)
        if ignore_threshold is not None and det == detectors[0]:
            # Mode pruning on the first detector, as the FD code path does with its fiducial
            # cross terms, but with time-dependent U: a mode is kept when max_t |U(t)| of
            # any of its pairs exceeds ignore_threshold * |U_22,22| at the grid centre.
            keep = identify_effective_modes_td(crossTerms[det], ignore_threshold)
            keep = keep.union(set([(l, -m) for (l, m) in keep]))
            print("  Worthwhile modes : ", keep)
            if len(keep) == 0:
                raise ValueError("td-gs: no mode passes ignore_threshold")
            hlms = {m: hlms[m] for m in hlms if m in keep}
            rholms[det] = {m: rholms[det][m] for m in keep if m in rholms[det]}
            crossTerms[det] = {k: v for k, v in crossTerms[det].items() if k[0] in keep and k[1] in keep}
            crossTermsV[det] = {k: v for k, v in crossTermsV[det].items() if k[0] in keep and k[1] in keep}
            if crossTermsCal is not None:
                crossTermsCal[det] = [{k: v for k, v in ct.items() if k[0] in keep and k[1] in keep}
                                      for ct in crossTermsCal[det]]
                crossTermsCalV[det] = [{k: v for k, v in ct.items() if k[0] in keep and k[1] in keep}
                                       for ct in crossTermsCalV[det]]
        # Time grid of one block of the returned series: N_window points at deltaT, or
        # K*N_window at deltaT/K with sub-sample placement (ctx.subsample = K); the same
        # expression for K = 1 (rholm deltaT is then the data deltaT).  With calibration the
        # rholms hold n_cal such blocks and are not interpolated.
        _rho0 = rholms[det][list(rholms[det].keys())[0]]
        _dt_rho = P.deltaT if ctx.subsample == 1 else float(_rho0.deltaT)
        t = np.arange(ctx.subsample * N_window) * _dt_rho + float(rho_epoch + N_shift * P.deltaT)
        if verbose:
            print("For detector", det, "(td-gs) ...")
            print("\tData starts at %.20g" % float(data.epoch))
            print("\tEvent time at detector is: %.18g" % float(t_det))
            print("\tComputed t_shift = %.20g, N_shift = %d" % (t_shift, N_shift))
            print("\tInterpolation starts at time %.20g" % t[0])
            if ctx.subsample > 1:
                print("\tSub-sample placement 1/%d: %d series entries at deltaT/%d%s"
                      % (ctx.subsample, _rho0.data.length, ctx.subsample,
                         " (%d realization blocks)" % len(crossTermsCal[det]) if crossTermsCal is not None else ""))
        if not skip_interpolation and not have_cal:
            rholms_intp[det] = fl.InterpolateRholms(rholms[det], t, verbose=verbose)
        else:
            rholms_intp[det] = None

    # SNR guess as in PrecomputeLikelihoodTerms, using the centre-of-grid U for scale.
    rho_max = 0
    for ifo in rholms:
        for mode in rholms[ifo]:
            u_c = np.real(crossterm_centre(crossTerms[ifo][(mode, mode)]))
            if u_c > 0:
                rho_max_ifo = np.max(np.abs(rholms[ifo][mode].data.data))
                rho_max += rho_max_ifo ** 2 / u_c
    rho_max = np.sqrt(rho_max) / 2.3
    print("SNR guess (internal, from det response) ", rho_max, rho_max ** 2 / 2)
    guess_snr = rho_max

    if return_calibration_crossterms:
        return rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, None, crossTermsCal, crossTermsCalV
    return rholms_intp, crossTerms, crossTermsV, rholms, guess_snr, None
