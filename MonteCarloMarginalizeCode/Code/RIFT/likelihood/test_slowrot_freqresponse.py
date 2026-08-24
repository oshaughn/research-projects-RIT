"""
Validate slowrot_freqresponse (Path D, frequency-dependent finite-size antenna response).

Checks:
  (A) LONG-WAVELENGTH LIMIT: antenna_response_fd(..., f=0) == lal.ComputeDetAMResponse to
      machine precision, over many random (ra,dec,psi) and H1/L1/V1/K1.  KEY CHECK.
  (B) FREE-SPECTRAL-RANGE STRUCTURE: the single-arm transfer's first null sits at the
      expected frequency c / (L (1 + a.n)); |F(f)| departs from |F(0)| on the f_FSR scale.
  (D) THE UNPAIRED NYQUIST BIN: the response weights must be Hermitian on the grid, which
      at the one bin that stands for both +fNyq and -fNyq means REAL -- pinned both as a
      consistency property (conj(W h) == W conj(h), which crossTermsV_fr relies on) and by
      VALUE (the Hermitian average).  See issue #164.
  (C) IN-BAND MAGNITUDE: fractional response change |F(f)/F(0) - 1| at 1 kHz and 2 kHz for
      (i) 4-km LIGO and (ii) a 40-km CE arm -- quantifies whether the effect matters in band.

Run directly:
    python test_slowrot_freqresponse.py
or under pytest (the test_* functions assert).
"""
from __future__ import print_function, division

import numpy as np
import lal
import lalsimulation as lalsim

try:
    import RIFT.likelihood.slowrot_freqresponse as fr
except Exception:
    import importlib.util, os
    _here = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        "slowrot_freqresponse", os.path.join(_here, "slowrot_freqresponse.py"))
    fr = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(fr)

DETECTORS = ["H1", "L1", "V1", "K1"]
_TOL = 1e-11                       # machine-precision agreement vs ComputeDetAMResponse

_RNG = np.random.RandomState(20260707)
_CASES = []
for _det in DETECTORS:
    for _ in range(8):
        _ra = _RNG.uniform(0, 2 * np.pi)
        _dec = np.arcsin(_RNG.uniform(-1, 1))
        _psi = _RNG.uniform(0, np.pi)
        _gmst = _RNG.uniform(0, 2 * np.pi)
        _CASES.append((_det, _ra, _dec, _psi, _gmst))


# ---- (A) long-wavelength limit vs LAL ------------------------------------------------
def test_long_wavelength_limit_matches_lal():
    worst = 0.0
    for det, ra, dec, psi, gmst in _CASES:
        lald = lalsim.DetectorPrefixToLALDetector(det)
        Fp_ref, Fc_ref = lal.ComputeDetAMResponse(lald.response, ra, dec, psi, gmst)
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst)
        e = max(abs(Fp - Fp_ref), abs(Fc - Fc_ref))
        worst = max(worst, e)
        assert e < _TOL, "FD(f=0) mismatch %g for %s ra=%.3f dec=%.3f psi=%.3f" % (
            e, det, ra, dec, psi)
    print("(A) long-wavelength limit: worst |F_fd(0) - ComputeDetAMResponse| = %.3e" % worst)
    return worst


def test_zero_frequency_is_real():
    """At f=0 the response must be purely real (imag part = 0 to machine precision)."""
    worst = 0.0
    for det, ra, dec, psi, gmst in _CASES:
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst)
        worst = max(worst, abs(Fp.imag), abs(Fc.imag))
    assert worst < 1e-14, "imag(F(0)) = %g" % worst
    print("(A') Im F(0): worst = %.3e" % worst)


# ---- (B) free-spectral-range / sinc-null structure -----------------------------------
def _first_local_min(a, L, fmax_frac=3.5, N=700000):
    """Frequency and depth of the first local minimum of |D~| in (0, fmax_frac*f_FSR]."""
    fFSR = fr.free_spectral_range(L)
    fg = np.linspace(1.0, fmax_frac * fFSR, N)
    mag = np.abs(fr.single_arm_transfer(a, fg, L))
    # first interior local minimum
    lo = (mag[1:-1] < mag[:-2]) & (mag[1:-1] < mag[2:])
    idx = np.nonzero(lo)[0]
    i = idx[0] + 1 if len(idx) else int(np.argmin(mag))
    return fg[i], mag[i], fFSR


def test_single_arm_null_at_fsr_for_transverse():
    """a.n = 0 (source transverse to arm): D~ = e^{-i2pi fT} sinc(2fT), EXACT null at f_FSR."""
    L = 40000.0
    T = L / fr.C_SI
    fFSR = fr.free_spectral_range(L)
    fg = np.linspace(1.0, 4.0 * fFSR, 200001)
    D = fr.single_arm_transfer(0.0, fg, L)
    ref = np.exp(-1j * 2.0 * np.pi * fg * T) * np.sinc(2.0 * fg * T)
    ident = np.max(np.abs(D - ref))
    assert ident < 1e-13, "a=0 closed form mismatch %g" % ident
    # exact first null at f_FSR
    val_at_fsr = abs(fr.single_arm_transfer(0.0, fFSR, L))
    assert val_at_fsr < 1e-12, "|D~(a=0, f_FSR)| = %g (should vanish)" % val_at_fsr
    print("(B) a.n=0: D~=e^{-i2pi fT} sinc(2fT) to %.2e; EXACT first null at f_FSR=c/2L=%.5g Hz"
          % (ident, fFSR))


def test_single_arm_dip_structure():
    """General a.n: |D~| dips on the f_FSR scale (exact nulls only for a.n=0)."""
    L = 40000.0
    for a in [-0.7, -0.3, 0.0, 0.4, 0.8]:
        fmin, depth, fFSR = _first_local_min(a, L)
        print("(B') a.n=%+.2f: first |D~| dip at %.4g Hz = %.3f f_FSR   (|D~|min=%.2e)"
              % (a, fmin, fmin / fFSR, depth))
        assert 0.3 * fFSR < fmin < 3.5 * fFSR, "dip off the f_FSR scale for a=%g" % a


def test_fsr_scale_departure():
    """|F(f)| departs from |F(0)| by O(1) once f ~ f_FSR (use a 40-km CE arm)."""
    det, ra, dec, psi, gmst = "H1", 1.2, 0.5, 0.7, 2.0
    L = 40000.0
    fFSR = fr.free_spectral_range(L)
    Fp0, Fc0 = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst, L_arm=L)
    F0 = abs(Fp0 + 1j * Fc0)
    for frac in [0.01, 0.1, 0.5, 1.0]:
        f = frac * fFSR
        Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, f, gmst=gmst, L_arm=L)
        rel = abs(abs(Fp + 1j * Fc) - F0) / F0
        print("(B') f/f_FSR=%.2f (f=%.4g Hz): ||F(f)|-|F(0)||/|F(0)| = %.3e" % (frac, f, rel))


# ---- (C) in-band magnitude: LIGO vs CE ----------------------------------------------
# ---------------------------------------------------------------- (D) the unpaired Nyquist bin
def _rift_fvals(npts, deltaF):
    """RIFT two-sided packing, f[k] = deltaF*(npts/2 - k): +fNyq at k=0, no -fNyq."""
    return deltaF * (npts / 2.0 - np.arange(npts))


def _geom(L):
    return dict(L=float(L), T=float(L) / lal.C_SI)


_NYQ_GEOM = _geom(4000.0)

# The projection must fire for EVERY geometry and basis size, not just the one that
# happened to expose the bug.  --freqresponse-arm-length and --freqresponse-qmax are both
# user-settable (bin/integrate_likelihood_extrinsic_batchmode), and the defect's size at the
# unpaired bin depends strongly on L: |Im W_p|/|W_p| for p = 1..5 is
#   L =  4 km   0.9935 0.9853 0.1708 0.9853 0.1708
#   L = 10 km   0.9596 0.9093 0.4162 0.9093 0.4162
#   L = 40 km   0.4655 0.1456 0.9893 0.1456 0.9893     (CE; 47% of |W| there)
# A builder that projects only at (4 km, Qmax=4) passed every check in this file until
# these loops existed.
# (arm length, Qmax, npts) -- npts varies for the same reason: a builder that projects
# only at npts = 16384 passed every check here until the grid size moved too.
_NYQ_CASES = [(4000.0, 4, 16384), (4000.0, 0, 8192), (4000.0, 1, 32768),
              (4000.0, 6, 4096), (10000.0, 4, 16384), (40000.0, 2, 32768),
              (40000.0, 6, 8192)]


def test_unpaired_extreme_bin_predicate():
    """The mask must fire on the RIFT packing and on NOTHING else that is well defined."""
    f = _rift_fvals(16, 1.0)
    m = fr.unpaired_extreme_bin(f)
    assert m.sum() == 1 and m[0], "RIFT packing: expected exactly bin 0 (%r)" % np.where(m)
    for name, axis in [
            ("one-sided band", np.arange(30., 513.)),          # top of a band is NOT Nyquist
            ("symmetric", np.arange(-4., 5.)),                 # extreme bin HAS a partner
            ("fftfreq order", np.concatenate((np.arange(0., 4.), np.arange(-4., 0.)))),
            ("single sample", np.array([7.])),
            ("all zero", np.zeros(4))]:
        mm = fr.unpaired_extreme_bin(axis)
        if name == "fftfreq order":
            # -fNyq is the unpaired one there; it must still be found, and only it.
            assert mm.sum() == 1 and axis[mm][0] == -4., "%s: got %r" % (name, axis[mm])
        else:
            assert not mm.any(), "%s: nothing is unpaired here, but mask flagged %r" % (
                name, axis[mm])


def test_weights_hermitian_on_the_grid():
    """W_p(-f) = conj(W_p(f)) at every PAIRED bin, and real at the unpaired one."""
    for L, Qmax, npts in _NYQ_CASES:
        _hermitian_one_case(L, Qmax, npts)


def _hermitian_one_case(L, Qmax, npts):
    deltaF = 0.25
    f = _rift_fvals(npts, deltaF)
    W = fr.finite_size_response_weights(f, _geom(L), Qmax)
    k = np.arange(1, npts)                      # every bin except the self-paired k=0
    for p in range(W.shape[0]):
        d = np.max(np.abs(W[p][npts - k] - np.conj(W[p][k])))
        scale = np.max(np.abs(W[p]))
        print("L=%6.0f Qmax=%d W_%d: paired-bin Hermiticity %.2e (scale %.2e)"
              % (L, Qmax, p, d, scale))
        assert d <= 1e-12 * scale, (
            "W_%d not Hermitian at paired bins (L=%g, Qmax=%d): %g" % (p, L, Qmax, d))
        im = abs(np.imag(W[p][0])) / max(abs(W[p][0]), 1e-300)
        print("L=%6.0f Qmax=%d W_%d(+fNyq) = %+.6e %+.6ej  |Im|/|W| = %.2e"
              % (L, Qmax, p, W[p][0].real, W[p][0].imag, im))
        assert im <= 1e-14, (
            "W_%d is complex at the UNPAIRED Nyquist bin at L=%g, Qmax=%d (|Im|/|W| = %g).  "
            "That bin stands "
            "for both +fNyq and -fNyq, so Hermiticity there means real, and crossTermsV_fr "
            "identifies conj(W h) with W conj(h) on the strength of it -- see issue #164"
            % (p, L, Qmax, im))


def test_weight_commutes_with_conjugation_at_nyquist():
    """conj(W_p h) == W_p conj(h), the identity crossTermsV_fr is built on.

    CONSISTENCY only: any REAL value at the unpaired bin satisfies this, so read it with
    test_nyquist_weight_value_is_the_hermitian_average, which pins the value.
    """
    npts, deltaF = 1024, 4.0
    f = _rift_fvals(npts, deltaF)
    W = fr.finite_size_response_weights(f, _NYQ_GEOM, 4)
    rng = np.random.default_rng(20260819)
    h = rng.normal(size=npts) + 1j * rng.normal(size=npts)
    h[0] = 3.0 - 1.5j                            # make the Nyquist bin carry real weight
    assert abs(h[0]) > 1e-3 * np.max(np.abs(h)), "vacuous without Nyquist content"
    for p in range(W.shape[0]):
        # conj in the TIME domain <-> conjugate-and-reverse in this packing (k -> npts-k)
        def conj_spec(x):
            xc = np.conj(x)
            return np.concatenate(([xc[0]], xc[1:][::-1]))
        a = conj_spec(W[p] * h)                  # conj(W h)
        b = W[p] * conj_spec(h)                  # W conj(h)
        err = np.max(np.abs(a - b)) / np.max(np.abs(b))
        print("W_%d: conj/weight commutation rel err = %.2e" % (p, err))
        assert err <= 1e-14, (
            "conj(W_%d h) != W_%d conj(h) (rel %g): the unpaired Nyquist bin is not real, "
            "so crossTermsV_fr = <conj(W h)|W' h'> is not the term it claims -- issue #164"
            % (p, p, err))


def _continuum_weights(fvals, geom, Qmax):
    """W_p(f) straight from the documented formula, with NO Nyquist projection.

    Independent of the projection logic under test, so it can say what the projection is
    allowed to touch.  W_0 = 1; W_{1+q} = e^{-i2pi f T} c_q(f) - [q==0].

    THIS IS A HAND COPY of finite_size_response_weights' formula, and deliberately so: the
    value guard's reference comes from the production function itself (evaluated on a
    one-sided axis, where the projection declines), so it pins "projected == Re(unprojected)"
    and nothing about the unprojected value.  This copy is the only thing in the file that
    would notice the FORMULA changing -- e.g. flipping the sign of the delay phase passes
    every other check here.  If the formula is deliberately revised, revise this too, and
    read a large "bins changed" count above as formula drift rather than a bad projection.
    """
    fvals = np.asarray(fvals, dtype=float)
    c = fr.finite_size_c_coeffs(fvals, geom['L'], Qmax)
    phase = np.exp(-1j * 2.0 * np.pi * fvals * geom['T'])
    W = np.empty((Qmax + 2, fvals.shape[0]), dtype=complex)
    W[0] = 1.0
    for q in range(Qmax + 1):
        W[1 + q] = phase * c[q] - (1.0 if q == 0 else 0.0)
    return W


def test_weights_untouched_away_from_the_unpaired_bin():
    """The projection must change the UNPAIRED bin and nothing else, on any axis.

    Without this, a builder that took the real part of EVERY bin -- destroying the entire
    response phase -- passes the Hermiticity, commutation and value checks above, because a
    wholly real weight is trivially Hermitian and its unpaired bin is trivially its own real
    part.  Same for one that projects the top of a ONE-SIDED analysis band, which is not a
    Nyquist bin at all.  Both were live holes until this test existed (issue #164).
    """
    cases = [("two-sided RIFT packing", _rift_fvals(4096, 1.0), 1),
             ("two-sided, other npts", _rift_fvals(2048, 0.5), 1),
             ("two-sided, large npts", _rift_fvals(32768, 0.125), 1),
             ("one-sided band", np.arange(30., 1025.), 0),
             ("symmetric axis", np.arange(-64., 65.), 0)]
    for L, Qmax, _npts in _NYQ_CASES:
        for label, f, n_expected in cases:
            _scope_one_case("%s L=%g Q=%d" % (label, L, Qmax), f, n_expected, L, Qmax)


def _scope_one_case(label, f, n_expected, L, Qmax):
    if True:
        W = fr.finite_size_response_weights(f, _geom(L), Qmax)
        ref = _continuum_weights(f, _geom(L), Qmax)
        changed = np.where(np.any(np.abs(W - ref) > 0, axis=0))[0]
        print("%-24s bins changed by the projection: %d (expected %d)"
              % (label, changed.size, n_expected))
        assert changed.size == n_expected, (
            "%s: projection touched %d bins (f = %r), expected %d.\n"
            "  A SMALL excess means the projection over-reached -- it must change only a "
            "genuinely unpaired extreme bin (issue #164).\n"
            "  A LARGE excess (most/all bins) instead means the PRODUCTION FORMULA moved "
            "away from _continuum_weights below, which is a hand copy of it; fix the copy "
            "or the formula, not the projection."
            % (label, changed.size, f[changed][:8], n_expected))
        if n_expected:
            assert changed[0] == 0 and f[0] == np.max(np.abs(f)), (
                "%s: the changed bin is not +fNyq" % label)
            # and it changed by exactly dropping the imaginary part
            assert np.max(np.abs(W[:, 0] - ref[:, 0].real)) <= 1e-300 + 1e-14 * np.max(
                np.abs(ref[:, 0])), "%s: unpaired bin is not Re(continuum)" % label


def test_nyquist_weight_value_is_the_hermitian_average():
    """PIN THE VALUE: the unpaired bin must be Re W_p(+fNyq), not merely some real number.

    The reference is the UNPROJECTED continuum weight, obtained by evaluating on a
    one-sided axis (where unpaired_extreme_bin correctly declines to touch anything, since
    the top of a one-sided band is not a Nyquist bin).  Zeroing the bin, or taking |W|, or
    any other real value, fails here while passing the commutation test above.
    """
    for L, Qmax, npts in _NYQ_CASES:
        _value_one_case(L, Qmax, npts)


def _value_one_case(L, Qmax, npts):
    deltaF = 0.25
    f = _rift_fvals(npts, deltaF)
    fnyq = deltaF * npts / 2.0
    geom = _geom(L)
    W = fr.finite_size_response_weights(f, geom, Qmax)

    one_sided = np.array([1.0, 10.0, 100.0, fnyq])          # positive only -> no projection
    assert not fr.unpaired_extreme_bin(one_sided).any()
    W_cont = fr.finite_size_response_weights(one_sided, geom, Qmax)[:, -1]

    for p in range(W.shape[0]):
        want = 0.5 * (W_cont[p] + np.conj(W_cont[p]))       # the Hermitian average, = Re
        got = W[p][0]
        d = abs(got - want) / max(abs(W_cont[p]), 1e-300)
        print("L=%6.0f Qmax=%d W_%d(+fNyq): got %+.9e  want Re = %+.9e  "
              "(|W_cont| = %.3e, rel %.2e)"
              % (L, Qmax, p, got.real, want.real, abs(W_cont[p]), d))
        assert d <= 1e-14, (
            "W_%d at the unpaired Nyquist bin (L=%g, Qmax=%d) is %r, not the Hermitian "
            "average %r of the continuum weight.  Any real value passes the commutation "
            "check; only this one is the response the grid's real (-1)^j Nyquist mode "
            "actually sees -- see #164" % (p, L, Qmax, got, want))


def _fractional_change(det, L, freqs, n_sky=4000):
    """Median-over-sky of complex |F(f)/F(0)-1| AND amplitude-only ||F(f)|-|F(0)||/|F(0)|,
    excluding sky positions near antenna-pattern nulls (|F(0)|<0.3) where the ratio blows
    up for reasons unrelated to the finite-size effect.

    Returns (complex_med, amp_med) arrays over freqs.  The complex ratio is DOMINATED by
    the overall light-crossing phase e^{-i2 pi f L/c} (a benign direction-independent
    delay of L/c, degenerate with coalescence time); the amplitude-only change is the
    physically meaningful measure of antenna-pattern SHAPE distortion.
    """
    rng = np.random.RandomState(1234)
    comp = [[] for _ in freqs]
    amp = [[] for _ in freqs]
    for _ in range(n_sky):
        ra = rng.uniform(0, 2 * np.pi)
        dec = np.arcsin(rng.uniform(-1, 1))
        psi = rng.uniform(0, np.pi)
        gmst = rng.uniform(0, 2 * np.pi)
        Fp0, Fc0 = fr.antenna_response_fd(det, ra, dec, psi, 0.0, gmst=gmst, L_arm=L)
        F0 = complex(Fp0) + 1j * complex(Fc0)
        if abs(F0) < 0.3:
            continue
        for i, f in enumerate(freqs):
            Fp, Fc = fr.antenna_response_fd(det, ra, dec, psi, f, gmst=gmst, L_arm=L)
            Ff = complex(Fp) + 1j * complex(Fc)
            comp[i].append(abs(Ff / F0 - 1.0))
            amp[i].append(abs(abs(Ff) - abs(F0)) / abs(F0))
    return (np.array([np.median(c) for c in comp]),
            np.array([np.median(a) for a in amp]))


def test_in_band_magnitude_ligo_vs_ce():
    freqs = [1000.0, 2000.0]
    ligo_c, ligo_a = _fractional_change("H1", 3994.5, freqs)
    ce_c, ce_a = _fractional_change("H1", 40000.0, freqs)
    print("(C) IN-BAND fractional response change (median over sky, away from nulls):")
    print("    complex |F(f)/F(0)-1|  (incl. benign overall e^{-i2pi f L/c} delay phase):")
    for i, f in enumerate(freqs):
        print("       f=%5.0f Hz :  LIGO(4km) = %.3e    CE(40km) = %.3e" % (f, ligo_c[i], ce_c[i]))
    print("    amplitude-only ||F(f)|-|F(0)||/|F(0)|  (pattern-SHAPE distortion, physical):")
    for i, f in enumerate(freqs):
        print("       f=%5.0f Hz :  LIGO(4km) = %.3e    CE(40km) = %.3e    (CE/LIGO ~%.0fx)"
              % (f, ligo_a[i], ce_a[i], ce_a[i] / max(ligo_a[i], 1e-30)))
    # LIGO amplitude distortion is sub-percent (tiny); CE is tens of percent (>> LIGO).
    assert ligo_a[0] < 1e-2, "LIGO 1kHz amplitude change unexpectedly large: %g" % ligo_a[0]
    assert ligo_a[1] < 2e-2, "LIGO 2kHz amplitude change unexpectedly large: %g" % ligo_a[1]
    assert ce_a[0] > 3e-2, "CE 1kHz amplitude change unexpectedly small: %g" % ce_a[0]
    assert ce_a[1] > ligo_a[1] * 30, "CE should be >> LIGO at 2 kHz"
    return ligo_a, ce_a


def test_ce_is_100x_longer_effect():
    """Finite-size amplitude distortion scales ~ (f L / c)^2 ; 10x arm -> ~10^2x effect."""
    f = 2000.0
    _, ligo_a = _fractional_change("H1", 3994.5, [f])
    _, ce_a = _fractional_change("H1", 40000.0, [f])
    ratio = ce_a[0] / max(ligo_a[0], 1e-30)
    print("(C') CE/LIGO amplitude-distortion ratio at 2 kHz = %.1f (expect ~10^2 for 10x arm)"
          % ratio)
    assert 30 < ratio < 250, "unexpected CE/LIGO scaling: %g" % ratio


if __name__ == "__main__":
    test_unpaired_extreme_bin_predicate()
    test_weights_hermitian_on_the_grid()
    test_weight_commutes_with_conjugation_at_nyquist()
    test_weights_untouched_away_from_the_unpaired_bin()
    test_nyquist_weight_value_is_the_hermitian_average()
    print("=" * 78)
    wA = test_long_wavelength_limit_matches_lal()
    test_zero_frequency_is_real()
    print("-" * 78)
    test_single_arm_null_at_fsr_for_transverse()
    test_single_arm_dip_structure()
    test_fsr_scale_departure()
    print("-" * 78)
    test_in_band_magnitude_ligo_vs_ce()
    test_ce_is_100x_longer_effect()
    print("=" * 78)
    print("ALL SLOWROT FREQ-RESPONSE CHECKS PASSED  (worst f=0 residual %.3e)" % wA)
