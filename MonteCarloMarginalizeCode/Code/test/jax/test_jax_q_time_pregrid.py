"""Q time pregrid on the JAX arm: correctness, the factor-1 identity, and its guards.

WHAT IS BEING TESTED, AND AGAINST WHAT

``--q-time-pregrid-factor`` refines the STORED rholm buffers onto a finer time grid
once, at build time, so the per-sample gather interpolates over a step ``factor``
times shorter.  The integration cadence does not change.  The claim is therefore
purely about interpolation error, and the only honest reference for interpolation
error is a Q whose exact value at arbitrary times is known independently.

THE ORACLE.  ``ComputeModeIPTimeSeries`` ends in
``CutCOMPLEX16TimeSeries(rhoTS, 0, N_window)``: the rholm buffer is a CROP of an
inverse-FFT series that is exactly periodic over the whole data segment.  So the
fixture here builds precisely that -- a band-limited series with a known finite
Fourier sum over a long period, cropped -- and evaluates the truth at arbitrary
real positions by summing that Fourier series directly.  Nothing in the fixture
reuses the stencil, the reflection, or the FFT-upsampler under test.

That matters more than it might look, because the alternative references are both
CIRCULAR here.  Converging the Lanczos half-width ``a`` converges onto the
truncated-sinc/zero-extension limit; converging the pregrid factor converges onto
the reflected limit.  They are different limits, they differ at the buffer ends,
and neither is the truth.  The exact Fourier sum is.

WHICH REFLECTION.  The two reflected upsamplers in this codebase disagree, and
each docstring asserts its own convention is the right one:
``jax_ile.core._reflected_fft_upsample`` omits the duplicate turning samples
(period ``2(n-1)``), ``time_marginalization_quadrature.reflected_bandlimited_upsample``
duplicates them (period ``2n``).  They are describing different problems -- see
``core.build_q_time_pregrid`` -- and ``test_duplicated_reflection_is_the_right_one_for_a_crop``
below measures them against the oracle so the choice is evidence rather than
inheritance.
"""
import os
import sys

import numpy as np
import pytest

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from RIFT.likelihood.jax_ile import core as C
from RIFT.likelihood.jax_ile import build_likelihood_data, fused_log_likelihood
from RIFT.likelihood.time_interp_choice import SINC_HALFWIDTH_DEFAULT

# GEOMETRY MIRRORS PRODUCTION, and that turned out to matter more than any
# threshold in this file.  The rholm buffer is 2*0.15 s at 4096 Hz = 1229 samples
# and the marginalization window is +-0.075 s = 614 samples centred in it, so a
# gathered position sits ~300 samples clear of the buffer ends.
#
# The first version of this fixture used a 257-sample crop and evaluated 10 samples
# from its ends.  Every refined-grid number it produced was the REFLECTION BOUNDARY
# error, not the stencil error: factor 8 and factor 32 agreed to 1%, the measured
# convergence rate was 0.96x per doubling instead of ~16x, and the pregrid looked
# only 9x better than production instead of 408x.  A fixture can be wrong about the
# thing it is measuring while every assertion in it still passes.
N_LONG = 4096          # the oracle's period; long relative to the crop, as the real segment is
# Band edge as a fraction of the long series' Nyquist.  The rholm timeseries is
# close to critically sampled -- that is exactly why its stencil error is large --
# so a fixture that band-limits gently would make every stencil look good and
# would not distinguish them.
BAND_FRACTION = 0.92
CROP_START = 700
CROP_N = 1229          # 2*0.15 s at 4096 Hz, the production buffer
CLEARANCE = 300        # production distance from a gathered position to the buffer end


def _oracle_series(seed=11, n_long=N_LONG, band_fraction=BAND_FRACTION):
    """Coefficients of an exactly periodic, band-limited complex series.

    Returns ``(k, c)`` such that ``f(t) = sum_j c[j] exp(2i pi k[j] t / n_long)``
    is exact for any real ``t``, and ``f`` at integer ``t`` is a legitimate stand-in
    for an inverse-FFT rholm series over its full period.

    The amplitude envelope decays with |k| so the series looks like a filtered
    matched-filter output rather than white noise at the band edge, but the band
    edge itself is hard, so the sampling theorem applies exactly.
    """
    rng = np.random.default_rng(seed)
    kmax = int(band_fraction * (n_long // 2))
    k = np.arange(-kmax, kmax + 1)
    env = np.exp(-0.5 * (k / (0.55 * kmax)) ** 2) + 0.15
    c = (rng.standard_normal(k.size) + 1j * rng.standard_normal(k.size)) * env
    return k, c


def _oracle_eval(k, c, t, n_long=N_LONG):
    """Exact ``f(t)`` at arbitrary real ``t`` (array), by direct Fourier sum."""
    t = np.asarray(t, dtype=np.float64)
    phase = np.exp(2j * np.pi * np.outer(t.ravel(), k) / float(n_long))
    return (phase @ c).reshape(t.shape)


def _cropped_Q(k, c, start=CROP_START, n=CROP_N, n_long=N_LONG):
    """The buffer a detector actually gets: ``n`` samples cropped out of the period."""
    return _oracle_eval(k, c, np.arange(start, start + n), n_long)


def _gather_at(Q_col, positions, interp, factor=1):
    """Evaluate one stencil at crop-local COARSE positions, via the shipped gatherers.

    ``factor`` selects the refined grid: the Q column is pre-refined and the
    positions scaled, exactly as :func:`core._q_sample_positions` does.
    """
    Q_col = np.asarray(Q_col)
    if factor != 1:
        fine, _ = C.build_q_time_pregrid(Q_col[None, :], factor)
        Q_col = fine[0]
    pos = np.asarray(positions, dtype=np.float64) * factor
    gather = C._GATHERERS[interp]
    return np.asarray(gather(jnp.asarray(Q_col), jnp.asarray(pos[None, :]),
                             None))[0]


def _interior_positions(n=CROP_N, seed=5, count=400, margin=None):
    """Fractional crop-local positions at the production distance from the ends.

    ``margin`` defaults to :data:`CLEARANCE`, NOT to the stencil footprint.  A
    footprint-sized margin is legal for every stencil and still wrong, because it
    measures the buffer's boundary condition rather than the interpolation -- see
    the geometry note at the top of this file.
    """
    if margin is None:
        margin = CLEARANCE
    rng = np.random.default_rng(seed)
    return rng.uniform(margin, n - 1 - margin, size=count)


# --------------------------------------------------------------------------
# 1.  The factor-1 path is the historical path, bit for bit.
# --------------------------------------------------------------------------

def test_factor_one_returns_the_same_array_object():
    """No copy, no round trip, no reflection: factor 1 must not touch the data.

    Asserting identity rather than equality is deliberate.  ``==`` would still pass
    if factor 1 quietly went through an FFT and came back within an ulp, and an ulp
    of Q is not nothing at rho 652.
    """
    rho = np.arange(12, dtype=np.complex128).reshape(2, 6)
    out, report = C.build_q_time_pregrid(rho, 1)
    assert out is rho
    assert report["factor"] == 1


def test_factor_one_positions_are_bit_identical_to_the_pre_pregrid_expressions():
    """``_q_sample_positions`` at factor 1 reproduces the inline code it replaced.

    The two expressions below are verbatim what ``_accumulate_unit`` and
    ``_accumulate_unit_banded`` computed before the pregrid landed.  Bitwise, not
    ``allclose``: the whole factor-1 claim is that nothing moved at all, and the
    additive/multiplicative reassociation this change introduces on the factor>1
    branch is exactly the kind of thing that shifts a last bit.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals)
    assert data.q_time_pregrid_factor == 1
    rng = np.random.default_rng(3)
    p0 = jnp.asarray(rng.uniform(1000.0, 1200.0, 17))
    t_offsets = jnp.arange(-4, data.npts + 4, dtype=jnp.float64)
    for interp in ("nearest", "linear", "cubic", "sinc"):
        pos, u = C._q_sample_positions(data, p0, t_offsets, interp)
        want_pos = p0[:, None] + t_offsets[None, :]
        assert np.array_equal(np.asarray(pos), np.asarray(want_pos))
        if interp == "nearest":
            assert u is None
        else:
            want_u = C._separable_u(p0)
            assert np.array_equal(np.asarray(u), np.asarray(want_u))


def test_factor_one_stencil_margins_match_the_table_they_replaced():
    """The margin table was inlined twice; hoisting it must not have changed a value."""
    assert C._STENCIL_MARGIN == {"nearest": 1, "linear": 2, "cubic": 3,
                                 "sinc": SINC_HALFWIDTH_DEFAULT + 1}


def _toy_packed(detectors=("H1", "L1"), deltaT=1.0 / 4096, tw=0.02, seed=7):
    """Small packed dict built on the oracle series, so the whole file shares one Q."""
    k, c = _oracle_series(seed=seed)
    npts = int(2 * tw / deltaT)
    tvals = (np.arange(npts) - npts // 2) * deltaT
    tref = 1126259462.413
    modes = ((2, 2), (2, -2))
    K = len(modes)
    rng = np.random.default_rng(seed + 1)
    packed = {}
    for j, det in enumerate(detectors):
        rho = np.stack([_cropped_Q(*_oracle_series(seed=seed + 10 * j + kk),
                                   n=1024) for kk in range(K)])
        U = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        V = rng.standard_normal((K, K)) + 1j * rng.standard_normal((K, K))
        packed[det] = dict(lms=np.array(modes, dtype=int), rholmArray=rho,
                           U=U, V=V, epoch=tref - 512 * deltaT)
    return packed, tvals, deltaT, tref


# --------------------------------------------------------------------------
# 2.  The refined grid is a refinement: it reproduces what it refines.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("factor", [2, 4, 8])
def test_every_factor_th_refined_sample_reproduces_the_input(factor):
    k, c = _oracle_series()
    Q = _cropped_Q(k, c)[None, :]
    fine, report = C.build_q_time_pregrid(Q, factor)
    assert fine.shape == (1, (Q.shape[-1] - 1) * factor + 1)
    assert np.max(np.abs(fine[..., ::factor] - Q)) < 1e-12 * np.max(np.abs(Q))
    assert report["factor"] == factor
    assert report["roundtrip_max"] < 5e-12


def test_refinement_commutes_with_conjugation():
    """Required by the phase-marginalized path, and easy to break unnoticed.

    ``_accumulate_unit`` conjugates the (2,-2) column of the ALREADY-REFINED Q::

        Q = jnp.concatenate([Q[:, 0:1], jnp.conj(Q[:, 1:2])], axis=1)

    so it uses ``conj(refine(x))`` where the physics wants ``refine(conj(x))``.
    Those agree only while the refinement is a real-linear operator -- true of the
    reflect/zero-pad/inverse-FFT construction, and NOT something a reader can see
    from the call site, which is why it is pinned here rather than left to a
    comment.  A boundary convention that treated the two halves of the spectrum
    asymmetrically (e.g. dumping the whole Nyquist bin into one side instead of
    splitting it) would break this and would bias only the phase-marginalized runs.
    """
    k, c = _oracle_series()
    Q = _cropped_Q(k, c)[None, :]
    scale = np.max(np.abs(Q))
    for factor in (2, 8, 16):
        a = np.conj(C.build_q_time_pregrid(Q, factor)[0])
        b = C.build_q_time_pregrid(np.conj(Q), factor)[0]
        rel = float(np.max(np.abs(a - b)) / scale)
        print("  factor %2d  max rel |conj(refine) - refine(conj)| %.3e" % (factor, rel))
        assert rel < 1e-13


def test_positions_stay_separable_on_the_refined_grid():
    """frac(pos) must not vary along the time axis, or ``_separable_u`` lies.

    ``_separable_u`` computes ONE fractional offset per sample and hands it to the
    gatherer for every time column.  That is only legitimate while the time offsets
    are exact integers in the units the gather indexes.  Scaling as ``pos * factor``
    instead of ``p0*factor + t_offsets*factor`` reassociates the product and breaks
    it by up to an ulp per column -- silently, since the result stays finite and
    close.  This pins the property that makes the memory optimisation sound.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals,
                                 q_time_pregrid_factor=8)
    rng = np.random.default_rng(4)
    p0 = jnp.asarray(rng.uniform(300.0, 700.0, 11))
    t_offsets = jnp.arange(0, data.npts, dtype=jnp.float64)
    pos, u = C._q_sample_positions(data, p0, t_offsets, "cubic")
    pos = np.asarray(pos)
    base = np.floor(pos)
    frac = pos - base
    # The base index must advance by EXACTLY the factor per column.  This is the
    # part that is not a rounding nicety: if it did not, a refined window would
    # cover the wrong span of time and the likelihood would still look finite.
    assert np.array_equal(np.diff(base, axis=1),
                          np.full((pos.shape[0], pos.shape[1] - 1), 8.0))
    # and the single offset handed to the gatherer must be the fractional part
    # every column actually has, to within a rounding of the position itself
    # (adding an integer can cross a binade and drop a low mantissa bit).
    tol = 8.0 * np.spacing(np.max(np.abs(pos)))
    assert np.max(np.abs(frac - np.asarray(u))) <= tol
    assert np.max(np.abs(frac - frac[:, :1])) <= tol


# --------------------------------------------------------------------------
# 3.  Accuracy against the exact oracle.  This is the point of the change.
# --------------------------------------------------------------------------

def _stencil_errors(margin=None, count=400):
    k, c = _oracle_series()
    Q = _cropped_Q(k, c)
    pos = _interior_positions(count=count, margin=margin)
    exact = _oracle_eval(k, c, CROP_START + pos)
    scale = np.max(np.abs(Q))
    out = {}
    for name, interp, factor in (("cubic/1", "cubic", 1),
                                 ("sinc/1", "sinc", 1),
                                 ("cubic/8", "cubic", 8),
                                 ("cubic/16", "cubic", 16),
                                 ("sinc/8", "sinc", 8)):
        got = _gather_at(Q, pos, interp, factor=factor)
        out[name] = float(np.max(np.abs(got - exact)) / scale)
    return out


def test_pregrid_cubic_beats_the_production_stencil_against_the_oracle():
    """The factor-8 pregrid must cut the interpolation error by orders of magnitude.

    Thresholds are set well inside the measured margins so this is a REGRESSION
    gate, not a re-measurement; the measured values are printed for the record.
    """
    err = _stencil_errors()
    for name in sorted(err):
        print("  oracle relative max error  %-9s %.3e" % (name, err[name]))
    # Measured on this fixture: sinc/1 1.88e-2, cubic/1 1.25e-1, cubic/8 4.62e-5,
    # i.e. 408x and 2706x.  The gate is 100x, well inside that.
    assert err["cubic/8"] < err["sinc/1"] / 100.0
    assert err["cubic/8"] < err["cubic/1"] / 100.0
    # cubic on the refined grid must also beat the 16-tap Lanczos on the SAME
    # refined grid (measured 4.62e-5 against 4.86e-4).  This is why the pregrid
    # ships with cubic rather than inheriting the arm's 'sinc' default: a fixed
    # 2a-tap window does not gain from a finer grid the way a 4th-order stencil does.
    assert err["cubic/8"] < err["sinc/8"] / 5.0


def test_refined_cubic_error_falls_like_the_fourth_power_of_the_step():
    """Doubling the factor must cut the cubic-Lagrange error by ~16x.

    This is the property that makes the factor a CONVERGENCE knob rather than a
    tuning constant: if the observed rate were ~1 the residual would be dominated
    by something other than the stencil step and the factor would not be buying
    what it claims.  The band is wide (6x-40x) because the fixture's error is a max
    over random sub-sample phases, not a smooth asymptotic.
    """
    k, c = _oracle_series()
    Q = _cropped_Q(k, c)
    pos = _interior_positions(count=400)
    exact = _oracle_eval(k, c, CROP_START + pos)
    scale = np.max(np.abs(Q))
    errs = {}
    for factor in (2, 4, 8, 16, 32):
        got = _gather_at(Q, pos, "cubic", factor=factor)
        errs[factor] = float(np.max(np.abs(got - exact)) / scale)
        print("  factor %3d  relative max error %.3e" % (factor, errs[factor]))
    for lo, hi in ((2, 4), (4, 8)):
        ratio = errs[lo] / errs[hi]
        print("  ratio %d->%d: %.1f" % (lo, hi, ratio))
        assert 8.0 < ratio < 30.0
    # SATURATION, and it is the operational point of the whole measurement: past
    # factor ~8 the residual is the reflection boundary condition, which no factor
    # can reduce.  Measured 8->16 9.5x but 16->32 only 1.4x.  That is why the
    # shipped factor is 8 and not 32: 32 costs 4x the Q memory for ~10x less error
    # than the boundary floor already permits.
    assert errs[16] / errs[32] < 4.0
    assert errs[4] / errs[8] > errs[16] / errs[32]


def test_duplicated_reflection_is_the_right_one_for_a_crop():
    """Settle 2n vs 2(n-1) against the oracle rather than by inheritance.

    ``core.build_q_time_pregrid`` uses the ``2n`` (duplicated turning samples) form
    that the conventional arm shipped, NOT this module's own
    ``_reflected_fft_upsample`` (``2(n-1)``).  Both are boundary heuristics for a
    crop; the choice has to be measured, and it is measured HERE, near the buffer
    end where the two actually differ -- in the deep interior both are exact and
    the test would be blind by construction.

    If a future change reroutes the pregrid through ``_reflected_fft_upsample``
    "for consistency", this fails.
    """
    k, c = _oracle_series()
    Q = _cropped_Q(k, c)
    n = Q.size
    factor = 8
    scale = np.max(np.abs(Q))
    dup, _ = C.build_q_time_pregrid(Q[None, :], factor)
    half = np.asarray(C._reflected_fft_upsample(jnp.asarray(Q[None, :]), factor))
    gather = C._GATHERERS["cubic"]

    rng = np.random.default_rng(9)
    bands = {
        # where production actually gathers
        "interior (clearance %d)" % CLEARANCE: _interior_positions(seed=9),
        # and hard against the ends, where the two conventions differ most
        "near-end": np.concatenate([rng.uniform(4.0, 24.0, 200),
                                    rng.uniform(n - 25.0, n - 5.0, 200)]),
    }
    for band, pos in bands.items():
        exact = _oracle_eval(k, c, CROP_START + pos)
        got = {}
        for name, fine in (("2n (shipped)", dup), ("2(n-1)", half)):
            v = np.asarray(gather(jnp.asarray(fine[0]),
                                  jnp.asarray((pos * factor)[None, :]), None))[0]
            got[name] = float(np.max(np.abs(v - exact)) / scale)
        print("  %-26s  2n %.3e   2(n-1) %.3e   ratio %.1f"
              % (band, got["2n (shipped)"], got["2(n-1)"],
                 got["2(n-1)"] / got["2n (shipped)"]))
        # Measured 16.8x in the interior and 6.9x near the ends.  Gate at 2x so
        # this is a direction check, not a re-measurement.
        assert got["2n (shipped)"] * 2.0 < got["2(n-1)"]


# --------------------------------------------------------------------------
# 4.  Guards.  Each of these is mutation-tested in the PR; see the description.
# --------------------------------------------------------------------------

def test_nearest_is_refused_on_a_refined_grid():
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals,
                                 q_time_pregrid_factor=8)
    p0 = jnp.asarray([100.0, 200.0])
    t_offsets = jnp.arange(0, 4, dtype=jnp.float64)
    with pytest.raises(NotImplementedError, match="nearest"):
        C._q_sample_positions(data, p0, t_offsets, "nearest")
    # and the same call is fine at factor 1, so the refusal is about the pair
    data1 = build_likelihood_data(packed, deltaT, tref, tvals)
    C._q_sample_positions(data1, p0, t_offsets, "nearest")


def test_unrefined_bank_with_a_declared_factor_fails_closed():
    """A stored Q that was never refined must not be silently mis-indexed.

    This is the failure ``banded._base_data`` would produce if the factor were ever
    forwarded to it without refining ``Q_bank``: shapes still broadcast, the
    likelihood still returns finite numbers, and a factor-8 window silently covers
    an eighth of the intended span.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals,
                                 q_time_pregrid_factor=8)
    det = data.detector_names[0]
    dd = data.detectors[det]
    coarse = dd["npts_full_coarse"]
    with pytest.raises(ValueError, match="not refined"):
        C._check_stored_q_length(dd, coarse, 8, "detector %s Q" % det)
    # the real, refined length passes
    C._check_stored_q_length(dd, dd["Q"].shape[0], 8, "detector %s Q" % det)
    # and the guard is live on the accumulator, not just callable directly
    bad = dict(dd)
    bad["Q"] = dd["Q"][:coarse]
    data.detectors[det] = bad
    with pytest.raises(ValueError, match="not refined"):
        C._accumulate_unit(data, jnp.asarray([1.0]), jnp.asarray([0.3]),
                           jnp.asarray([0.5]), jnp.asarray([1.05]),
                           jnp.asarray([0.7]), "cubic", True)


def test_a_declared_factor_without_refinement_metadata_is_refused():
    """The metadata-free escape hatch is for factor 1 only.

    ``_check_stored_q_length`` used to return early whenever ``npts_full_coarse``
    was absent, on the grounds that such a dict cannot have been refined by
    ``build_q_time_pregrid``.  That is true and it is not the hazard.  A caller
    can hand-build a detector dict that DECLARES a factor above 1 and omits the
    metadata; the early return then skipped every check, and
    ``_q_sample_positions`` scaled each index by the factor over a coarse Q.  A
    factor-8 window covers an eighth of the intended span, shapes broadcast, and
    the likelihood returns finite wrong numbers.

    Factor 1 keeps the hatch: no index is scaled, so an unrefined buffer is the
    correct buffer.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals)
    det = data.detector_names[0]
    bare = {k: v for k, v in data.detectors[det].items()
            if k not in ("npts_full_coarse", "q_time_pregrid_factor")}
    assert "npts_full_coarse" not in bare

    # factor 1 is still allowed through with no metadata at all
    C._check_stored_q_length(bare, bare["Q"].shape[0], 1, "detector Q")

    # above factor 1 the missing metadata is itself the fault
    for factor in (2, 8):
        with pytest.raises(ValueError, match="npts_full_coarse"):
            C._check_stored_q_length(bare, bare["Q"].shape[0], factor, "detector Q")

    # and it is refused even when the stored length would satisfy the arithmetic
    # a refined bank of that factor requires, so the guard is not accidentally
    # passing on a length coincidence
    n = bare["Q"].shape[0]
    with pytest.raises(ValueError, match="npts_full_coarse"):
        C._check_stored_q_length(bare, (n - 1)*8 + 1, 8, "detector Q")


def test_a_refined_bank_reaching_a_factorless_namespace_is_refused():
    """The paired half of the ``getattr(..., 1)`` default in ``_q_sample_positions``.

    Duck-typed ``data`` objects (benchmark shims, several tests in this directory)
    do not carry ``q_time_pregrid_factor``, so the lookup defaults to 1.  That is
    only safe because the detector dict carries its OWN declaration and this check
    refuses the mismatch: without it, a refined Q handed to such a namespace would
    be indexed at the coarse stride and quietly evaluate the wrong samples.
    """
    import types
    packed, tvals, deltaT, tref = _toy_packed()
    real = build_likelihood_data(packed, deltaT, tref, tvals,
                                 q_time_pregrid_factor=8)
    det = real.detector_names[0]
    dd = real.detectors[det]
    shim = types.SimpleNamespace(
        feature=None, detectors={det: dd}, detector_names=[det],
        gmst=real.gmst, deltaT=real.deltaT, npts=real.npts,
        tval0=real.tval0, tref_minus_epoch=real.tref_minus_epoch)
    assert not hasattr(shim, "q_time_pregrid_factor")
    with pytest.raises(ValueError, match="being indexed at factor 1"):
        C._accumulate_unit(shim, jnp.asarray([1.0]), jnp.asarray([0.3]),
                           jnp.asarray([0.5]), jnp.asarray([1.05]),
                           jnp.asarray([0.7]), "cubic", True)


def test_bad_factors_are_rejected():
    """Each rejection at ITS OWN entry point, not just through the builder.

    There are two: ``build_q_time_pregrid`` and ``JAXLikelihoodData.__init__``.
    Going through ``build_likelihood_data`` exercises neither in isolation --
    it calls the first, so the second is unreachable that way, and the second
    would have caught a hole in the first.  Mutation-testing showed BOTH
    survived a test written that way: two redundant guards, each masking the
    other, and the pair reads as coverage.  ``build_q_time_pregrid`` is also
    public (`banded` and offline analysis call it directly), so its own
    rejection is not a formality.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    rho = packed["H1"]["rholmArray"]
    # ``match=`` IS THE ASSERTION.  A bare ``pytest.raises(ValueError)`` cannot see
    # this guard at all: delete it and #261's own "Q pregrid factor must be
    # positive" raises ValueError one frame down, so the test passes on a
    # different rejection.  Measured 2026-09-07 by mutation -- the bare form
    # survives deleting BOTH guards, because the value is refused deeper still.
    # Defense in depth is fine; a test that cannot tell which layer refused is not.
    for bad in (0, -3):
        with pytest.raises(ValueError, match="q_time_pregrid_factor must be"):
            C.build_q_time_pregrid(rho, bad)
        with pytest.raises(ValueError, match="q_time_pregrid_factor must be"):
            C.JAXLikelihoodData({}, deltaT, 0.0, tvals, tref,
                                q_time_pregrid_factor=bad)
        with pytest.raises(ValueError, match="q_time_pregrid_factor must be"):
            build_likelihood_data(packed, deltaT, tref, tvals,
                                  q_time_pregrid_factor=bad)


# --------------------------------------------------------------------------
# 5.  End to end: the whole likelihood, and the gradient it exists to provide.
# --------------------------------------------------------------------------

def test_whole_likelihood_moves_toward_the_refined_answer():
    """lnL at factor 1 vs 8 vs 32: factor 8 must sit far closer to the converged value.

    The pregrid is a numerical-accuracy change, so "runs without error" is not the
    assertion.  ``factor 32`` stands in for the converged interpolant here (the
    per-position convergence rate is pinned above); the claim is that the shipped
    factor 8 removes most of the gap that the production stencil leaves.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    rng = np.random.default_rng(12)
    S = 24
    args = (rng.uniform(0, 2 * np.pi, S), rng.uniform(-1.2, 1.2, S),
            rng.uniform(0, np.pi, S), rng.uniform(0, np.pi, S),
            rng.uniform(0, 2 * np.pi, S), np.full(S, 400.0))
    vals = {}
    for factor in (1, 8, 32):
        data = build_likelihood_data(packed, deltaT, tref, tvals,
                                     q_time_pregrid_factor=factor)
        vals[factor] = np.asarray(fused_log_likelihood(data, *args,
                                                       interp="sinc" if factor == 1
                                                       else "cubic"))
    gap1 = np.max(np.abs(vals[1] - vals[32]))
    gap8 = np.max(np.abs(vals[8] - vals[32]))
    print("  max|lnL(a=8 coarse) - lnL(f=32)| = %.4e" % gap1)
    print("  max|lnL(f=8)        - lnL(f=32)| = %.4e" % gap8)
    assert gap8 < gap1 / 10.0


def test_gradient_still_flows_through_the_refined_gather():
    """The whole reason this arm exists is ``jax.grad``; a pregrid must not break it.

    A gather that lost its dependence on ``pos`` -- e.g. by rounding the scaled
    position to the refined grid -- would still return sensible values and a
    silently ZERO sky gradient.
    """
    packed, tvals, deltaT, tref = _toy_packed()
    data = build_likelihood_data(packed, deltaT, tref, tvals,
                                 q_time_pregrid_factor=8)

    def f(ra):
        return fused_log_likelihood(
            data, ra, jnp.asarray([0.3]), jnp.asarray([0.5]),
            jnp.asarray([1.05]), jnp.asarray([0.7]), jnp.asarray([400.0]),
            interp="cubic")[0]

    g = jax.grad(f)(jnp.asarray([1.2]))
    assert np.all(np.isfinite(np.asarray(g)))
    assert np.max(np.abs(np.asarray(g))) > 0.0


# --------------------------------------------------------------------------
# 6.  The SEAMS.  Everything above tests the library; a flag that never reaches
#     it is still a no-op, and a --help that parses is not a wired option.
# --------------------------------------------------------------------------

def test_wrapper_forwards_the_factor_to_the_data():
    """``build_data_from_precompute`` must carry the factor into the built data.

    Exercised through the REAL function with the two expensive production calls
    stubbed, rather than by reading the source: a keyword that is accepted,
    documented and then dropped on the floor is exactly the silent-no-op shape
    this option could most easily take.
    """
    from RIFT.likelihood.jax_ile import wrapper as W

    packed, tvals, deltaT, tref = _toy_packed(detectors=("H1", "L1"))

    class _P:
        deltaT = None

    P = _P()
    P.deltaT = deltaT
    dets = list(packed)

    def _fake_precompute(*a, **k):
        empty = {d: {} for d in dets}
        return empty, empty, empty, {d: {} for d in dets}, 1.0, None

    def _fake_pack(keys, intp, rho, ct, ctV, _packed=packed, _dets=iter(dets)):
        det = next(_dets)
        d = _packed[det]
        return (d["lms"], None, None, d["U"], d["V"], d["rholmArray"], None,
                d["epoch"])

    old_pre = W.factored_likelihood.PrecomputeLikelihoodTerms
    old_pack = W.factored_likelihood.PackLikelihoodDataStructuresAsArrays
    try:
        W.factored_likelihood.PrecomputeLikelihoodTerms = _fake_precompute
        W.factored_likelihood.PackLikelihoodDataStructuresAsArrays = _fake_pack
        data, _extras = W.build_data_from_precompute(
            P, {d: None for d in dets}, {d: None for d in dets}, 1126259462.0,
            0.15, 0.075, 2, 1700.0, tvals=tvals, q_time_pregrid_factor=8)
    finally:
        W.factored_likelihood.PrecomputeLikelihoodTerms = old_pre
        W.factored_likelihood.PackLikelihoodDataStructuresAsArrays = old_pack

    n_coarse = packed[dets[0]]["rholmArray"].shape[-1]
    assert data.q_time_pregrid_factor == 8
    for det in dets:
        dd = data.detectors[det]
        assert dd["q_time_pregrid_factor"] == 8
        assert dd["npts_full_coarse"] == n_coarse
        assert dd["Q"].shape[0] == (n_coarse - 1) * 8 + 1


def test_driver_passes_the_factor_at_its_call_site():
    """The driver's own ``build_data_from_precompute`` call must name the option.

    Parsed with ``ast`` rather than grepped, so a mention in a comment, a help
    string or a dead branch does not satisfy it.  This is the one seam a library
    test cannot reach: the driver is a script with no ``.py`` extension and its
    ``analyze_one`` needs real frames to run.
    """
    import ast

    here = os.path.dirname(os.path.abspath(__file__))
    driver = os.path.join(here, "..", "..", "bin",
                          "integrate_likelihood_extrinsic_jax")
    tree = ast.parse(open(driver).read())
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
        if name != "build_data_from_precompute":
            continue
        sites.append({k.arg for k in node.keywords if k.arg})
    assert sites, "driver no longer calls build_data_from_precompute"
    for kwargs in sites:
        assert "q_time_pregrid_factor" in kwargs, (
            "a build_data_from_precompute call site does not forward "
            "--q-time-pregrid-factor; the flag would parse and do nothing")


def test_pregrid_and_the_phase_marg_mode_permutation_compose():
    """Both packings of the (2,+-2) pair must agree ON A REFINED GRID.

    This path is born at the merge and neither side covers it.  #272 made
    ``_accumulate_unit`` permute ``lms``, ``Q``, ``U`` and ``V`` to canonical
    order under phase marginalization; its fixtures never set
    ``q_time_pregrid_factor``.  This file exercises the pregrid; its fixtures
    never pack the pair the other way round.  The permutation takes ``Q`` on
    axis 1 while every pregrid index acts on axis 0, so the two are expected to
    be independent -- but "expected to be independent" is the claim, and the
    merge is where it first has to hold.

    The last assertion is what stops this being vacuous.  Permuting a mode axis
    would agree at every factor even if the pregrid were doing nothing at all,
    so the refined answer must first be shown to DIFFER from the coarse one.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import test_jax_phase_marg_mode_order as M

    pk = M._packed()
    swapped = M._relabel(pk, [1, 0])
    th = M._angles()

    def acc(packed, factor):
        tw = 32 * (1.0 / 1024) / 2.0
        data = build_likelihood_data(packed, 1.0 / 1024, M.TREF,
                                     np.linspace(-tw, tw, 32),
                                     q_time_pregrid_factor=factor)
        k, r = C._accumulate_unit(data, *th, "cubic", True, guard=0)
        return np.asarray(k), np.asarray(r)

    for factor in (1, 2, 8):
        k0, r0 = acc(pk, factor)
        k1, r1 = acc(swapped, factor)
        scale = max(np.abs(k0).max(), 1.0)
        assert np.abs(k0 - k1).max() <= 1e-12 * scale, (
            "packing order changes kappa at q_time_pregrid_factor=%d" % factor)
        assert np.abs(r0 - r1).max() <= 1e-12 * max(np.abs(r0).max(), 1.0), (
            "packing order changes rho^2 at q_time_pregrid_factor=%d" % factor)

    k1c, _ = acc(pk, 1)
    k8c, _ = acc(pk, 8)
    assert np.abs(k1c - k8c).max() / max(np.abs(k1c).max(), 1.0) > 1e-9, (
        "factor 8 reproduces factor 1 on this fixture, so the agreement above "
        "says nothing about the refined grid")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-q"]))
