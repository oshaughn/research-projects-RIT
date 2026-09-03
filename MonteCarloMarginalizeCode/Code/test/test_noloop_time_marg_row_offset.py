#!/usr/bin/env python
"""The time-marginalization offset in the vectorized NoLoop likelihood must be
PER EXTRINSIC SAMPLE, not per batch.

WHAT IS BEING TESTED, AND AGAINST WHAT
--------------------------------------
``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` marginalizes over time by

    L   = simps(exp(lnL_t - lnLmax), dx=deltaT, axis=-1)
    lnL = lnLmax + log(L)

on ``lnL_t`` of shape ``(npts_extrinsic, npts_time)``.  The expression is exactly
offset-invariant in real arithmetic, so ANY ``lnLmax`` gives the same answer --
in real arithmetic.  In float64 it does not: a single scalar batch maximum shifts
every extrinsic sample by the LOUDEST sample's peak, and any row sitting more
than ~745 nats below it underflows ``exp()`` to 0 across the whole time axis, so
``L = 0`` and ``log(L) = -inf`` at a sample where the likelihood is finite and
perfectly ordinary.  With ``lnL ~ rho^2/2`` at the peak and ``lnL ~ 0`` for a
typical prior draw, that fires above ``rho ~ 40`` and takes out the BULK of the
prior, not a tail -- which is what collapses ``mcsamplerAV`` on loud events.
See oshaughnessy-junior/research-projects-RIT#232 for the real-data measurement.

The reference here is NOT a reimplementation of the likelihood.  Each test asks
the SHIPPED function for its own verbatim ``lnL_t`` (``return_lnLt=True``, a path
that does no shifting at all) and integrates that with the SAME Simpson rule the
function uses, row by row, each row offset by its own maximum.  The only thing
that differs between the reference and the code under test is the offset -- which
is the whole subject of the test.

``keepdims=True`` is load-bearing and is guarded separately.  With a bare
``axis=-1`` the ``(n,)`` maximum broadcasts along the TIME axis instead of the
sample axis.  That RAISES when ``npts_extrinsic != npts_time`` -- and is silently
wrong when they are equal, which is why one case below is deliberately square.
(The shipped path always builds a 2-D ``lnL_t``: the function requires
array-valued extrinsic parameters, so a 1-D ``lnL_t`` does not arise here.)
"""
from __future__ import print_function, division

import os

os.environ.setdefault("RIFT_LOWLATENCY", "1")

import numpy as np
import pytest
from scipy import integrate

import lal
import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl

# lal / lalsimutils are imported at module scope on purpose, NOT via importorskip:
# lalsuite is in requirements.txt and both CI jobs that run this file install it, so a
# missing lal here is a broken job, not an unsupported platform -- and an importorskip
# would turn that into a green skip.

simpson = getattr(integrate, 'simpson', None) or integrate.simps

SRATE = 4096.0
DELTAT = 1.0 / SRATE
NPTS = 614                       # marginalization_time_grid(0.075, 1/4096)
N_BUFFER = 4096
UNDERFLOW_NATS = 745.0           # log(smallest positive float64 normal), roughly


def _kappa_buffer():
    """A band-limited, periodic-on-its-own-length kappa(t) buffer.

    Periodic so that whatever integer window the code gathers is a genuine
    segment of it; the test does not need to predict ``ifirst``.
    """
    ts = np.arange(N_BUFFER) * DELTAT
    ms = np.arange(1, 400)
    T = N_BUFFER * DELTAT
    c = np.exp(-2j * np.pi * ms * (N_BUFFER // 2) * DELTAT / T) / (1.0 + (ms / 120.0) ** 2)
    return np.exp(2j * np.pi * np.outer(ts, ms) / T) @ c


def _inputs(dists_Mpc):
    """Minimal inputs that drive the SHIPPED NoLoop function on the numpy backend.

    One detector, one (l,m) pair and zero U/V cross terms, so the self-term
    ``rho_sq`` vanishes and ``lnL_t`` is just the response-scaled ``Re kappa(t)``
    times ``distMpcRef/dist``.  Distance is therefore a clean per-row amplitude
    knob: it sets each extrinsic sample's peak ``lnL`` independently, which is
    exactly the axis this test needs to separate.
    """
    dists_Mpc = np.asarray(dists_Mpc, dtype=float)
    n = dists_Mpc.size
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.tref = 1000000000.0
    for name in ('phi', 'theta', 'phiref', 'incl', 'psi'):
        setattr(P, name, np.zeros(n))
    P.dist = dists_Mpc * 1e6 * lal.PC_SI
    det = 'H1'
    # The window sits well inside the buffer: the epoch offset sets ifirst, and a
    # window running off the front would be zero-extended rather than gathered.
    return (P, {det: np.asarray(_kappa_buffer(), dtype=complex)[None, :]},
            {det: np.array([[2, 2]])},
            {det: np.zeros((1, 1), dtype=complex)},
            {det: P.tref - 0.5})


def _shipped(tvals, args, **kw):
    P, rholms, lookupNK, ct, epochs = args
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rholms, ct, ct, epochs, Lmax=2, xpy=np, **kw)


def _per_row_reference(lnL_t, xp=np, simps_fn=None):
    """log int exp(lnL_t) dt, row by row, each row offset by its OWN maximum.

    Same closed domain, same grid and the same Simpson rule as the code; only the
    offset differs, which is the whole subject of the test.  Done one row at a time
    so no other row can reach this one's value.

    ``xp``/``simps_fn`` select WHICH backend and WHICH Simpson rule, because the two
    backends genuinely do not share one: on numpy the code integrates with scipy, on
    cupy with ``optimized_gpu_tools.simps`` -- an old scipy with ``even='avg'`` against
    modern scipy's Cartwright correction -- and for EVEN ``npts`` (production is 614)
    they differ, by an amount that depends on how sharply peaked the row is
    (issue #204).  So each backend is compared against its own rule.
    """
    simps_fn = simpson if simps_fn is None else simps_fn
    out = np.empty(lnL_t.shape[0], dtype=float)
    for i in range(lnL_t.shape[0]):
        row = lnL_t[i]
        m = xp.max(row)
        out[i] = float(m) + float(xp.log(simps_fn(xp.exp(row - m), dx=DELTAT)))
    return out


def tvals_grid():
    grid = fl.marginalization_time_grid(0.075, DELTAT)
    assert len(grid) == NPTS
    return grid


@pytest.fixture(scope='module')
def tvals():
    return tvals_grid()


def test_quiet_sample_stays_finite_beside_a_loud_one(tvals):
    """The regression, in the shape production actually runs.

    Two extrinsic samples whose peak ``lnL`` differ by far more than the float64
    underflow budget.  With a batch-wide offset the quiet row returns ``-inf``;
    with a per-row offset it returns its own, finite, correct value.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    args = _inputs(dists)

    lnL_t = np.asarray(_shipped(tvals, args, return_lnLt=True))
    assert lnL_t.shape == (2, NPTS)
    peaks = lnL_t.max(axis=-1)
    # The premise of the test: the rows really are separated by more than the
    # underflow budget, so a batch offset MUST kill the quiet one.  If the
    # harness ever stops producing that separation this assert says so, instead
    # of the test passing vacuously.
    assert peaks[0] - peaks[1] > 2.0 * UNDERFLOW_NATS, peaks

    lnL = np.asarray(_shipped(tvals, args))
    assert lnL.shape == (2,)
    assert np.all(np.isfinite(lnL)), lnL
    np.testing.assert_allclose(lnL, _per_row_reference(lnL_t), rtol=0, atol=1e-9)


def test_offset_is_per_row_even_when_the_batch_is_square(tvals):
    """``keepdims=True``, guarded where its absence is SILENT.

    ``npts_extrinsic == npts_time`` on purpose: that is the one shape in which a
    bare ``axis=-1`` maximum broadcasts along the wrong axis without raising.
    All rows but the first are identical by construction, so they must return
    identical values -- an offset that leaks across the sample axis does not.
    """
    dists = np.full(NPTS, fl.distMpcRef * 1.0)
    dists[0] = fl.distMpcRef / 80.0
    args = _inputs(dists)

    lnL_t = np.asarray(_shipped(tvals, args, return_lnLt=True))
    assert lnL_t.shape == (NPTS, NPTS)          # square, deliberately
    peaks = lnL_t.max(axis=-1)
    assert peaks[0] - peaks[1] > 2.0 * UNDERFLOW_NATS, peaks[:2]

    lnL = np.asarray(_shipped(tvals, args))
    assert lnL.shape == (NPTS,)
    assert np.all(np.isfinite(lnL)), lnL[~np.isfinite(lnL)]
    # Identical inputs -> identical outputs, whatever else is in the batch.
    assert np.all(lnL[1:] == lnL[1]), np.unique(lnL[1:]).size
    np.testing.assert_allclose(lnL, _per_row_reference(lnL_t), rtol=0, atol=1e-9)


def test_onset_is_the_underflow_budget_not_a_general_offset_error(tvals):
    """Below the underflow budget the two offsets agree; above it they cannot.

    This pins the MECHANISM rather than just the symptom.  A batch offset is
    harmless while every row is within ~745 nats of the batch peak -- the
    expression is offset-invariant apart from rounding -- so a separation just
    under the budget must still come out finite and correct, and a separation
    well over it must be the only thing that breaks.  A "fix" that changed the
    integral itself, rather than only its offset, would fail the first half.
    """
    # MEASURE the unit-distance peak rather than hardcoding it, so the harness
    # stays self-calibrating if the buffer or the response factor is ever retuned.
    unit_peak = float(np.asarray(
        _shipped(tvals, _inputs([fl.distMpcRef]), return_lnLt=True)).max())
    for gap_nats in (400.0, 700.0):
        # peak lnL scales as 1/dist, so this places row 0 gap_nats above row 1.
        dists = np.array([fl.distMpcRef * unit_peak / (gap_nats + unit_peak),
                          fl.distMpcRef * 1.0])
        args = _inputs(dists)
        lnL_t = np.asarray(_shipped(tvals, args, return_lnLt=True))
        peaks = lnL_t.max(axis=-1)
        gap = float(peaks[0] - peaks[1])
        assert 0.9 * gap_nats < gap < UNDERFLOW_NATS, gap
        lnL = np.asarray(_shipped(tvals, args))
        assert np.all(np.isfinite(lnL)), (gap, lnL)
        np.testing.assert_allclose(lnL, _per_row_reference(lnL_t), rtol=0, atol=1e-9)


def test_return_lnLt_is_still_verbatim_and_unshifted(tvals):
    """The early ``return_lnLt`` return must keep returning UNSHIFTED values.

    Downstream time resampling and the band-limited/peak-local quadratures all
    consume this array and apply their own offsets; subtracting anything here
    would silently rescale them.  Checked against the absolute value the physics
    fixes -- ``lnL_t = Re kappa(t) * distMpcRef/dist`` with ``rho_sq = 0`` -- so a
    shift of any size, per-row or batch, is visible.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    lnL_t = np.asarray(_shipped(tvals, _inputs(dists), return_lnLt=True))
    ratio = lnL_t[0] / lnL_t[1]
    np.testing.assert_allclose(ratio, 80.0, rtol=1e-10, atol=0)
    # And the quiet row is the same array a one-sample call produces.
    solo = np.asarray(_shipped(tvals, _inputs(dists[1:]), return_lnLt=True))
    np.testing.assert_allclose(lnL_t[1], solo[0], rtol=0, atol=0)


def test_gpu_offset_is_per_row_too():
    """The same guard on the backend the defect was MEASURED on.

    The path is selected by ``opts.gpu``, not by the device -- ``--force-xpy`` keeps it
    on with no cupy, and the issue reproduces on plain numpy that way -- so the tests
    above are the real regression.  This one exists because ``keepdims=True`` and the
    ``[..., 0]`` add-back are xpy API calls: untested GPU code is broken code.  It skips
    without cupy and is run by hand on a GPU node, the way this repo's other GPU legs
    are (measured on ldas-pcdev11, cupy 14.1.1, cuda 12.8 container).

    Compared against the GPU's OWN Simpson rule, not against the numpy answer.  The two
    rules differ for even ``npts`` and the difference is NOT a constant offset -- it is
    per row, set by how sharply peaked that row's integrand is: measured here
    0.405, 0.0089 and -4.3e-5 nats for peaks of 1694, 21 and 7 nats (issue #204).  That
    is a real, separate, already-known discrepancy and it is not this test's subject; a
    cross-backend equality assertion here would be asserting #204 is absent.
    """
    cupy = pytest.importorskip('cupy')
    from RIFT.likelihood import optimized_gpu_tools
    import copy

    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0, fl.distMpcRef * 3.0])
    args = _inputs(dists)
    P, rholms, lookupNK, ct, epochs = args

    Pg = copy.deepcopy(P)
    for name in ('phi', 'theta', 'phiref', 'incl', 'psi', 'dist'):
        setattr(Pg, name, cupy.asarray(np.asarray(getattr(P, name))))
    g_args = (Pg, {k: cupy.asarray(v) for k, v in rholms.items()}, lookupNK,
              {k: cupy.asarray(v) for k, v in ct.items()}, epochs)

    def _gpu(**kw):
        Q, R, LK, C, E = g_args
        return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals_grid(), Q, LK, R, C, C, E, Lmax=2, xpy=cupy, **kw)

    lnL_t = _gpu(return_lnLt=True)
    peaks = cupy.asnumpy(lnL_t.max(axis=-1))
    assert peaks[0] - peaks[-1] > 2.0 * UNDERFLOW_NATS, peaks

    lnL = cupy.asnumpy(cupy.asarray(_gpu()))
    assert np.all(np.isfinite(lnL)), lnL              # the regression, on the device
    np.testing.assert_allclose(
        lnL, _per_row_reference(lnL_t, xp=cupy, simps_fn=optimized_gpu_tools.simps),
        rtol=0, atol=1e-9)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
