"""
Regression test for the per-row offset in the vectorized time-marginalized
likelihood (``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``).

The function reduces an array ``lnL_t`` of shape ``(npts_extrinsic, npts_time)``
over its time axis.  The offset used to stabilize ``exp()`` must be taken
*per extrinsic sample*.  If the whole-array (batch) maximum is used instead,
any row whose own peak sits more than ~745 nats below the loudest row in the
batch underflows to zero across the entire time axis and the function returns
``-inf`` where the likelihood is finite.  See
https://github.com/oshaughnessy-junior/research-projects-RIT/issues/232 .

The invariant tested here is *batch independence*: the marginalized ``lnL`` of
an extrinsic sample must not depend on which other samples happen to share its
batch.  It is checked by comparing a batch evaluation against evaluating the
same samples one at a time through the same function (a batch of one is
trivially offset by its own maximum, so the singleton path is correct even on
the unpatched code).  This is a property of the shipped function, not of a
re-implementation of its kernel.

The tests deliberately use ``npts_extrinsic != npts_time`` so that an offset
array of the wrong shape (``axis=-1`` without ``keepdims=True``, i.e. shape
``(npts_extrinsic,)``) raises rather than broadcasting silently along the
time axis.
"""

import numpy as np

# Hard imports, not importorskip: lalsuite is a declared requirement, and a
# silently skipped regression test is not a regression test.
import lal

import RIFT.lalsimutils as lalsimutils
from RIFT.likelihood import factored_likelihood

DET = "H1"
LMS = np.array([[2, 2], [2, -2]], dtype=int)
NPTS_TIME = 65        # odd, so Simpson needs no even-sample special case
NPTS_FULL = 1024
DELTA_T = 1.0 / 4096
IFIRST_NOMINAL = 384  # keeps [ifirst, ifirst+npts) inside the buffer for any sky location
TREF = 1000000000.0


def _rholms_array():
    """A smooth, band-limited complex 'bump' timeseries for each (l,m) mode."""
    n = np.arange(NPTS_FULL)
    center = IFIRST_NOMINAL + NPTS_TIME // 2
    width = 12.0
    envelope = np.exp(-0.5 * ((n - center) / width) ** 2)
    phase = 2 * np.pi * n / 97.0
    out = np.empty((len(LMS), NPTS_FULL), dtype=np.complex128)
    out[0] = envelope * np.exp(1j * phase)
    out[1] = envelope * np.exp(-1j * phase) * 0.8
    return out


def _make_inputs(n_extrinsic, dist_mpc, cross_terms_scale=0.0):
    """Build the argument set for the vectorized likelihood.

    ``cross_terms_scale = 0`` zeroes the U and V cross terms, hence rho^2, so
    that lnL(t) is exactly linear in 1/distance.  That is what lets a test
    place rows a chosen number of nats apart.
    """
    rng = np.random.default_rng(20260902)

    P = lalsimutils.ChooseWaveformParams()
    P.deltaT = DELTA_T
    P.tref = TREF
    P.phi = rng.uniform(0.0, 2 * np.pi, n_extrinsic)          # RA
    P.theta = np.arcsin(rng.uniform(-1.0, 1.0, n_extrinsic))  # DEC
    P.phiref = rng.uniform(0.0, 2 * np.pi, n_extrinsic)
    P.incl = np.arccos(rng.uniform(-1.0, 1.0, n_extrinsic))
    P.psi = rng.uniform(0.0, np.pi, n_extrinsic)
    P.dist = np.asarray(dist_mpc, dtype=np.float64) * 1e6 * lal.PC_SI

    n_lms = len(LMS)
    U = np.zeros((n_lms, n_lms), dtype=np.complex128)
    V = np.zeros((n_lms, n_lms), dtype=np.complex128)
    if cross_terms_scale:
        U[:] = cross_terms_scale * np.eye(n_lms)
        V[:] = cross_terms_scale * 0.1 * np.eye(n_lms)

    tvals = np.arange(NPTS_TIME) * DELTA_T
    epoch = TREF - IFIRST_NOMINAL * DELTA_T

    return dict(
        tvals=tvals,
        P_vec=P,
        lookupNKDict={DET: LMS},
        rholmsArrayDict={DET: _rholms_array()},
        ctUArrayDict={DET: U},
        ctVArrayDict={DET: V},
        epochDict={DET: epoch},
    )


def _lnL(kwargs, **extra):
    return factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        kwargs["tvals"], kwargs["P_vec"], kwargs["lookupNKDict"],
        kwargs["rholmsArrayDict"], kwargs["ctUArrayDict"], kwargs["ctVArrayDict"],
        kwargs["epochDict"], Lmax=2, xpy=np, **extra
    )


def _select_row(kwargs, i):
    """A one-sample copy of ``kwargs``, sharing everything but the extrinsic row."""
    P = kwargs["P_vec"]
    P1 = lalsimutils.ChooseWaveformParams()
    P1.deltaT = P.deltaT
    P1.tref = P.tref
    for name in ("phi", "theta", "phiref", "incl", "psi", "dist"):
        setattr(P1, name, np.atleast_1d(getattr(P, name))[i:i + 1].copy())
    out = dict(kwargs)
    out["P_vec"] = P1
    # rholms are consumed in place by the exp(..., out=...) call: hand out fresh copies
    out["rholmsArrayDict"] = {k: v.copy() for k, v in kwargs["rholmsArrayDict"].items()}
    return out


def _singleton_reference(dist_mpc, cross_terms_scale=0.0):
    """Evaluate each extrinsic sample in its own batch of one."""
    n = len(dist_mpc)
    ref = np.empty(n, dtype=np.float64)
    for i in range(n):
        kw = _make_inputs(n, dist_mpc, cross_terms_scale)
        ref[i] = np.asarray(_lnL(_select_row(kw, i)))[0]
    return ref


def _distances_for_peaks(target_peaks):
    """Distances placing each sample's own peak lnL at the requested value.

    lnL(t) is linear in distMpcRef/dist when the cross terms are zero, so one
    unit-distance evaluation calibrates the whole ladder.
    """
    n = len(target_peaks)
    unit = np.full(n, float(factored_likelihood.distMpcRef))
    kw = _make_inputs(n, unit)
    lnL_t = np.asarray(_lnL(kw, return_lnLt=True))
    peak_unit = lnL_t.max(axis=-1)
    assert np.all(peak_unit > 0), "test fixture degenerate: unit-distance peaks not positive"
    return unit * peak_unit / np.asarray(target_peaks, dtype=np.float64)


def test_loud_batch_does_not_underflow_quiet_rows():
    """A batch spanning far more than the 745-nat float64 exp() budget.

    Row 0 peaks near 1500 nats (rho ~ 55); the rest sit near the prior bulk.
    With a batch-wide offset every row but the loudest underflows to -inf.
    """
    targets = np.array([1500.0, 900.0, 400.0, 5.0, 0.5])
    dist = _distances_for_peaks(targets)

    kw = _make_inputs(len(targets), dist)
    lnL = np.asarray(_lnL(kw))

    assert np.all(np.isfinite(lnL)), (
        "batch-max offset underflowed finite rows: lnL = {}".format(lnL))

    ref = _singleton_reference(dist)
    assert np.allclose(lnL, ref, rtol=0, atol=1e-6), (
        "batch result differs from one-at-a-time evaluation:\n"
        "  batch     = {}\n  singleton = {}\n  delta     = {}".format(
            lnL, ref, lnL - ref))


def test_well_resolved_batch_matches_singleton():
    """A batch where nothing underflows: the result must be unchanged.

    Per-row and batch offsets are not bit-identical -- they are different
    floating-point reductions.  Measured on this fixture (igwn python 3.11,
    numpy float64), batch-max vs per-row offsets agree to 1.8e-15 nats at
    peaks of 12..1 nats, 3.6e-15 nats at 40..2 nats, and 7.1e-14 nats at
    700..100 nats: ~1 ulp relative in every case.  The per-row batch result
    is exactly equal to the one-at-a-time result, because it is the same
    arithmetic on the same row, so the tolerance here is tight.
    """
    targets = np.array([12.0, 9.0, 6.0, 3.0, 1.0])
    dist = _distances_for_peaks(targets)

    kw = _make_inputs(len(targets), dist)
    lnL = np.asarray(_lnL(kw))
    ref = _singleton_reference(dist)

    assert np.all(np.isfinite(lnL))
    assert np.max(np.abs(lnL - ref)) < 1e-12, (
        "well-resolved batch/singleton disagreement {} exceeds rounding scale".format(
            np.max(np.abs(lnL - ref))))


def test_matches_singleton_with_nonzero_cross_terms():
    """Same invariant with rho^2 present, i.e. the full physical lnL."""
    dist = np.array([120.0, 200.0, 350.0, 600.0, 1000.0])

    kw = _make_inputs(len(dist), dist, cross_terms_scale=3.0)
    lnL = np.asarray(_lnL(kw))
    ref = _singleton_reference(dist, cross_terms_scale=3.0)

    assert np.all(np.isfinite(lnL))
    assert np.allclose(lnL, ref, rtol=0, atol=1e-6), (
        "batch = {}\nsingleton = {}".format(lnL, ref))


def test_single_sample_batch_is_finite():
    """The degenerate (1, npts_time) case: the offset must still be per row."""
    dist = _distances_for_peaks(np.array([1500.0]))
    kw = _make_inputs(1, dist)
    lnL = np.asarray(_lnL(kw))
    assert lnL.shape == (1,)
    assert np.all(np.isfinite(lnL))


def test_return_lnLt_is_unshifted():
    """The ``return_lnLt`` early return must hand back lnL(t) verbatim.

    Guards against 'fixing' the offset by moving the subtraction above the
    early return, which would silently shift every exported lnL(t).
    """
    targets = np.array([1500.0, 5.0, 0.5])
    dist = _distances_for_peaks(targets)

    kw = _make_inputs(len(targets), dist)
    lnL_t = np.asarray(_lnL(kw, return_lnLt=True))

    assert lnL_t.shape == (len(targets), NPTS_TIME)
    assert np.allclose(lnL_t.max(axis=-1), targets, rtol=1e-8), (
        "return_lnLt no longer returns unshifted lnL(t): peaks = {}".format(
            lnL_t.max(axis=-1)))
