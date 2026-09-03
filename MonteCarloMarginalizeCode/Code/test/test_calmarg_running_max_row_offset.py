#!/usr/bin/env python
"""The in-loop calibration-marginalization log-sum-exp offset must be PER EXTRINSIC
SAMPLE, not per batch.

WHAT IS BEING TESTED, AND AGAINST WHAT
--------------------------------------
With ``n_cal > 1`` and ``cal_method='loop'`` (the DEFAULT calmarg reduction -- the fused
kernel is opt-in behind ``--calibration-fused-kernel``),
``DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop`` marginalizes over calibration
realizations with a streaming log-sum-exp::

    m_c = max(lnL_t_c)                  # over the WHOLE (npts_extrinsic, npts_time) block
    running_max = max(running_max, m_c)
    S          += exp(lnL_t_c - running_max)
    lnL         = running_max + log(simps(S, dx=deltaT)) - log(n_cal)

``running_max`` was a SCALAR shared by every extrinsic sample and every realization, so
every row was shifted by the LOUDEST row's peak.  Any row sitting more than ~745 nats
below it underflows ``exp()`` to 0 across its whole time axis and across every
realization: its ``S`` row is 0, ``log(0) = -inf``, and the likelihood comes back
``-inf`` at a sample where it is finite and perfectly ordinary.  With ``lnL ~ rho^2/2``
at the peak and ``lnL ~ 0`` for a typical prior draw, that fires once ``max lnL > ~745``
(``rho ~ 40``) and then applies to the BULK of the prior, not a tail -- which is what
collapses ``mcsamplerAV`` on loud events.  See
oshaughnessy-junior/research-projects-RIT#232 for the real-data measurement of the same
defect on the ``n_cal == 1`` leg of this function, and #234 for that fix.

THE REFERENCE IS THE SHIPPED CODE'S OWN ALREADY-CORRECT BRANCH.  Five lines above the
site under test, the ``return_cal_components`` branch computes the same per-realization
time integral with ``m_raw = max(lnL_t_c, axis=-1, keepdims=True)`` -- per row, correctly
-- and returns it RAW (no importance weight).  Each test below asks the shipped function
for those components and combines them by hand,

    lnL_ref = logsumexp_c( components[:, c] + cal_log_w[c] ) - log(n_cal),

which is algebraically the identical quantity.  Nothing is reimplemented: the same
detector response, the same Q window, the same loglikelihood callback and the same
Simpson rule produce both sides.  The ONLY thing that differs is the offset, which is the
whole subject of the test.

``keepdims=True`` is load-bearing and is guarded separately.  With a bare ``axis=-1``
the ``(n,)`` maximum broadcasts along the TIME axis instead of the sample axis.  That
RAISES when ``npts_extrinsic != npts_time`` -- and is silently wrong when they are equal,
which is why one case below is deliberately square.  The add-back
``running_max[..., 0]`` is guarded by the same cases: keeping the axis there broadcasts
the (n, 1) offset against the (n,) time integral into an (n, n) result.

    OMP_NUM_THREADS=1 PYTHONPATH=<worktree>/MonteCarloMarginalizeCode/Code \
      python -m pytest -q MonteCarloMarginalizeCode/Code/test/test_calmarg_running_max_row_offset.py
"""
from __future__ import print_function, division

import os

os.environ.setdefault("RIFT_LOWLATENCY", "1")

import numpy as np
import pytest
from scipy.special import logsumexp

import lal
import RIFT.lalsimutils as lsu
from RIFT.likelihood import factored_likelihood as fl

# lal / lalsimutils are imported at module scope on purpose, NOT via importorskip:
# lalsuite is in requirements.txt and both CI jobs that run this file install it, so a
# missing lal here is a broken job, not an unsupported platform -- and an importorskip
# would turn that into a green skip.

SRATE = 4096.0
DELTAT = 1.0 / SRATE
NPTS = 614                       # len(marginalization_time_grid(0.075, 1/4096))
N_CAL = 2
N_WINDOW = 4096                  # per-realization block length; the gathered window
                                 # (ifirst ~ 1894, npts 614) sits well inside one block
UNDERFLOW_NATS = 745.0           # -log(smallest positive float64 normal), roughly


def _kappa_buffer():
    """A band-limited, periodic-on-its-own-length kappa(t) block.

    Periodic so that whatever integer window the code gathers is a genuine segment of
    it; the test does not need to predict ``ifirst``.
    """
    ts = np.arange(N_WINDOW) * DELTAT
    ms = np.arange(1, 400)
    T = N_WINDOW * DELTAT
    c = np.exp(-2j * np.pi * ms * (N_WINDOW // 2) * DELTAT / T) / (1.0 + (ms / 120.0) ** 2)
    return np.exp(2j * np.pi * np.outer(ts, ms) / T) @ c


_BASE_KAPPA = _kappa_buffer()
# Per-realization amplitudes.  Deliberately unequal, so the n_cal reduction is doing
# real work: the two realizations' time integrals differ by ~500 nats at the loud row,
# which is what makes the streaming rescale branch (S *= exp(...)) execute.
CAL_AMPS = (1.0, 0.7)


def _inputs(dists_Mpc):
    """Minimal inputs that drive the SHIPPED NoLoop function on the numpy backend.

    One detector, one (l,m) pair and zero U/V cross terms, so the self-term ``rho_sq``
    vanishes and ``lnL_t`` is just the response-scaled ``Re kappa(t)`` times
    ``distMpcRef/dist``.  Distance is therefore a clean per-row amplitude knob: it sets
    each extrinsic sample's peak ``lnL`` independently, which is exactly the axis this
    test needs to separate.  The rholm buffer holds ``N_CAL`` CONTIGUOUS blocks, which
    is the layout the calmarg path assumes (realization c is selected by shifting the
    window into block c).
    """
    dists_Mpc = np.asarray(dists_Mpc, dtype=float)
    n = dists_Mpc.size
    P = lsu.ChooseWaveformParams()
    P.deltaT = DELTAT
    P.tref = 1000000000.0
    for name in ('phi', 'theta', 'phiref', 'incl', 'psi'):
        setattr(P, name, np.zeros(n))
    P.dist = dists_Mpc * 1e6 * lal.PC_SI
    blocks = np.concatenate([_BASE_KAPPA * a for a in CAL_AMPS[:N_CAL]])
    det = 'H1'
    return (P, {det: np.asarray(blocks, dtype=complex)[None, :]},
            {det: np.array([[2, 2]])},
            {det: np.zeros((1, 1), dtype=complex)},
            {det: P.tref - 0.5})


def _shipped(tvals, args, **kw):
    P, rholms, lookupNK, ct, epochs = args
    kw.setdefault('n_cal', N_CAL)
    kw.setdefault('cal_method', 'loop')
    return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
        tvals, P, lookupNK, rholms, ct, ct, epochs, Lmax=2, xpy=np, **kw)


def _per_row_reference(tvals, args, cal_log_weights=None, **kw):
    """The cal-marginalized lnL built from the shipped function's OWN per-realization
    components, which are already offset per row.

        lnL = log( (1/n_cal) sum_c exp(log_w_c) * int dt exp(lnL_t,c) )

    ``return_cal_components`` returns log(int dt exp(lnL_t,c)) RAW -- before the
    importance log-weight -- so the weights are folded in here, exactly as the loop
    reduction folds them into ``lnL_t_c`` before its own log-sum-exp.
    """
    comps = np.asarray(_shipped(tvals, args, return_cal_components=True, **kw), dtype=float)
    assert comps.shape[-1] == N_CAL
    log_w = np.zeros(N_CAL) if cal_log_weights is None else np.asarray(cal_log_weights, dtype=float)
    return logsumexp(comps + log_w[None, :], axis=-1) - np.log(N_CAL)


@pytest.fixture(scope='module')
def tvals():
    grid = fl.marginalization_time_grid(0.075, DELTAT)
    assert len(grid) == NPTS
    return grid


def test_quiet_sample_stays_finite_beside_a_loud_one(tvals):
    """The regression, in the shape production actually runs.

    Two extrinsic samples whose peak ``lnL`` differ by far more than the float64
    underflow budget.  With a batch-wide ``running_max`` the quiet row returns ``-inf``;
    with a per-row offset it returns its own, finite, correct value.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    args = _inputs(dists)

    comps = np.asarray(_shipped(tvals, args, return_cal_components=True), dtype=float)
    assert comps.shape == (2, N_CAL)
    # The premise of the test: the rows really are separated by more than the underflow
    # budget, so a batch-wide offset MUST kill the quiet one.  If the harness ever stops
    # producing that separation this assert says so, instead of passing vacuously.
    assert comps[0].max() - comps[1].max() > 2.0 * UNDERFLOW_NATS, comps

    lnL = np.asarray(_shipped(tvals, args))
    assert lnL.shape == (2,), lnL.shape        # (2, 2) means the add-back kept its axis
    assert np.all(np.isfinite(lnL)), lnL
    ref = _per_row_reference(tvals, args)
    np.testing.assert_allclose(lnL, ref, rtol=0, atol=1e-9)


def test_offset_is_per_row_even_when_the_batch_is_square(tvals):
    """``keepdims=True``, guarded where its absence is SILENT.

    ``npts_extrinsic == npts_time`` on purpose: that is the one shape in which a bare
    ``axis=-1`` maximum broadcasts along the wrong axis without raising.  All rows but
    the first are identical by construction, so they must return identical values -- an
    offset that leaks across the sample axis does not.
    """
    dists = np.full(NPTS, fl.distMpcRef * 1.0)
    dists[0] = fl.distMpcRef / 80.0
    args = _inputs(dists)

    comps = np.asarray(_shipped(tvals, args, return_cal_components=True), dtype=float)
    assert comps.shape == (NPTS, N_CAL)        # square in (npts_extrinsic, npts_time)
    assert comps[0].max() - comps[1].max() > 2.0 * UNDERFLOW_NATS, comps[:2]

    lnL = np.asarray(_shipped(tvals, args))
    assert lnL.shape == (NPTS,), lnL.shape
    assert np.all(np.isfinite(lnL)), lnL[~np.isfinite(lnL)]
    # Identical inputs -> identical outputs, whatever else is in the batch.
    assert np.all(lnL[1:] == lnL[1]), np.unique(lnL[1:]).size
    np.testing.assert_allclose(lnL, _per_row_reference(tvals, args), rtol=0, atol=1e-9)


def test_return_lnLt_timeseries_is_offset_per_row_too(tvals):
    """The ``return_lnLt`` calmarg branch shares ``running_max`` and the same defect.

    That branch returns the cal-marginalized ``lnL(t)``; the driver resamples it to draw
    an event time (``--time-marginalization`` with the resampling output).  A quiet row
    came back ``-inf`` at EVERY time bin, so the row carried no time information at all.
    The check is the branch's own identity with the scalar return:
    ``log int dt exp(lnL_t) == lnL``, taken row by row.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    args = _inputs(dists)

    lnLt = np.asarray(_shipped(tvals, args, return_lnLt=True), dtype=float)
    assert lnLt.shape == (2, NPTS), lnLt.shape
    # The loud row legitimately underflows FAR from its peak (a 1685-nat peak in a
    # 614-bin window); what must never happen is a row with no finite bin at all.
    assert np.any(np.isfinite(lnLt[1])), "quiet row is -inf at every time bin"

    lnL = np.asarray(_shipped(tvals, args))
    for i in range(2):
        m = lnLt[i].max()
        got = m + np.log(fl.my_simps(np.exp(lnLt[i] - m), dx=DELTAT))
        assert abs(got - lnL[i]) < 1e-9, (i, got, lnL[i])


def test_nonuniform_cal_weights_are_carried_through(tvals):
    """Importance-weighted cal draws (``--calibration-proposal-breadcrumb``) too.

    The weight is folded into ``lnL_t_c`` BEFORE the offset is taken, so a per-row
    offset must be taken after the fold.  A weight large enough to reorder which
    realization dominates makes the ordering observable.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    args = _inputs(dists)
    log_w = np.array([-3.0, +3.0])             # mean-1 weights are not required here

    lnL = np.asarray(_shipped(tvals, args, cal_log_weights=log_w))
    assert lnL.shape == (2,), lnL.shape
    assert np.all(np.isfinite(lnL)), lnL
    np.testing.assert_allclose(
        lnL, _per_row_reference(tvals, args, cal_log_weights=log_w), rtol=0, atol=1e-9)


def test_a_row_that_is_minus_inf_everywhere_stays_minus_inf_not_nan(tvals):
    """A per-row offset must not turn an empty row into ``nan``.

    A scalar offset was shielded from this by any other finite row in the batch:
    ``exp(-inf - finite) = 0``.  A per-row offset is not -- ``exp(-inf - -inf) = nan``
    -- and a nan in ``S`` is permanent, so the guard ships with the per-row offset.
    A distance-marginalization callback that rejects a whole row (out-of-table distance)
    is the shipped way to produce one.
    """
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    args = _inputs(dists)

    def reject_first_row(kappa_sq, rho_sq):
        out = fl._factored_lnL_helper(kappa_sq, rho_sq)
        out = np.array(out, dtype=float, copy=True)
        out[0, :] = -np.inf
        return out

    lnL = np.asarray(_shipped(tvals, args, loglikelihood=reject_first_row))
    assert lnL.shape == (2,), lnL.shape
    assert not np.any(np.isnan(lnL)), lnL
    assert lnL[0] == -np.inf, lnL
    assert np.isfinite(lnL[1]), lnL


def test_onset_is_the_underflow_budget_not_a_general_offset_error(tvals):
    """Below the underflow budget the two offsets agree; above it they cannot.

    This pins the MECHANISM rather than the symptom.  A batch-wide offset is harmless
    while every row is within ~745 nats of the batch peak -- the expression is
    offset-invariant apart from rounding -- so a separation well under the budget must
    still come out finite and correct, and only a separation over it may break.  A
    "fix" that changed the integral itself rather than only its offset would fail this,
    and it PASSES on the unpatched code by design.

    The separations are MEASURED off the shipped components rather than prescribed, so
    the case stays meaningful if the harness buffer or the response factor is retuned:
    the loop skips any ladder rung that has drifted over the budget, and the test then
    checks that at least one rung within a few hundred nats of it survived.
    """
    checked = []
    for scale in (16.0, 24.0, 32.0):
        dists = np.array([fl.distMpcRef / scale, fl.distMpcRef * 1.0])
        args = _inputs(dists)
        comps = np.asarray(_shipped(tvals, args, return_cal_components=True), dtype=float)
        sep = float(comps[0].max() - comps[1].max())
        if sep >= UNDERFLOW_NATS:                 # over budget: not this test's subject
            continue
        checked.append(sep)
        lnL = np.asarray(_shipped(tvals, args))
        assert np.all(np.isfinite(lnL)), (sep, lnL)
        np.testing.assert_allclose(
            lnL, _per_row_reference(tvals, args), rtol=0, atol=1e-9)
    # Non-vacuity: a ladder that only ever reached a 10-nat separation would prove
    # nothing about a batch-wide offset, which is harmless at 10 nats.
    assert checked and max(checked) > 0.5 * UNDERFLOW_NATS, checked


@pytest.mark.skipif(fl.xpy_default is np, reason="no cupy on this host")
def test_gpu_offset_is_per_row_too(tvals):
    """Same guard on the GPU backend.  The defect is device-independent -- the path is
    selected by ``opts.gpu``, not by the device -- but the calmarg loop runs different
    kernels there (``_q_inner_product_gpu``), so the reduction is checked on both.
    """
    import cupy
    dists = np.array([fl.distMpcRef / 80.0, fl.distMpcRef * 1.0])
    P, rholms, lookupNK, ct, epochs = _inputs(dists)
    rholms_g = {k: cupy.asarray(v) for k, v in rholms.items()}
    ct_g = {k: cupy.asarray(v) for k, v in ct.items()}
    # The extrinsic arrays must be on the device, as the driver puts them
    # (integrate_likelihood_extrinsic_batchmode: ``P.phi = xpy_default.asarray(...)``);
    # a host P_vec reaches a cupy elementwise kernel and raises
    # "TypeError: Unsupported type <class 'numpy.ndarray'>" inside
    # SphericalHarmonicsVectorized, which is a harness error, not a likelihood one.
    Pg = P.manual_copy()
    for attr in ('phi', 'theta', 'psi', 'incl', 'phiref', 'dist'):
        Pg.__dict__[attr] = cupy.asarray(np.asarray(getattr(P, attr), dtype=np.float64))
    Pg.tref = float(P.tref)
    Pg.deltaT = float(P.deltaT)
    tvals_g = cupy.asarray(tvals)

    def _call(**kw):
        return fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            tvals_g, Pg, lookupNK, rholms_g, ct_g, ct_g, epochs, Lmax=2, xpy=cupy,
            n_cal=N_CAL, cal_method='loop', **kw)

    lnL = cupy.asnumpy(_call())
    assert lnL.shape == (2,), lnL.shape
    assert np.all(np.isfinite(lnL)), lnL
    comps = cupy.asnumpy(_call(return_cal_components=True))
    ref = logsumexp(comps, axis=-1) - np.log(N_CAL)
    np.testing.assert_allclose(lnL, ref, rtol=0, atol=1e-7)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
