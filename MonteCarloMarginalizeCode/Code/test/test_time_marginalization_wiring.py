#!/usr/bin/env python3
"""Does --time-marginalization-quadrature / time_quadrature actually DO anything?

A flag that is accepted, documented, and silently inert is a known failure mode
in this codebase, and a test that only exercises
``time_marginalization_quadrature`` in isolation would not catch one.  These
tests drive the SHIPPED
``factored_likelihood.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop``.

They use synthetic rholm timeseries rather than a waveform: the likelihood only
ever sees <h_lm(t)|d> as an array, so a band-limited synthetic exercises exactly
the same code path in a second, with no LAL waveform generation and no data
files.  The reference is that SAME shipped function run on a 16x finer sampling
of the SAME continuous rholm -- so the comparison is against the production code
at a resolution where its own Simpson rule has converged, and does not go
through the FFT machinery under test.
"""

import numpy as np
import pytest

import RIFT.lalsimutils as lalsimutils
import RIFT.likelihood.factored_likelihood as fl
from RIFT.likelihood import time_marginalization_quadrature as tq

import lal

DETS = ["H1", "L1", "V1"]
#: Reference comparisons use a SINGLE detector.  Not a simplification -- a
#: necessity.  The gather is nearest-sample (ifirst = rint(tfirst/deltaT)), so
#: refining deltaT re-quantizes EACH detector's arrival time independently, by
#: up to deltaT/2.  Against a ~1 kHz carrier that is tens of degrees of relative
#: phase, so the three contributions add up differently at every resolution:
#: a "finer" run is then not a finer estimate of the same integral, it is a
#: different integrand, and a reference built that way does not converge (it
#: wandered by ~1 nat even at 256x oversampling).  With one detector there is no
#: relative phase to re-quantize and the reference converges cleanly.  The
#: nearest-sample gather is a SEPARATE known defect and is not what this change
#: addresses.
DETS_REF = ["H1"]
MODES = [(2, 2), (2, -2)]
EPOCH = 1000000014.0
WINDOW_HALF = 0.075
BASE_SRATE = 4096


ENV_SIGMA = 2.0e-3   # matched-filter envelope width, seconds


def _rholm_functions(seed=11, n_modes=5, f_max=1500.0, t_peak=0.0,
                     dets=None):
    """Band-limited <h_lm(t)|d> as closed-form functions of time.

    A carrier (real positive mode amplitudes, so every row peaks at `t_peak`)
    times a Gaussian envelope.  The envelope is what makes this a fair stand-in
    for a real rholm: the integrand must DIE at the window edges, or the test
    is dominated by where the window happens to be cut rather than by the
    quadrature -- and the coarse and fine samplings quantise the window start
    to their own deltaT, so their domains differ by up to one coarse sample.
    At ENV_SIGMA=2 ms against a +-75 ms window that contribution is e^-700.

    The envelope costs exact band-limitation in principle; in practice its
    bandwidth (~1/(2 pi ENV_SIGMA) ~ 80 Hz) added to f_max stays well inside
    the 2048 Hz Nyquist, which is the same approximation the real rholms make.
    """
    rng = np.random.default_rng(seed)
    out = {}
    out_dets = list(DETS if dets is None else dets)
    for det in out_dets:
        for lm in MODES:
            f = rng.uniform(0.25 * f_max, f_max, size=n_modes)
            a = rng.uniform(0.5, 1.5, size=n_modes)
            a = a / a.sum()
            ph = rng.uniform(0, 2 * np.pi)   # per-mode carrier phase

            def fn(t, f=f, a=a, ph=ph):
                t = np.asarray(t, dtype=float)
                dt = t.ravel() - t_peak
                z = (a[:, None] * np.exp(2j * np.pi * f[:, None]
                                         * dt[None, :])).sum(axis=0)
                z = z * np.exp(-0.5 * (dt / ENV_SIGMA) ** 2)
                return (z * np.exp(1j * ph)).reshape(t.shape)

            out[(det, lm)] = fn
    out["dets"] = out_dets
    return out


def _build(funcs, oversample=1, amp=400.0):
    """Pack the synthetic rholms into exactly the structures NoLoop consumes."""
    deltaT = 1.0 / (BASE_SRATE * oversample)
    npts_full = int(4 * WINDOW_HALF / deltaT)
    t = (np.arange(npts_full) - npts_full // 2) * deltaT
    lookupNK, rholm, ctU, ctV, epoch = {}, {}, {}, {}, {}
    dets = funcs["dets"]
    for det in dets:
        lookupNK[det] = np.array(MODES, dtype=int)
        rholm[det] = np.stack([amp * funcs[(det, lm)](t) for lm in MODES])
        # cross terms: Hermitian, positive definite, time-independent
        ctU[det] = np.eye(len(MODES), dtype=complex) * (amp / len(dets))
        ctV[det] = np.zeros((len(MODES), len(MODES)), dtype=complex)
        epoch[det] = EPOCH + t[0]
    return dict(lookupNKDict=lookupNK, rholmArrayDict=rholm, ctUArrayDict=ctU,
                ctVArrayDict=ctV, epochDict=epoch, deltaT=deltaT)


def _P_vec(deltaT, n=3):
    P = lalsimutils.ChooseWaveformParams()
    P.deltaT = deltaT
    P.tref = EPOCH
    P.radec = True
    P.phi = np.linspace(1.0, 1.4, n)
    P.theta = np.linspace(0.2, 0.4, n)
    P.psi = np.linspace(0.3, 0.9, n)
    P.incl = np.linspace(0.8, 1.3, n)
    P.phiref = np.linspace(0.0, 1.0, n)
    P.dist = np.full(n, 1000.0 * 1e6 * lal.PC_SI)
    return P


def _tvals(deltaT):
    n = int(2 * WINDOW_HALF / deltaT)
    return np.linspace(-WINDOW_HALF, WINDOW_HALF, n)


def _call(ctx, **kw):
    P = _P_vec(ctx["deltaT"])
    return np.atleast_1d(np.asarray(
        fl.DiscreteFactoredLogLikelihoodViaArrayVectorNoLoop(
            _tvals(ctx["deltaT"]), P, ctx["lookupNKDict"],
            ctx["rholmArrayDict"], ctx["ctUArrayDict"], ctx["ctVArrayDict"],
            ctx["epochDict"], Lmax=2, xpy=np, **kw)))



#: Oversampling for the reference: high enough that the SHIPPED Simpson rule
#: has itself converged (single detector, it is flat from 16x on).  Every
#: comparison below goes through `_reference`, whose second return value is its
#: OWN convergence estimate, so no test can be tighter than the truth it is
#: checked against.
REF_OVERSAMPLE = 32


def _reference(funcs, **kw):
    """(value, convergence_estimate) from the shipped Simpson path, refined.

    `funcs` must be single-detector -- see DETS_REF.
    """
    assert len(funcs["dets"]) == 1, "reference must be single-detector; see DETS_REF"
    fine = _call(_build(funcs, oversample=REF_OVERSAMPLE), **kw)
    coarser = _call(_build(funcs, oversample=REF_OVERSAMPLE // 2), **kw)
    return fine, np.abs(fine - coarser).max()


# --------------------------------------------------------------------------
def test_default_reproduces_the_historical_simpson_expression_exactly():
    """The default must be the OLD number, bit for bit.

    Recomputed here straight from the returned lnL(t) timeseries, so this is an
    independent statement of the historical formula rather than a re-run of the
    same branch.
    """
    from scipy import integrate
    simps = getattr(integrate, "simpson", None) or integrate.simps
    ctx = _build(_rholm_functions())
    got = _call(ctx)
    lnLt = _call(ctx, return_lnLt=True)
    m = lnLt.max()
    expect = m + np.log(simps(np.exp(lnLt - m), dx=ctx["deltaT"], axis=-1))
    assert np.array_equal(got, expect)


def test_the_flag_is_not_inert():
    """Flipping time_quadrature must change the returned lnL on a peaked case."""
    ctx = _build(_rholm_functions())
    default = _call(ctx)
    band = _call(ctx, time_quadrature="bandlimited")
    assert tq.last_report()["factor"] > 1
    assert np.all(np.abs(band - default) > 0.1), (default, band)


def test_bandlimited_matches_the_shipped_code_at_finer_sampling():
    """Reference is the production function itself, at a resolution where its
    own Simpson rule has converged -- not this module's FFT."""
    funcs = _rholm_functions(dets=DETS_REF)
    band = _call(_build(funcs), time_quadrature="bandlimited")
    ref, conv = _reference(funcs)
    assert np.abs(band - ref).max() < max(10 * conv, 1e-3), (band, ref, conv)
    # and the historical path is NOT this close, on the same comparison
    default = _call(_build(funcs))
    assert np.abs(default - ref).max() > 0.1, (default, ref)


def test_bandlimited_is_the_one_that_is_stable_under_grid_phase():
    """Slide the integrand under a fixed grid: the truth cannot move."""
    deltaT = 1.0 / BASE_SRATE
    d_default, d_band = [], []
    for shift in np.linspace(0.0, 2 * deltaT, 5):
        ctx = _build(_rholm_functions(t_peak=shift))
        d_default.append(_call(ctx))
        d_band.append(_call(ctx, time_quadrature="bandlimited"))
    span = lambda v: (np.max(v, axis=0) - np.min(v, axis=0)).max()
    assert span(d_default) > 0.5, span(d_default)
    assert span(d_band) < 1e-3, span(d_band)


@pytest.mark.parametrize("marg", [False, True])
def test_wiring_holds_under_phase_marginalization(marg):
    funcs = _rholm_functions(dets=DETS_REF)
    band = _call(_build(funcs), time_quadrature="bandlimited",
                 phase_marginalization=marg)
    ref, conv = _reference(funcs, phase_marginalization=marg)
    assert np.abs(band - ref).max() < max(10 * conv, 1e-3), (band, ref, conv)


def test_wiring_holds_for_a_nonlinear_loglikelihood_callback():
    """Production passes distmarg_loglikelihood, not the affine default.

    The callback is applied AFTER upsampling, so a nonlinear one is evaluated
    exactly rather than approximated -- and the resolution is derived from
    lnL(t), which already contains it.
    """
    nl = lambda k, r: np.log1p(np.exp(np.clip(k - 0.5 * r, -700, 700)))
    funcs = _rholm_functions(dets=DETS_REF)
    band = _call(_build(funcs), time_quadrature="bandlimited", loglikelihood=nl)
    ref, conv = _reference(funcs, loglikelihood=nl)
    assert np.abs(band - ref).max() < max(10 * conv, 1e-3), (band, ref, conv)


def test_return_lnLt_is_unaffected_by_the_quadrature_choice():
    """The exported lnL(t) timeseries is the COARSE one either way.

    Documented, not incidental: --srate-resample-time-marginalization is a
    separate mechanism on that path and this change deliberately leaves it
    alone.
    """
    ctx = _build(_rholm_functions())
    a = _call(ctx, return_lnLt=True)
    b = _call(ctx, return_lnLt=True, time_quadrature="bandlimited")
    assert np.array_equal(a, b)


def test_an_unknown_quadrature_is_rejected_loudly():
    ctx = _build(_rholm_functions())
    with pytest.raises(ValueError):
        _call(ctx, time_quadrature="trapezoid")


def test_scope_is_the_production_path_only():
    """NoLoopOrig and the non-NoLoop vector path keep Simpson, deliberately.

    Adding an untested second copy of a numerical change is how one instance
    becomes three different behaviours.
    """
    import inspect
    for name in ("DiscreteFactoredLogLikelihoodViaArrayVectorNoLoopOrig",
                 "DiscreteFactoredLogLikelihoodViaArrayVector",
                 "DiscreteFactoredLogLikelihoodViaArray"):
        sig = inspect.signature(getattr(fl, name))
        assert "time_quadrature" not in sig.parameters, name
