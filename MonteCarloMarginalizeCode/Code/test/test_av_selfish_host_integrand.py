#!/usr/bin/env python
"""
The backend contract of mcsamplerAdaptiveVolume.update_sampling_prior_selfish.

Every integrand call site in RIFT feeds the target the DEVICE array and retries on a host
copy if the target refuses it, remembering the verdict in `self._integrand_wants_host`
(mcsamplerAdaptiveVolume.integrate_log, mcsamplerPortfolio.integrate_log in both its main
loop and its oracle block, mcsamplerNFlow.integrate_log).  update_sampling_prior_selfish --
how mcsamplerPortfolio drives a VARAHA/AV member -- was the exception: it evaluated the
device array with no try/except and no fallback.

So on any cupy-importable host a host-only integrand (a CI toy, a benchmark, a user's CPU
likelihood) driving a portfolio that contains an AV member died with

    TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()`.

and passed wherever cupy is absent, which is every CI runner.  Measured 2026-09-14: four
tests of test_l0_rescue_seed.py failed on ldas-pcdev2 (cupy 12.0.0) and in the Blackwell
container on ldas-pcdev11 (cupy 14.1.1); all 54 passed on ldas-grid, which has no cupy.

These tests run that path on a host WITHOUT cupy, using a stand-in for a device array:
numpy dispatches to it through __array_function__/__array_ufunc__ the way it dispatches to
cupy, it converts only through .get(), and it refuses implicit conversion with cupy's own
message.  Reverting either half of the fix fails them on a cupy-free runner.
"""

import types

import numpy as np
import pytest

import RIFT.integrators.mcsamplerAdaptiveVolume as mcsamplerAV

NAMES = ['right_ascension', 'declination', 'phi_orb', 'inclination', 'psi', 'distance']
NDIM = len(NAMES)

# Captured before any fixture can patch it.  Where cupy is genuinely present the tests run
# against the real thing and the stand-in below is not installed; where it is absent the
# stand-in puts the same data flow under test.  Either way the assertions are the same.
REAL_CUPY = bool(getattr(mcsamplerAV, 'cupy_ok', False))

_CUPY_MESSAGE = ('Implicit conversion to a NumPy array is not allowed. '
                 'Please use `.get()`.')


###
### the device stand-in
###

def _unwrap(x):
    if isinstance(x, _DeviceLike):
        return x._a
    if isinstance(x, tuple):
        return tuple(_unwrap(v) for v in x)
    if isinstance(x, list):
        return [_unwrap(v) for v in x]
    if isinstance(x, dict):
        return dict((k, _unwrap(v)) for k, v in x.items())
    return x


def _wrap(x):
    if isinstance(x, np.ndarray):
        return _DeviceLike(x)
    if isinstance(x, tuple):
        return tuple(_wrap(v) for v in x)
    return x           # scalars stay scalars, as cupy reductions effectively do here


class _DeviceLike(object):
    """A cupy array as far as this code path can tell.

    Three behaviours matter and all three are cupy's.  It REFUSES implicit numpy
    conversion with cupy's message, so a host-only integrand raises TypeError on it exactly
    as it does on the real thing.  It converts only through ``.get()`` (``identity_convert``
    is ``cupy.asnumpy`` on a GPU build, the identity otherwise).  And numpy DISPATCHES to it
    -- ``np.max``, ``np.sort``, ``np.append``, ``np.isfinite`` -- because the integrator and
    ``get_likelihood_threshold`` call bare numpy on arrays that are on the device in
    production, and that only works because cupy implements the same two protocols.
    """

    __array_priority__ = 100.0

    def __init__(self, a):
        # dtype is NOT forced: np.where returns integer index arrays and the comparisons
        # return booleans, and both are re-wrapped and then used as indices.
        self._a = np.asarray(a)

    def __array__(self, *args, **kwargs):
        raise TypeError(_CUPY_MESSAGE)

    def get(self):
        return self._a

    def __array_function__(self, func, types_, args, kwargs):
        return _wrap(func(*_unwrap(args), **_unwrap(kwargs)))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _wrap(getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs)))

    @property
    def shape(self):
        return self._a.shape

    @property
    def T(self):
        return _DeviceLike(self._a.T)

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        for row in self._a:
            yield _DeviceLike(row)

    def __getitem__(self, key):
        return _wrap(self._a[_unwrap(key)])

    def __repr__(self):
        return '_DeviceLike(shape={})'.format(self._a.shape)


def _binop(name):
    op = getattr(np.ndarray, name)

    def f(self, other):
        return _wrap(op(self._a, _unwrap(other)))
    f.__name__ = name
    return f


for _name in ('__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
              '__truediv__', '__rtruediv__', '__pow__', '__rpow__', '__neg__',
              '__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__'):
    if _name == '__neg__':
        setattr(_DeviceLike, _name, lambda self: _wrap(-self._a))
    else:
        setattr(_DeviceLike, _name, _binop(_name))


def _to_device(x):
    return x if isinstance(x, _DeviceLike) else _DeviceLike(x)


def _to_host(x):
    return x.get() if isinstance(x, _DeviceLike) else x


def _is_device(a):
    """True for a cupy array and for the stand-in; False for numpy."""
    return not isinstance(a, np.ndarray)


def _device(x):
    """Push to whatever the module currently calls the device (cupy, or the stand-in)."""
    return mcsamplerAV.identity_convert_togpu(x)


@pytest.fixture
def on_a_device(monkeypatch):
    """Make the sampler modules believe they have cupy, with the stand-in as its array type.

    This is the whole point of the suite: the defect is invisible where cupy is absent,
    because `identity_convert_togpu` is the identity there and the integrand never meets a
    device array.  Patching the backend globals of BOTH modules -- the AV member and the
    portfolio that drives it -- reproduces a GPU host's data flow on a cupy-free runner, CI
    included, including the portfolio learning its own `_integrand_wants_host` verdict.
    `self.xpy` stays numpy, which is what the dispatch protocols above are for.
    """
    if REAL_CUPY:
        return          # the real backend is already what we would be simulating
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    for mod in (mcsamplerAV, mcsamplerPF):
        monkeypatch.setattr(mod, 'cupy_ok', True)
        monkeypatch.setattr(mod, 'cupy', types.SimpleNamespace(ndarray=_DeviceLike),
                            raising=False)
        monkeypatch.setattr(mod, 'identity_convert_togpu', _to_device)
        monkeypatch.setattr(mod, 'identity_convert', _to_host)


###
### a host-only integrand, and a count of how often it was handed a device array
###

class _HostOnly(object):
    """6-D Gaussian that converts its arguments with a bare np.asarray, as a CPU likelihood
    written against numpy does.  Records every call and every DEVICE call, so a test can ask
    not just "did it survive" but "was the failing device call made at all"."""

    def __init__(self, rho=8.0):
        self.rho = rho
        self.n_calls = 0
        self.n_device_attempts = 0

    def __call__(self, *args, **kwargs):
        self.n_calls += 1
        if any(_is_device(a) for a in args):
            self.n_device_attempts += 1
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        w = (0.5 / self.rho) * np.ones(NDIM)
        out = 0.5 * self.rho ** 2 - 0.5 * np.sum(((x - 0.5) / w) ** 2, axis=-1)
        return np.where(out > 0.5 * self.rho ** 2 - 745.0, out, -np.inf)


def _av_member(n_chunk=2000):
    s = mcsamplerAV.MCSampler(n_chunk=n_chunk)
    s.xpy = mcsamplerAV.xpy_default          # numpy here; dispatch carries the stand-in
    for name in NAMES:
        s.add_parameter(name, pdf=None, left_limit=0.0, right_limit=1.0,
                        prior_pdf=lambda x: np.ones(np.shape(x)), adaptive_sampling=True)
    s.setup()
    return s


def _portfolio(n_chunk=2000):
    """AV + GMM portfolio built the way the ILE builds one (see test_l0_rescue_seed)."""
    import RIFT.integrators.mcsamplerPortfolio as mcsamplerPF
    import RIFT.integrators.mcsamplerEnsemble as mcsamplerEnsemble
    members = [mcsamplerAV.MCSampler(n_chunk=n_chunk), mcsamplerEnsemble.MCSampler()]
    s = mcsamplerPF.MCSampler(portfolio=members)
    # the ILE binds the sampler to the active backend this way (integrate_likelihood_
    # extrinsic_batchmode: sampler.xpy / sampler.identity_convert)
    s.identity_convert = mcsamplerPF.identity_convert
    s.identity_convert_togpu = mcsamplerPF.identity_convert_togpu
    pdf = np.vectorize(lambda x: 1.0)
    for name in NAMES:
        s.add_parameter(name, pdf, prior_pdf=pdf, left_limit=0.0, right_limit=1.0,
                        adaptive_sampling=True)
    s.setup()
    return s


###
### 0. the stand-in itself -- if this stops behaving like cupy the suite proves nothing
###

def test_the_stand_in_refuses_implicit_conversion_and_converts_through_get():
    d = _DeviceLike(np.arange(4.0))
    with pytest.raises(TypeError) as e:
        np.asarray(d)
    assert 'Please use `.get()`' in str(e.value)
    assert np.array_equal(d.get(), np.arange(4.0))


def test_numpy_dispatches_to_the_stand_in():
    """get_likelihood_threshold calls bare np.max/np.sort on arrays that live on the device
    in production; that works on cupy through these two protocols, so it must work here."""
    d = _DeviceLike(np.array([3.0, 1.0, 2.0]))
    assert np.max(d) == 3.0
    assert np.array_equal(np.sort(d).get(), np.array([1.0, 2.0, 3.0]))
    assert np.all(np.isfinite(d).get())


###
### 1. the fix: the selfish update must fall back to the host, and remember it
###

def test_the_selfish_update_survives_a_host_only_integrand(on_a_device):
    """THE DEFECT.  Reverting the try/except in update_sampling_prior_selfish makes this
    raise TypeError('Implicit conversion to a NumPy array is not allowed...'), which is
    verbatim the failure four test_l0_rescue_seed tests hit on every GPU host."""
    np.random.seed(20260914)
    s = _av_member()
    lnF = _HostOnly()

    s.update_sampling_prior_selfish(lnF)

    assert lnF.n_device_attempts == 1, 'expected exactly one device probe, then the host'
    assert s._integrand_wants_host is True
    assert s.V < 1.0, 'the live volume did not contract: the update did not do its work'


def test_the_host_verdict_is_remembered_across_steps(on_a_device):
    """Not just "does not crash": the fallback is a LEARNED verdict, so the failing device
    call is paid once per run, not once per chunk.  A bare try/except with no flag leaves
    this at one probe per step and passes the test above."""
    np.random.seed(20260914)
    s = _av_member()
    lnF = _HostOnly()

    for _ in range(3):
        s.update_sampling_prior_selfish(lnF)

    assert lnF.n_calls >= 3
    assert lnF.n_device_attempts == 1


def test_a_device_native_integrand_still_gets_device_arrays(on_a_device):
    """The other half of the contract, and the reason the retry is device-FIRST: the
    production ILE likelihood is device-native and fails on a host copy.  A target that
    accepts the device array must never be demoted to numpy."""
    np.random.seed(20260914)
    s = _av_member()
    seen = []

    def lnF(*args):
        seen.append(all(_is_device(a) for a in args))
        x = np.array([np.asarray(a.get(), dtype=float).ravel() for a in args]).T
        return _device(-0.5 * np.sum((x - 0.5) ** 2, axis=-1) * 400.0)

    s.update_sampling_prior_selfish(lnF)

    assert seen and all(seen), 'a device-native integrand was handed host arrays'
    assert getattr(s, '_integrand_wants_host', False) is False


def test_a_value_error_also_earns_the_host_fallback(on_a_device):
    """The retry catches TypeError AND ValueError.  cupy does not speak with one voice:
    a host array reaching a device routine surfaces as either, depending on where it is
    rejected.  Narrowing the except clause to TypeError alone passes every other test."""
    np.random.seed(20260914)
    s = _av_member()
    calls = []

    def lnF(*args):
        calls.append(_is_device(args[0]))
        if _is_device(args[0]):
            raise ValueError('object __array__ method not producing an array')
        x = np.array([np.asarray(a, dtype=float).ravel() for a in args]).T
        return -0.5 * np.sum((x - 0.5) ** 2, axis=-1) * 400.0

    s.update_sampling_prior_selfish(lnF)

    assert calls[:2] == [True, False], 'device probe then host retry expected, got %r' % calls
    assert s._integrand_wants_host is True


def test_an_unrelated_exception_is_not_swallowed_by_the_retry(on_a_device):
    """The fallback must not become a bare `except Exception`.  A likelihood that is simply
    broken -- a RuntimeError out of a waveform generator, a KeyError from a lookup -- has to
    reach the caller, not be silently re-run on the host and have its failure recorded as a
    backend preference.  Widening the except clause passes every other test in this file."""
    np.random.seed(20260914)
    s = _av_member()
    calls = []

    def lnF(*args):
        calls.append(1)
        raise RuntimeError('the waveform generator failed')

    with pytest.raises(RuntimeError):
        s.update_sampling_prior_selfish(lnF)

    assert len(calls) == 1, 'a non-backend failure was retried'
    assert getattr(s, '_integrand_wants_host', False) is False


@pytest.mark.skipif(REAL_CUPY, reason='there IS a device here; the single-call rule is the no-cupy one')
def test_without_cupy_a_failing_integrand_runs_exactly_once():
    """No fixture: this is the REAL cupy-free path, the one every CI runner takes.

    `identity_convert` is the identity there, so a retry would re-invoke the integrand with
    the byte-identical array it just refused.  The exception is the same either way, but the
    integrand runs twice: doubled side effects, and doubled RNG consumption for an ILE
    likelihood that marginalizes distance or calibration internally, which moves a seeded
    run.  Dropping the `not cupy_ok` short circuit makes this count 2.
    """
    np.random.seed(20260914)
    s = _av_member()
    calls = []

    def lnF(*args):
        calls.append(1)
        raise TypeError('a genuine bug in the user likelihood')

    with pytest.raises(TypeError):
        s.update_sampling_prior_selfish(lnF)

    assert len(calls) == 1, 'the integrand was invoked %d times on a host with no device' % len(calls)


###
### 2. the reported failure: a portfolio pass driving a VARAHA member
###

def test_a_portfolio_with_an_av_member_survives_a_host_only_integrand(on_a_device):
    """The shape that took down test_l0_rescue_seed on ldas-pcdev2 and ldas-pcdev11.  The
    portfolio's own evaluation already retried on the host; the member's did not, so the
    TypeError surfaced from inside the weight-update block."""
    np.random.seed(20260914)
    s = _portfolio(256)
    lnF = _HostOnly()

    s.integrate_log(lnF, *NAMES, nmax=512, neff=1, n=256,
                    no_protect_names=True, verbose=False, save_intg=True)

    member = s.portfolio_realizations[0]
    assert hasattr(member, 'is_varaha')
    assert member._integrand_wants_host is True


def test_a_wrong_propagated_verdict_self_corrects_instead_of_killing_the_member(on_a_device):
    """The risk the propagation introduces, and the guard that pays for it.

    A member can now be handed a verdict it never learned.  If the portfolio latched "host"
    on a transient failure of a device-native likelihood, the member would feed that
    likelihood a numpy array; cupy answers `TypeError: Unsupported type numpy.ndarray`.
    Before the guard that branch had no try/except, so a wrong verdict killed the point.
    Now it unlatches and goes back to the device.
    """
    np.random.seed(20260914)
    s = _av_member()
    s._integrand_wants_host = True          # as mcsamplerPortfolio would have set it
    seen = []

    def lnF(*args):
        on_device = all(_is_device(a) for a in args)
        seen.append(on_device)
        if not on_device:
            raise TypeError('Unsupported type numpy.ndarray')
        x = np.array([np.asarray(mcsamplerAV.identity_convert(a), dtype=float).ravel()
                      for a in args]).T
        return _device(-0.5 * np.sum((x - 0.5) ** 2, axis=-1) * 400.0)

    s.update_sampling_prior_selfish(lnF)

    assert seen[:2] == [False, True], 'expected a host attempt then a device retry, got %r' % seen
    assert s._integrand_wants_host is False, 'the wrong verdict was not unlatched'


def test_the_portfolio_hands_its_backend_verdict_to_the_member(on_a_device):
    """The propagation.  The portfolio learns the verdict on its OWN evaluation, which runs
    earlier in the same chunk, so the member need not rediscover it with a call already
    known to fail.  ONE device probe for the whole run -- the portfolio's -- is the
    assertion: reverting the two propagation lines in mcsamplerPortfolio makes the member
    probe too, on its first update, and the count is 2.

    Propagation only ever SETS the flag.  A portfolio verdict of "device is fine" is not
    distinguishable from "not learned yet" (both are the absent attribute), so there is
    nothing to mirror in that direction, and a member that learned "host" by another route
    must keep it.
    """
    np.random.seed(20260914)
    s = _portfolio(256)
    lnF = _HostOnly()

    s.integrate_log(lnF, *NAMES, nmax=512, neff=1, n=256,
                    no_protect_names=True, verbose=False, save_intg=True)

    assert lnF.n_calls > 2, 'the run was too short to include a member update'
    assert lnF.n_device_attempts == 1, 'the member re-discovered a verdict it was given'
    assert s._integrand_wants_host is True
    assert s.portfolio_realizations[0]._integrand_wants_host is True
