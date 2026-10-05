#!/usr/bin/env python
"""RIFT's own GMM must pick its array backend from the ARRAYS IT WAS GIVEN.

`gaussian_mixture_model.gmm.__init__` set `self.xpy = xpy_default` unconditionally and
took no argument to override it, so on any host where cupy imports the class became
device-backed regardless of what the caller was using.  `_normalize` then allocated

    out = self.xpy.empty((n, d))        # on the DEVICE

and wrote host `samples` into it, which cupy refuses:

    ValueError: non-scalar numpy.ndarray cannot be used for fill

so `fit()`, `update()` and `score()` all rejected plain numpy -- the ordinary way to call
this class.  Measured at base 167365289 on the four suites that reach this module:

    test/integrators/test_portfolio_restrict_and_warm.py     22
    test/integrators/test_gmm_adaptive.py                     6
    test/integrators/test_portfolio_gmm_member_trains.py      2
    test/test_gmm_truncated_score.py                          2   = 32 tests

27 passed / 5 failed on ldas-pcdev2 (RTX 3080 cc86), the same 27/5 on ldas-pcdev13 (RTX
2080 Ti cc75, its slot 3), both CVMFS cupy 12.0.0; 32 passed / 0 failed on ldas-grid,
which has no cupy, so CI never saw it.  Same family as junior PR #342 (mcsamplerAdaptive
Volume.update_sampling_prior_selfish): a module-global backend decision that ignores the
caller.

The GMM's fix is inference rather than PR #342's device-first-then-retry.  A retry is the
right shape when the unknown is somebody else's callable -- you cannot ask a likelihood
which backend it wants, you can only try.  Here the backend is written on the argument:
`fit(X)` is told by X, and guessing wrong then catching the exception would be a slower
way to read the same fact.  Inference also fixes the half a retry cannot reach, which is
that a host caller was getting a DEVICE array back out of `score()`.

These tests run the device path on a host WITHOUT cupy, standing in for the device with
an array class numpy dispatches to through __array_function__/__array_ufunc__ -- a raising
__array__ alone is not enough -- and an array module that allocates it.  Where cupy is
genuinely present the same tests run against the real thing.  Either way the assertions
are the same.

Eighteen mutations were revert-checked against this suite on a cupy-free runner, with an
unmutated control through the same scoring path; 17 fail it.  The one that does not
replaces `fit_gmm_adaptive`'s `xpy = _xpy_for(sample_array)` with `xpy = xpy_default`,
and it survives because every value that line feeds -- the Kish N_eff, the BIC weights,
the default log_sample_weights -- is converted back to the samples' backend before it is
used.  If you are adding to fit_gmm_adaptive, that conversion is what you are relying on.
"""

import types

import numpy as np
import pytest

from RIFT.integrators import gaussian_mixture_model as GMM

# Captured before any fixture can patch it.
REAL_CUPY = bool(getattr(GMM, 'cupy_ok', False))

_FILL_MESSAGE = 'non-scalar numpy.ndarray cannot be used for fill'
_CONVERT_MESSAGE = ('Implicit conversion to a NumPy array is not allowed. '
                    'Please use `.get()`.')
_MIXED_MESSAGE = "Unsupported type <class 'numpy.ndarray'>"


###
### the device stand-in: an array, and the module that allocates it
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
    # numpy SCALARS are wrapped too, as 0-d device arrays.  cupy reductions and element
    # indexing return 0-d cupy.ndarray, not python floats, so leaving them bare made
    # `_xpy_for(result)` take the opposite branch under the stand-in from the one it takes
    # on a GPU -- and let two real defects through, where an unconverted device `bounds` or
    # `weights` decayed to host scalars on indexing and never mixed.
    if isinstance(x, (np.ndarray, np.generic)):
        return _DeviceLike(x)
    if isinstance(x, tuple):
        return tuple(_wrap(v) for v in x)
    return x


def _reject_mixed(args):
    """cupy refuses to compute on a HOST array alongside a device one.

    This is the behaviour the whole defect class turns on, so the stand-in has to have it
    too: without it, a backend mistake inside the library still produces a plausible
    number here and the suite reports a pass that a GPU host would not.  cupy does accept
    python and numpy SCALARS (and 0-d arrays) as operands, so only ndim>0 host arrays are
    refused, which is cupy's own rule."""
    stack = list(args)
    while stack:
        a = stack.pop()
        if isinstance(a, (tuple, list)):
            stack.extend(a)
        elif isinstance(a, dict):
            stack.extend(a.values())
        elif isinstance(a, np.ndarray) and a.ndim > 0:
            raise TypeError(_MIXED_MESSAGE)


class _DeviceLike(object):
    """A cupy array as far as this module can tell.

    Four behaviours matter and all four are cupy's.  It REFUSES implicit numpy conversion
    with cupy's message.  It converts only through ``.get()``.  numpy DISPATCHES to it
    through the two array protocols, which is why bare ``np.linalg.inv`` and
    ``scipy``-free numpy calls elsewhere in the module keep working on device arrays in
    production.  And -- the behaviour this suite turns on -- assigning a non-scalar HOST
    array into a slice of it raises ValueError, because cupy routes that through ``fill``.
    """

    __array_priority__ = 100.0

    def __init__(self, a):
        # dtype is NOT forced: np.where returns integer index arrays and comparisons
        # return booleans, and both get re-wrapped and used as indices.
        self._a = np.asarray(a)

    def __array__(self, *args, **kwargs):
        raise TypeError(_CONVERT_MESSAGE)

    def get(self):
        return self._a

    def __array_function__(self, func, types_, args, kwargs):
        _reject_mixed(args)
        return _wrap(func(*_unwrap(args), **_unwrap(kwargs)))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _reject_mixed(inputs)
        out = getattr(ufunc, method)(*_unwrap(inputs), **_unwrap(kwargs))
        return _wrap(out)

    # --- the defect's surface -------------------------------------------------
    def __setitem__(self, key, value):
        if isinstance(value, np.ndarray) and value.ndim > 0:
            # cupy._core.core._ndarray_base.fill, verbatim
            raise ValueError(_FILL_MESSAGE)
        self._a[_unwrap(key)] = _unwrap(value)

    def __getitem__(self, key):
        return _wrap(self._a[_unwrap(key)])

    # --- enough ndarray surface for the EM to run -----------------------------
    @property
    def shape(self):
        return self._a.shape

    @property
    def dtype(self):
        return self._a.dtype

    @property
    def ndim(self):
        return self._a.ndim

    @property
    def T(self):
        return _DeviceLike(self._a.T)

    def copy(self):
        return _DeviceLike(self._a.copy())

    def flatten(self):
        return _DeviceLike(self._a.flatten())

    def reshape(self, *a, **kw):
        return _wrap(self._a.reshape(*_unwrap(a), **_unwrap(kw)))

    def astype(self, *a, **kw):
        return _wrap(self._a.astype(*_unwrap(a), **_unwrap(kw)))

    def sum(self, *a, **kw):
        return _wrap(self._a.sum(*_unwrap(a), **_unwrap(kw)))

    def __len__(self):
        return len(self._a)

    def __iter__(self):
        for row in self._a:
            yield _wrap(row)     # rows of a 1-d array are 0-d DEVICE arrays, as in cupy

    def __bool__(self):
        return bool(self._a)

    __nonzero__ = __bool__

    def __float__(self):
        return float(self._a)

    def __repr__(self):
        return '_DeviceLike(shape={})'.format(self._a.shape)


def _binop(name):
    op = getattr(np.ndarray, name)

    def f(self, other):
        _reject_mixed((other,))
        return _wrap(op(self._a, _unwrap(other)))
    f.__name__ = name
    return f


for _name in ('__add__', '__radd__', '__sub__', '__rsub__', '__mul__', '__rmul__',
              '__truediv__', '__rtruediv__', '__floordiv__', '__mod__',
              '__pow__', '__rpow__', '__matmul__', '__rmatmul__',
              '__and__', '__or__', '__xor__',
              '__lt__', '__le__', '__gt__', '__ge__', '__eq__', '__ne__'):
    setattr(_DeviceLike, _name, _binop(_name))

# unary: 0-d device arrays support all of these in cupy, and the EM loop uses abs()
for _name in ('__neg__', '__pos__', '__invert__', '__abs__'):
    setattr(_DeviceLike, _name,
            (lambda op: lambda self: _wrap(op(self._a)))(getattr(np.ndarray, _name)))


class _DeviceXpy(object):
    """The array module of that device: numpy's API, allocating `_DeviceLike`.

    This is the global the fix reads -- `gmm.__init__` took `self.xpy` straight from
    `xpy_default` -- so patching it is what puts a GPU host's data flow under a cupy-free
    runner.  Every callable forwards to numpy and re-wraps; constants (`pi`, `inf`,
    `newaxis`) pass through untouched, because `newaxis` in particular must stay None.
    """

    ndarray = _DeviceLike

    def __init__(self, mod=np):
        self._mod = mod

    # Entry points that CREATE a device array from host data.  cupy allows exactly these
    # to take a host argument; everything else refuses one.
    _ALLOCATORS = frozenset(('array', 'asarray', 'asanyarray', 'ascontiguousarray',
                             'empty', 'zeros', 'ones', 'full', 'eye', 'identity',
                             'arange', 'linspace', 'empty_like', 'zeros_like',
                             'ones_like', 'full_like'))

    def __getattr__(self, name):
        obj = getattr(self._mod, name)
        if isinstance(obj, types.ModuleType):          # np.linalg, np.random
            return _DeviceXpy(obj)
        if callable(obj):
            allocator = name in self._ALLOCATORS

            def f(*a, **kw):
                # `xpy.sqrt(host_array)` is the OTHER half of the defect class: a device
                # routine handed a host array.  cupy raises; so must this, or a backend
                # mistake reached through the module namespace scores as a pass.
                if not allocator:
                    _reject_mixed(a)
                return _wrap(obj(*_unwrap(a), **_unwrap(kw)))
            f.__name__ = name
            return f
        return obj                                     # pi, inf, e, newaxis(None)

    def asnumpy(self, a):
        return _unwrap(a)


_DEVICE_XPY = _DeviceXpy()


###
### backend-agnostic probes: they mean the same thing for cupy and for the stand-in
###

def _device(x):
    """Push `x` onto whatever stands for the device here.  Deliberately does NOT go
    through the module's own `_to_backend`: if that were broken, every device-arm
    assertion below would quietly be testing host arrays instead."""
    if REAL_CUPY:
        import cupy
        return cupy.asarray(np.asarray(x))
    return _DeviceLike(np.asarray(x))


def _is_device(a):
    """True for a DEVICE ARRAY -- the real cupy one, or the stand-in.  Decided without
    consulting the module under test, and False for a plain python/numpy scalar, which
    neither backend wraps."""
    if REAL_CUPY:
        import cupy
        return isinstance(a, cupy.ndarray)
    return isinstance(a, _DeviceLike)


def _host(a):
    return a.get() if hasattr(a, 'get') else np.asarray(a)


@pytest.fixture
def on_a_device(monkeypatch, request):
    """Make the module believe it has cupy, with the stand-in as its array type.

    This is the whole point of the suite: the defect is INVISIBLE where cupy is absent,
    because `xpy_default is np` there and allocating from it is accidentally right.  All
    five globals the pre-fix code read are patched, so a revert reproduces the GPU host's
    failure here rather than passing by luck.
    """
    # Both backends draw their EM initialization and truncnorm samples from a GLOBAL
    # generator, so an unseeded suite depends on ambient state.  Seed it, but put it back
    # afterwards: seeding and walking away is not a fix, it is the same leak with a
    # different value.  Measured -- seeding cupy's global generator and leaving it made
    # test_seeding_public_paths.py::test_skymap_oracle_draws_do_not_depend_on_string_hash
    # _salt fail later in the same pytest process, on a GPU host only.
    _np_state = np.random.get_state()
    np.random.seed(20260915)
    request.addfinalizer(lambda: np.random.set_state(_np_state))
    if REAL_CUPY:
        import cupy
        # seeding mutates the live RandomState in place, so swap in a fresh one and put
        # the original object back untouched
        _cp_state = cupy.random.get_random_state()
        cupy.random.set_random_state(cupy.random.RandomState(20260915))
        request.addfinalizer(lambda: cupy.random.set_random_state(_cp_state))
        return          # the real backend is already what we would be simulating
    monkeypatch.setattr(GMM, 'xpy_default', _DEVICE_XPY)
    monkeypatch.setattr(GMM, 'cupy_ok', True)
    # _DEVICE_XPY, not a bare namespace: code that reads the module-level `cupy` directly
    # (the pre-fix _xpy_logsumexp calls cupy.asarray/amax/where/...) must meet a working
    # backend, or a revert fails with AttributeError and scores as caught for the wrong
    # reason.
    monkeypatch.setattr(GMM, 'cupy', _DEVICE_XPY, raising=False)
    monkeypatch.setattr(GMM, 'identity_convert', _host)
    monkeypatch.setattr(GMM, 'identity_convert_togpu',
                        lambda x: x if isinstance(x, _DeviceLike) else _DeviceLike(x))


def _cloud(n=300, seed=5):
    rng = np.random.default_rng(seed)
    return 0.5 + 0.15 * rng.normal(size=(n, 2))


# ASYMMETRIC on purpose, and not its own transpose.  With [[0,1],[0,1]] the bounds array
# equals its transpose and both dimensions share limits, so dropping BOTH transposes from
# score()'s `_to_host(self._normalize(_to_host(self.bounds).T).T)` -- a line this change
# rewrote -- survives every assertion.
BOUNDS = np.array([[0.0, 1.0], [-2.0, 3.0]])


###
### 0. the stand-in itself -- if this stops behaving like a device the suite proves nothing
###

def _the_backend_under_test():
    """The real cupy where there is one, the stand-in otherwise.  The tests below assert
    the SAME things of both, which is the only way the stand-in's fidelity gets CHECKED
    rather than asserted: written against the stand-in alone they would just be the file
    agreeing with itself."""
    if REAL_CUPY:
        import cupy
        return cupy.asarray(np.arange(4.0)), cupy
    return _DeviceLike(np.arange(4.0)), _DEVICE_XPY


def test_the_backend_refuses_implicit_conversion_and_converts_through_get():
    d, _ = _the_backend_under_test()
    with pytest.raises(TypeError):
        np.asarray(d)
    assert np.array_equal(_host(d), np.arange(4.0))
    # numpy DISPATCHES to it: the module calls bare np.* on arrays that are on the device
    assert float(np.max(d)) == 3.0
    assert np.array_equal(_host(np.sort(d)), np.arange(4.0))


def test_the_backend_refuses_to_compute_alongside_a_host_array():
    """One half of the defect class: a device routine handed a host array.  Both spellings
    matter -- the operator, and the call through the module namespace."""
    d, xpy = _the_backend_under_test()
    host = np.arange(4.0)
    with pytest.raises(TypeError):
        d + host
    with pytest.raises(TypeError):
        host + d
    with pytest.raises(TypeError):
        xpy.sqrt(host)
    with pytest.raises(TypeError):
        xpy.outer(host, host)
    # scalars are fine on both, which is why the rule is ndim>0 and not "is a numpy thing"
    assert np.array_equal(_host(d * 2.0), host * 2.0)
    assert np.array_equal(_host(d * np.float64(2.0)), host * 2.0)


def test_the_backend_refuses_a_host_array_written_into_a_device_slice():
    """The OTHER half, and the defect's own mechanism: `out = xpy.empty(...);
    out[:, i] = <numpy>` is exactly what _normalize did.  cupy routes it through fill."""
    d, xpy = _the_backend_under_test()
    out = xpy.empty((4, 2))
    with pytest.raises(ValueError) as e:
        out[:, 0] = np.arange(4.0)
    assert _FILL_MESSAGE in str(e.value)
    out[:, 0] = d                                  # a device array is accepted
    out[:, 1] = 0.0                                # and so is a scalar


###
### 1. THE DEFECT: host arrays must work on a device-capable host
###

def test_fit_accepts_host_samples_on_a_device_capable_host(on_a_device):
    """Reverting `_normalize` to `self.xpy.empty` raises
    ValueError('non-scalar numpy.ndarray cannot be used for fill') here."""
    model = GMM.gmm(3, BOUNDS)
    model.fit(_cloud())
    assert model.xpy is np, 'fit() must bind the backend of the samples it was given'
    assert all(isinstance(m, np.ndarray) for m in model.means)
    assert all(isinstance(c, np.ndarray) for c in model.covariances)
    assert isinstance(model.weights, np.ndarray)
    assert np.all(np.isfinite(np.concatenate([m.ravel() for m in model.means])))


def test_score_returns_a_host_array_for_host_samples(on_a_device):
    """The half a device-first retry could not have fixed: even where `_normalize`
    survived, `scores = self.xpy.zeros(n)` handed a host caller a DEVICE array back."""
    X = _cloud()
    model = GMM.gmm(2, BOUNDS)
    model.fit(X)
    s = model.score(X)
    assert isinstance(s, np.ndarray), 'host samples in, host scores out'
    assert s.shape == (len(X),)
    # `> 0` could never fail: score() ends in xpy.maximum(scores, 1e-300).  The floor
    # itself is the interesting failure, so exclude it.
    assert np.all(s > 1e-300), 'every sample sits on the 1e-300 floor'
    assert np.all(np.isfinite(s))


def test_update_accepts_host_samples(on_a_device):
    X = _cloud()
    model = GMM.gmm(2, BOUNDS)
    model.fit(X)
    model.update(_cloud(seed=6), log_sample_weights=np.zeros(300))
    assert model.xpy is np
    assert all(isinstance(m, np.ndarray) for m in model.means)
    assert np.all(np.isfinite(np.asarray(model.score(X))))


def test_sample_follows_the_models_own_parameters(on_a_device):
    """`sample()` has no argument to read a backend off, so it reads the model.  A model
    configured with host parameters -- which is how test_gmm_truncated_score builds one,
    without ever calling fit -- must return host draws.  Reverting this to
    `identity_convert_togpu` returns a device array to a caller holding numpy."""
    model = GMM.gmm(2, np.array([[0.0, 1.0]]))
    model.d = 1
    model.means = [np.array([0.0]), np.array([0.4])]
    model.covariances = [np.array([[0.04]]), np.array([[0.09]])]
    model.weights = np.array([0.5, 0.5])
    draws = model.sample(200)
    assert isinstance(draws, np.ndarray), 'host parameters must draw host samples'
    assert draws.shape == (200, 1)
    assert np.all((draws >= 0.0) & (draws <= 1.0))


def test_the_adaptive_fit_and_the_defensive_component_stay_on_the_host(on_a_device):
    model = GMM.fit_gmm_adaptive(_cloud(), BOUNDS, k_max=3, defensive_frac=0.05)
    assert all(isinstance(m, np.ndarray) for m in model.means)
    assert isinstance(model.weights, np.ndarray)
    assert model.defensive_frac == 0.05
    assert isinstance(model.score(_cloud(seed=7)), np.ndarray)


def test_logsumexp_dispatches_on_its_argument(on_a_device):
    """It read module-level cupy_ok, so on a GPU host it pushed a host array onto the
    device and returned a device result to a caller that had asked for neither."""
    a = np.log(np.arange(1.0, 6.0))
    out = GMM._xpy_logsumexp(a)
    assert not _is_device(out)
    np.testing.assert_allclose(float(out), np.log(np.sum(np.exp(a))), rtol=1e-12)
    rows = GMM._xpy_logsumexp(np.stack([a, a + 1.0]), axis=1)
    assert isinstance(rows, np.ndarray) and not _is_device(rows)
    # and a DEVICE argument must still come back on the device
    dev = GMM._xpy_logsumexp(_device(a))
    # both halves asserted separately: `A or B` with A always true on this arm made the
    # shape half dead, and dropping the device branch's reshape(()) survived.
    assert np.ndim(_host(dev)) == 0, 'axis=None must reduce to a scalar on either backend'
    np.testing.assert_allclose(float(_host(dev)), float(out), rtol=1e-12)


###
### 2. the control: the fix must NOT have simply forced everything onto the host
###

def test_a_device_fitted_model_stays_on_the_device(on_a_device):
    """The production ILE path: MonteCarloEnsemble._sample writes `model.sample(n)`
    straight into a `self.xpy` array, so a device-fitted model must keep returning device
    arrays.  A 'fix' that just replaced self.xpy with numpy would pass every test above
    and break this one -- and break the GPU run."""
    Xd = _device(_cloud())
    Bd = _device(BOUNDS)
    assert _is_device(Xd)
    model = GMM.gmm(2, Bd)
    model.fit(Xd)
    assert model.xpy is GMM.xpy_default, 'device samples must bind the device backend'
    assert all(_is_device(m) for m in model.means)
    assert _is_device(model.score(Xd))
    assert _is_device(model.sample(100))
    model.update(_device(_cloud(seed=6)))
    assert all(_is_device(m) for m in model.means)
    GMM.add_defensive_component(model, defensive_frac=0.05)
    assert _is_device(model.means[-1]) and _is_device(model.weights)


def test_the_adaptive_fit_stays_on_the_device(on_a_device):
    Xd = _device(_cloud())
    model = GMM.fit_gmm_adaptive(Xd, _device(BOUNDS), k_max=3, defensive_frac=0.05)
    assert all(_is_device(m) for m in model.means)
    assert _is_device(model.score(Xd))


###
### 3. the two backends must agree, which is the point of having both
###

def _fixed_model(bounds, means, covs, weights):
    """A 2-component mixture with the parameters PINNED, so nothing here depends on the
    EM's random initialization -- which is drawn from `self.xpy.random` and so is a
    different stream on each backend by construction."""
    m = GMM.gmm(2, bounds)
    m.d = 2
    m.means = list(means)
    m.covariances = list(covs)
    m.weights = weights
    return m


def test_host_and_device_score_the_same_mixture_alike(on_a_device):
    """The same mixture, the same points, both backends.  This is what stops the backend
    inference from silently selecting a DIFFERENT numerical path: the gpu_logpdf-vs-scipy
    dispatch under score() keys on the resolved backend, so getting the backend wrong and
    getting the density wrong are the same mistake."""
    X = _cloud(seed=11)
    means = [np.array([-0.2, 0.1]), np.array([0.3, -0.4])]
    covs = [np.array([[0.09, 0.01], [0.01, 0.16]]),
            np.array([[0.25, -0.05], [-0.05, 0.12]])]
    w = np.array([0.4, 0.6])

    m_host = _fixed_model(BOUNDS, means, covs, w)
    m_dev = _fixed_model(_device(BOUNDS),
                         [_device(m) for m in means], [_device(c) for c in covs],
                         _device(w))

    s_host = m_host.score(X)
    s_dev = m_dev.score(_device(X))
    assert not _is_device(s_host) and _is_device(s_dev)
    np.testing.assert_allclose(_host(s_host), _host(s_dev), rtol=1e-8)

    # the coordinate transform itself, which is where the defect lived
    np.testing.assert_allclose(_host(m_host._normalize(X)),
                               _host(m_dev._normalize(_device(X))), rtol=1e-12)
    np.testing.assert_allclose(_host(m_host._unnormalize(m_host._normalize(X))),
                               X, rtol=1e-12)


###
### 4. models built by ASSIGNMENT rather than by fit()
###
### These are where most of the remaining module-global reads were observable.  A fitted
### model agrees with itself -- fit() binds self.xpy from the samples, so self.xpy and the
### parameters' own backend say the same thing and reading either works.  A model built by
### assignment does not: self.xpy is still xpy_default (the device) while every parameter
### is host numpy.  Two places in the tree build exactly that: test_gmm_truncated_score.py,
### and any hand-built model that assigns host means,
### covariances and weights onto a fresh gmm.
###
### The converse shape -- a model whose parameters are all on the DEVICE -- is what
### calmarg.extrinsic_handoff.reconstruct_gmm builds, and it must stay that way: see the
### device-arm tests above, and that function's docstring for what silently breaks if it
### does not.
###

def _hand_built_host_model():
    m = GMM.gmm(2, BOUNDS)
    m.d = 2
    m.N = 400
    m.means = [np.array([-0.2, 0.1]), np.array([0.3, -0.4])]
    m.covariances = [np.array([[0.09, 0.01], [0.01, 0.16]]),
                     np.array([[0.25, -0.05], [-0.05, 0.12]])]
    m.weights = np.array([0.4, 0.6])
    m.adapt = [True, True]
    assert m.xpy is GMM.xpy_default, 'nothing has told this model which backend it is on'
    return m


def test_a_hand_built_host_model_updates_on_the_host(on_a_device):
    m = _hand_built_host_model()
    X = _cloud(seed=8)
    m.update(X, log_sample_weights=np.zeros(len(X)))
    assert m.xpy is np
    assert all(isinstance(mu, np.ndarray) for mu in m.means), 'the merge left the host'
    assert all(isinstance(c, np.ndarray) for c in m.covariances)
    assert isinstance(m.weights, np.ndarray)
    assert np.all(np.isfinite(m.score(X)))


def test_a_hand_built_host_model_prunes_on_the_host(on_a_device):
    m = _hand_built_host_model()
    m.weights = np.array([0.9998, 0.0002])
    m.prune_components(weight_floor=1e-3)
    assert m.k == 1
    assert isinstance(m.weights, np.ndarray)
    assert all(isinstance(mu, np.ndarray) for mu in m.means)


def test_a_hand_built_host_model_takes_a_host_defensive_component(on_a_device):
    m = _hand_built_host_model()
    GMM.add_defensive_component(m, defensive_frac=0.05)
    assert m.k == 3 and m.defensive_frac == 0.05
    assert all(isinstance(mu, np.ndarray) for mu in m.means)
    assert isinstance(m.weights, np.ndarray)
    # and the round trip back off again
    m._strip_defensive_component()
    assert m.k == 2 and all(isinstance(mu, np.ndarray) for mu in m.means)


def test_the_mixture_log_density_reads_its_argument(on_a_device):
    """_mixture_log_density_normalized drives the BIC ladder in fit_gmm_adaptive.  It took
    its backend from `model.xpy`, which for an unfitted model is whatever hardware the host
    happens to have."""
    m = _hand_built_host_model()
    X = _cloud(seed=9)
    out = GMM._mixture_log_density_normalized(m, m._normalize(X))
    assert isinstance(out, np.ndarray) and not _is_device(out)
    assert out.shape == (len(X),) and np.all(np.isfinite(out))


def test_the_instance_converters_follow_the_bound_backend(on_a_device):
    """`identity_convert_togpu` is part of this class's surface, and it was the module
    global: a host-fitted model would have pushed anything handed to it onto the device."""
    m = GMM.gmm(2, BOUNDS)
    m.fit(_cloud())
    assert not _is_device(m.identity_convert_togpu(np.zeros(3)))
    assert not _is_device(m.identity_convert(np.zeros(3)))

    md = GMM.gmm(2, _device(BOUNDS))
    md.fit(_device(_cloud()))
    assert _is_device(md.identity_convert_togpu(np.zeros(3)))
    assert not _is_device(md.identity_convert(md.means[0]))

    # `estimator` carries the same two attributes and the same contract; nothing inside
    # the module reads them any more, which is exactly why they would rot unwatched.
    e = GMM.estimator(2)
    e.fit(_cloud(), None)
    assert not _is_device(e.identity_convert_togpu(np.zeros(3)))
    ed = GMM.estimator(2)
    ed.fit(_device(_cloud()), None)
    assert _is_device(ed.identity_convert_togpu(np.zeros(3)))


def test_the_density_routine_is_chosen_by_the_resolved_backend(on_a_device, monkeypatch):
    """gpu_logpdf and scipy's logpdf are not interchangeable -- scipy takes
    allow_singular=True, gpu_logpdf adds an epsilon ridge and solves -- so which one runs
    has to follow the backend that was actually resolved.  Keyed on module-level cupy_ok,
    a HOST fit on a GPU host silently took the device routine; keyed the other way, scipy
    would be handed a device array.  A call-site spy is the only way to see this: both
    routines return the same numbers on the cases that work."""
    calls = {'n': 0}
    real = GMM.gpu_logpdf

    def spy(x, mean, cov, xpy):
        calls['n'] += 1
        assert xpy is not np, 'gpu_logpdf was selected for the HOST backend'
        assert _is_device(x), 'gpu_logpdf was handed a host array'
        return real(x, mean, cov, xpy)

    monkeypatch.setattr(GMM, 'gpu_logpdf', spy)

    X = _cloud()
    m = GMM.gmm(2, BOUNDS)
    m.fit(X)
    m.score(X)
    GMM._mixture_log_density_normalized(m, m._normalize(X))
    assert calls['n'] == 0, 'a host fit/score must not go through gpu_logpdf'

    Xd = _device(X)
    md = GMM.gmm(2, _device(BOUNDS))
    md.fit(Xd)
    md.score(Xd)
    assert calls['n'] > 0, 'a device fit/score must go through gpu_logpdf'


def _three_modes(n=1200, seed=17):
    """A cloud that visibly earns more than one component, so the BIC ladder has something
    to find and a test on the chosen k is not vacuous."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.2, 0.25], [0.75, 0.3], [0.5, 0.8]])
    which = rng.integers(0, 3, size=n)
    return np.clip(centres[which] + 0.04 * rng.normal(size=(n, 2)), 0.001, 0.999)


def test_the_adaptive_fit_chooses_the_same_k_on_both_backends(on_a_device):
    """fit_gmm_adaptive picks its component count from a BIC ladder in which EVERY candidate
    fit is wrapped in `except Exception: continue`, with a silent fall back to k_min if all
    of them fail.  So a backend mistake inside the ladder does not raise -- it quietly
    returns the floor component count and a worse proposal.

    Reading the backend off the module global instead of off the samples does exactly that,
    and only where the two backends differ: `wn_scaled` stays on the device while `logmix`
    comes back on the samples' backend, and the product of the two is the mixed-backend
    operation cupy refuses.  Asserting only that the fit SURVIVES misses it entirely -- it
    does survive, at k_min.  The observable is whether the ladder collapsed to the floor.

    What this deliberately does NOT assert is that the two backends pick the SAME k.  The
    EM initialization draws from `self.xpy.random`, so the two arms run different random
    streams by construction (cupy keeps its own generator), and the BIC optimum moves with
    the initialization.  Pinning equality would be pinning the RNG, and it fails on the
    unmutated code: measured k=4 host, k=6 device.
    """
    X = _three_modes()
    k_min = 1
    k_host = GMM.fit_gmm_adaptive(X, BOUNDS, k_max=8, k_min=k_min, defensive_frac=0.0).k
    assert k_host > k_min, 'this cloud must earn more than the floor, or the test is vacuous'
    k_dev = GMM.fit_gmm_adaptive(_device(X), _device(BOUNDS), k_max=8, k_min=k_min,
                                 defensive_frac=0.0).k
    assert k_dev > k_min, (
        'the adaptive ladder fell back to k_min={} on the device while the host earned '
        'k={}: every candidate fit raised and `except Exception: continue` swallowed it, '
        'which is what a backend read off the module global does here'.format(k_min, k_host))


def test_update_keeps_weights_floating_point(on_a_device):
    """`self.weights` is not always the float array fit() leaves behind.
    A hand-built model may assign a PYTHON LIST of ints, and _merge writes float weights back
    into whatever container it finds.  _merge's own np.asarray would coerce that list to
    dtype int64 without the explicit dtype=float, so `self.weights[i] = weight` truncates
    every merged weight to 0 and score() returns its 1e-300 floor for every sample -- no
    exception, no warning, just a proposal density of zero."""
    m = GMM.gmm(2, BOUNDS)
    m.d, m.N = 2, 200
    m.means = [np.array([0.3, 0.3]), np.array([-0.2, 0.1])]
    m.covariances = [np.eye(2) * 0.05, np.eye(2) * 0.08]
    m.weights = [1, 0]          # the shape that breaks it: a list of INTs
    m.adapt = [True, True]

    X = _cloud(seed=21)
    m.update(X)

    w = np.asarray(_host(m.weights), dtype=float)
    assert np.all(np.isfinite(w)) and w.sum() > 0, 'merged weights collapsed to zero'
    assert w.max() < 1.0 and w.min() > 0.0, 'both components should carry weight: %r' % (w,)
    s = np.asarray(_host(m.score(X[:5])))
    assert np.all(s > 1e-300), 'score() fell to its floor: %r' % (s,)


def test_update_does_not_change_the_means_container_type(on_a_device):
    """fit() leaves `means` as the (k,d) array the estimator built; consumers index it both
    ways.  Converting it unconditionally in _merge turned it into a list of (d,) arrays on
    every update -- same numbers, different type, no diff line to notice it by."""
    X = _cloud(seed=22)
    m = GMM.gmm(2, BOUNDS)
    m.fit(X)
    before = type(m.means)
    m.update(_cloud(seed=23))
    assert type(m.means) is before, (
        'means went from {} to {} across update()'.format(before.__name__,
                                                          type(m.means).__name__))


###
### 5. MIXED backends: a model on one, the arrays handed to it on the other
###
### Every test above keeps the model, its bounds and the samples on a single backend.  The
### comments in the library claim more than that -- score() says it brings the parameters
### across "if they were fitted on the other one", _merge says "a device-fitted model
### refitted on host samples, say", fit() says "a host fit cannot be handed device weights
### half way through".  Those are the shapes below.  They are also the shapes that decide
### whether the conversions are real or decorative.
###

def test_a_device_fitted_model_scores_host_samples(on_a_device):
    """score() is a query: it answers on the backend of the samples it is ASKED about, and
    brings the mixture across.  Dropping either `_to_backend` in that loop hands a device
    array to scipy, or mixes the two in `scores += w * pdf`."""
    X = _cloud()
    m = GMM.gmm(2, _device(BOUNDS))
    m.fit(_device(X))
    s = m.score(X)                      # HOST samples, device model
    assert not _is_device(s), 'host samples in, host scores out'
    np.testing.assert_allclose(_host(s), _host(m.score(_device(X))), rtol=1e-8)


def test_a_host_fitted_model_scores_device_samples(on_a_device):
    X = _cloud()
    m = GMM.gmm(2, BOUNDS)
    m.fit(X)
    s = m.score(_device(X))             # DEVICE samples, host model
    assert _is_device(s), 'device samples in, device scores out'
    np.testing.assert_allclose(_host(s), _host(m.score(X)), rtol=1e-8)


def test_a_device_fitted_model_updates_from_host_samples(on_a_device):
    """_merge's own comment names this case.  The refit decides the backend, and the old
    parameters are brought across to meet it."""
    m = GMM.gmm(2, _device(BOUNDS))
    m.fit(_device(_cloud()))
    m.update(_cloud(seed=31))           # HOST refit of a device model
    assert all(not _is_device(mu) for mu in m.means)
    assert np.all(np.isfinite(np.asarray(_host(m.score(_cloud(seed=32))))))


def test_a_half_converted_model_still_scores_and_samples(on_a_device):
    """Device means/covariances/weights with HOST bounds.  This is the exact state
    calmarg.extrinsic_handoff.reconstruct_gmm's docstring now says no longer crashes, so it
    needs a test rather than a claim.  _normalize brings the bounds across per call."""
    m = GMM.gmm(2, BOUNDS)              # bounds left on the HOST
    m.d = 2
    m.means = [_device(np.array([-0.2, 0.1])), _device(np.array([0.3, -0.4]))]
    m.covariances = [_device(np.eye(2) * 0.09), _device(np.eye(2) * 0.16)]
    m.weights = _device(np.array([0.4, 0.6]))
    m.adapt = [True, True]
    s = m.score(_device(_cloud()))
    assert _is_device(s) and np.all(np.isfinite(np.asarray(_host(s))))
    assert _is_device(m.sample(32)), 'the parameters are on the device, so the draws are'


def test_fit_accepts_weights_on_the_other_backend(on_a_device):
    """fit()'s own comment: 'a host fit cannot be handed device weights half way through'."""
    X = _cloud()
    m = GMM.gmm(2, BOUNDS)
    m.fit(X, log_sample_weights=_device(np.zeros(len(X))))
    assert m.xpy is np and all(not _is_device(mu) for mu in m.means)

    md = GMM.gmm(2, _device(BOUNDS))
    md.fit(_device(X), log_sample_weights=np.zeros(len(X)))
    assert all(_is_device(mu) for mu in md.means)


def test_the_adaptive_fit_accepts_weights_on_the_other_backend(on_a_device):
    X = _three_modes()
    m = GMM.fit_gmm_adaptive(X, BOUNDS, k_max=4, defensive_frac=0.0,
                             log_sample_weights=_device(np.zeros(len(X))))
    assert all(not _is_device(mu) for mu in m.means)


def test_the_adaptive_fit_inflates_on_the_models_own_backend(on_a_device):
    """The `inflate` block is a separate backend read from the rest of fit_gmm_adaptive and
    no other test reaches it."""
    X = _three_modes()
    plain = GMM.fit_gmm_adaptive(_device(X), _device(BOUNDS), k_max=3, defensive_frac=0.0,
                                 inflate=1.0)
    wide = GMM.fit_gmm_adaptive(_device(X), _device(BOUNDS), k_max=3, defensive_frac=0.0,
                                inflate=2.0)
    assert all(_is_device(c) for c in wide.covariances)
    tr_plain = sum(float(np.trace(_host(c))) for c in plain.covariances) / plain.k
    tr_wide = sum(float(np.trace(_host(c))) for c in wide.covariances) / wide.k
    assert tr_wide > 2.0 * tr_plain, 'inflate=2 must widen the components (4x in variance)'


###
### 6. an ABSOLUTE anchor, on the device arm
###
### Everything above compares one backend to the other, so a numerical mutation that moves
### BOTH identically survives all of it.  test_gmm_truncated_score.py pins score() against
### an independent computation, but only on the host.  This pins the device arm, in d=1,
### where the normalization is a deterministic erf rather than the randomized mvnun.
###

def test_the_device_score_matches_an_independent_computation(on_a_device):
    from scipy.stats import norm as _norm
    bounds = np.array([[0.0, 1.0]])
    mu, sd, w = np.array([0.1]), 0.35, np.array([1.0])

    m = GMM.gmm(1, _device(bounds))
    m.d = 1
    m.means = [_device(mu)]
    m.covariances = [_device(np.array([[sd ** 2]]))]
    m.weights = _device(w)
    m.adapt = [False]

    x = np.array([0.15, 0.4, 0.62, 0.91])
    got = _host(m.score(_device(x[:, None])))

    # independent: normalize to [-1,1], truncated normal density, then the box Jacobian
    xn = 2.0 * x - 1.0
    mass = _norm.cdf((1.0 - mu[0]) / sd) - _norm.cdf((-1.0 - mu[0]) / sd)
    expected = (_norm.pdf(xn, loc=mu[0], scale=sd) / mass) * (2.0 ** 1) / 1.0
    np.testing.assert_allclose(np.asarray(got, dtype=float), expected, rtol=1e-10)


def test_the_score_carries_the_right_box_jacobian_in_more_than_one_dimension(on_a_device):
    """The d==1 anchor above cannot see the box volume: with one dimension prod and sum of
    the widths agree, and so do 2**d and 2.  This pins both in d=2, with UNEQUAL widths.

    The d>1 normalization goes through mvnun, which is randomized at abseps/releps 1e-5, so
    this compares at 1e-3 rather than to roundoff.  That is ample: dropping 2**d is a factor
    4, and prod -> sum on these bounds is a factor 6/5.
    """
    from scipy.stats import multivariate_normal as _mvn, norm as _norm
    bounds = np.array([[0.0, 1.0], [-2.0, 3.0]])     # widths 1 and 5: prod 5, sum 6
    mu = np.array([0.1, -0.2])
    cov = np.diag([0.3 ** 2, 0.45 ** 2])

    m = GMM.gmm(1, _device(bounds))
    m.d = 2
    m.means = [_device(mu)]
    m.covariances = [_device(cov)]
    m.weights = _device(np.array([1.0]))
    m.adapt = [False]

    x = np.array([[0.3, 0.5], [0.6, -0.4], [0.45, 1.2]])
    got = np.asarray(_host(m.score(_device(x))), dtype=float)

    # independent: normalize each column to [-1,1], separable truncated normal, box Jacobian
    xn = np.empty_like(x)
    for i, (lo, hi) in enumerate(bounds):
        xn[:, i] = (2.0 * x[:, i] - (hi + lo)) / (hi - lo)
    mass = 1.0
    for i in range(2):
        sd = np.sqrt(cov[i, i])
        mass *= _norm.cdf((1.0 - mu[i]) / sd) - _norm.cdf((-1.0 - mu[i]) / sd)
    vol = np.prod(bounds[:, 1] - bounds[:, 0])
    expected = _mvn.pdf(xn, mean=mu, cov=cov) / mass * (2.0 ** 2) / vol

    np.testing.assert_allclose(got, expected, rtol=1e-3)


def test_the_adaptive_fit_inflates_on_the_host_too(on_a_device):
    """The device arm of this is above.  The `inflate` block reads the backend separately
    from the rest of fit_gmm_adaptive, so reading a module global there is invisible
    whenever the samples happen to be on the device."""
    X = _three_modes()
    plain = GMM.fit_gmm_adaptive(X, BOUNDS, k_max=3, defensive_frac=0.0, inflate=1.0)
    wide = GMM.fit_gmm_adaptive(X, BOUNDS, k_max=3, defensive_frac=0.0, inflate=2.0)
    assert all(not _is_device(c) for c in wide.covariances)
    tr_plain = sum(float(np.trace(_host(c))) for c in plain.covariances) / plain.k
    tr_wide = sum(float(np.trace(_host(c))) for c in wide.covariances) / wide.k
    assert tr_wide > 2.0 * tr_plain, 'inflate=2 must widen the components on the host too'
